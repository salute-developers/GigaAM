from dataclasses import replace
from typing import Dict, List, Optional, Tuple, Union, cast

import hydra
import omegaconf
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .decoder import CTCHead, DualASRExport
from .preprocess import SAMPLE_RATE, load_audio
from .timestamps_utils import (
    compute_frame_shift,
    confidence_from_logprobs,
    frames_to_words,
)
from .types import LongformTranscriptionResult, Segment, TranscriptionResult
from .utils import (
    AudioDataset,
    _export_advance,
    _export_heads,
    _export_streaming,
    _save_onnx_config,
    export_onnx,
    onnx_converter,
)

LONGFORM_THRESHOLD = 30 * SAMPLE_RATE


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.language_fusion: Optional[nn.Module] = None
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)
        self.encoder.frame_shift_ms = (
            1000
            * self.preprocessor.hop_length
            * self.encoder.subsampling_factor
            / self.cfg.preprocessor.sample_rate
        )

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        language=None,
        *,
        future_context_ms=None,
    ):
        """
        Perform forward pass through the preprocessor and encoder.
        """
        embedding = self.encode_language(language, features.shape[0])
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        kwargs = {"lang_embedding": embedding} if embedding is not None else {}
        if future_context_ms:
            context = {
                int(c.future_context_ms): c for c in self.cfg.streaming.contexts
            }[future_context_ms]
            frame_ms = self.encoder.frame_shift_ms
            kwargs.update(
                chunk_size=int(future_context_ms / frame_ms),
                left_context=int(context.left_context),
            )
        with torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=self._device.type == "cuda"
            and self.cfg.get("encoder_autocast_dtype") != "float32",
        ):
            encoded, lengths = self.encoder(features, feature_lengths, **kwargs)
        if self.language_fusion is not None:
            encoded = self.language_fusion(encoded, embedding)
        return encoded, lengths

    def encode_language(self, language, batch_size):
        return None

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(
        self, wav_file: str, max_samples: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare audio on the model device, optionally rejecting overlong input first.
        """
        wav = load_audio(wav_file)
        if max_samples is not None and wav.numel() > max_samples:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")
        wav = wav.to(device=self._device, dtype=torch.float32).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length

    def embed_audio(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(
        self,
        dir_path=".",
        dtype=torch.float32,
        future_context_ms=None,
        streaming=False,
        dynamic_context=None,
    ):
        """
        Export onnx model encoder to the specified dir.
        """
        return export_onnx(
            self, dir_path, dtype, future_context_ms, streaming, dynamic_context
        )

    def _to_onnx(
        self, dir_path: str = ".", dtype: Optional[torch.dtype] = torch.float32
    ) -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        encoder = self.encoder
        module = self if encoder.export_head or hasattr(self, "heads") else encoder
        saved_forward = module.forward
        module.forward = module.forward_for_export  # type: ignore[method-assign]
        name = self.cfg.model_name
        try:
            with encoder.onnx_export_mode():
                onnx_converter(
                    name if encoder.export_head else f"{name}_encoder",
                    module,
                    dir_path,
                    inputs=(
                        encoder.input_example(batch_size=2, seqlen=96)
                        if hasattr(self, "heads")
                        else encoder.input_example()
                    ),
                    input_names=encoder.input_names(),
                    output_names=encoder.output_names(),
                    dynamic_axes=encoder.dynamic_axes(),
                    export_dtype=dtype,
                )
        finally:
            module.forward = saved_forward  # type: ignore[method-assign]


class _LongformASR(GigaAM):
    def _decode(
        self, encoded, encoded_len, wav_lens, word_timestamps, return_confidence
    ):
        raise NotImplementedError

    def _decode_head(
        self,
        head,
        decoding,
        encoded,
        encoded_lengths,
        wav_lengths=None,
        word_timestamps=False,
        return_confidence=False,
    ) -> List[TranscriptionResult]:
        parameter = next(head.parameters(), None)
        if parameter is not None:
            encoded = encoded.to(parameter.dtype)
        scores: Optional[List[List[float]]] = [] if return_confidence else None
        kwargs = {"token_logprobs": scores} if return_confidence else {}
        decoded = decoding.decode(head, encoded, encoded_lengths, **kwargs)
        results = []
        for i, (text, tokens, frames) in enumerate(decoded):
            logprobs = scores[i] if scores is not None else None
            words = None
            if word_timestamps:
                shift = compute_frame_shift(
                    int(wav_lengths[i]), int(encoded_lengths[i])
                )
                words = frames_to_words(
                    decoding.tokenizer, tokens, frames, shift, logprobs
                )
            results.append(
                TranscriptionResult(
                    text=text,
                    words=words,
                    confidence=confidence_from_logprobs(logprobs),
                )
            )
        return results

    @torch.inference_mode()
    def transcribe_longform(
        self,
        wav_file: str,
        word_timestamps: bool = False,
        fr_batch_size: int = 16,
        fr_num_workers: int = 0,
        language=None,
        future_context_ms=None,
        *,
        return_confidence: bool = False,
        merge_gap: float = 0.7,
        pad: float = 0.2,
        strict_limit_duration: float = 30.0,
        vad_batch_size: int = 32,
        vad_model_path: Optional[str] = None,
        vad=None,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.1,
        **kwargs,
    ) -> Union[LongformTranscriptionResult, Dict[str, LongformTranscriptionResult]]:
        """
        Transcribes a long audio file by splitting it into segments and
        then transcribing each segment (batched inference via AudioDataset).
        Use fr_batch_size and fr_num_workers to control the batched inference.
        Use return_confidence for segment scores and, with word_timestamps, word scores.
        """
        from .vad_utils import segment_audio_file

        segments, boundaries = segment_audio_file(
            wav_file,
            SAMPLE_RATE,
            merge_gap=merge_gap,
            pad=pad,
            strict_limit_duration=strict_limit_duration,
            device=self._device,
            vad_batch_size=vad_batch_size,
            vad_model_path=vad_model_path,
            vad=vad,
            threshold=threshold,
            neg_threshold=neg_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            **kwargs,
        )

        result_segments: Dict[str, List[Segment]] = (
            {name: [] for name in cast(nn.ModuleDict, self.heads)}
            if hasattr(self, "heads")
            else {"": []}
        )
        if not segments:
            results = {
                name: LongformTranscriptionResult(segments=items)
                for name, items in result_segments.items()
            }
            return results[""] if "" in results else results

        ds = AudioDataset(segments, tokenizer=None)
        dl = DataLoader(
            ds,
            batch_size=fr_batch_size,
            shuffle=False,
            collate_fn=AudioDataset.collate,
            num_workers=fr_num_workers,
        )

        idx = 0
        for wav_pad, wav_lens in dl:
            wav_pad = wav_pad.to(device=self._device, dtype=torch.float32)
            wav_lens = wav_lens.to(self._device)
            encoded, encoded_len = self.forward(
                wav_pad, wav_lens, language, future_context_ms=future_context_ms
            )
            decoded = self._decode(
                encoded, encoded_len, wav_lens, word_timestamps, return_confidence
            )
            batches = decoded if isinstance(decoded, dict) else {"": decoded}
            for name, items in batches.items():
                for offset, item in enumerate(items):
                    seg_start, seg_end = boundaries[idx + offset]
                    word_segments = (
                        [
                            replace(
                                w,
                                start=round(w.start + seg_start, 3),
                                end=round(w.end + seg_start, 3),
                            )
                            for w in item.words or []
                        ]
                        if word_timestamps
                        else None
                    )
                    result_segments[name].append(
                        Segment(
                            text=item.text,
                            confidence=item.confidence,
                            start=seg_start,
                            end=seg_end,
                            words=word_segments,
                        )
                    )
            idx += wav_pad.shape[0]
        results = {
            name: LongformTranscriptionResult(segments=items)
            for name, items in result_segments.items()
        }
        return results[""] if "" in results else results


class GigaAMASR(_LongformASR):
    """
    Giga Acoustic Model for Speech Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    def _decode(
        self,
        encoded: Tensor,
        encoded_len: Tensor,
        wav_lens: Tensor,
        word_timestamps: bool = False,
        return_confidence: bool = False,
    ) -> List[TranscriptionResult]:
        return self._decode_head(
            self.head,
            self.decoding,
            encoded,
            encoded_len,
            wav_lens,
            word_timestamps,
            return_confidence,
        )

    @torch.inference_mode()
    def transcribe(
        self,
        wav_file: str,
        word_timestamps: bool = False,
        *,
        return_confidence: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribes a short audio file into text.
        Returns a transcription with optional word timestamps and confidence scores.
        """
        wav, length = self.prepare_wav(wav_file, max_samples=LONGFORM_THRESHOLD)

        encoded, encoded_len = self.forward(wav, length)
        return self._decode(
            encoded, encoded_len, length, word_timestamps, return_confidence
        )[0]

    def forward_for_export(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, encoded_len = self.encoder(features, feature_lengths)
        return self.head(encoded), encoded_len

    def _to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Export onnx ASR model.
        `ctc`:  exported entirely in encoder-decoder format.
        `rnnt`: exported in encoder/decoder/joint parts separately.
        """
        self.encoder.export_head = isinstance(self.head, CTCHead)
        super()._to_onnx(dir_path, dtype)
        if hasattr(self.head, "decoder"):
            for part in ("decoder", "joint"):
                module = getattr(self.head, part)
                onnx_converter(
                    f"{self.cfg.model_name}_{part}",
                    module,
                    dir_path,
                    dynamic_axes=module.dynamic_axes(),
                    export_dtype=dtype,
                )


class GigaAMDualASR(_LongformASR):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.heads = nn.ModuleDict(
            {name: hydra.utils.instantiate(cfg.heads[name]) for name in ("e2e", "raw")}
        )
        self.language_embedding = self.language_fusion = None
        self.default_language = None
        self._initialize_language_conditioning(cfg)
        self.decodings = {
            name: hydra.utils.instantiate(cfg.decodings[name]) for name in self.heads
        }

    def _initialize_language_conditioning(self, cfg):
        conditioning = cfg.get("language_conditioning")
        if conditioning and (
            conditioning.get("film", {}).get("enabled", False)
            or conditioning.get("late_fusion", {}).get("enabled", False)
        ):
            from .language import LanguageEmbedding, LanguageFiLM, LanguageLateFusion

            self.language_embedding = LanguageEmbedding(conditioning)
            if conditioning.get("film", {}).get("enabled", False):
                self.encoder.language_film = LanguageFiLM(
                    self.encoder.layer_dims,
                    conditioning["embedding_dim"],
                    conditioning["film"],
                )
            if conditioning.get("late_fusion", {}).get("enabled", False):
                self.language_fusion = LanguageLateFusion(
                    cfg.heads.raw.joint.enc_hidden,
                    conditioning["embedding_dim"],
                    conditioning["late_fusion"],
                )

    def forward_for_export(self, features, lengths, *values):
        embedding = None
        if self.language_embedding is not None:
            values = list(values)
            context = self.encoder._onnx_context
            index = (
                2 + int(context[2])
                if isinstance(context, tuple)
                else int(isinstance(context, dict))
            )
            vector = values.pop(index)
            embedding = self.language_embedding.forward_for_export(vector)
        with torch.autocast(
            device_type=features.device.type,
            dtype=torch.float16,
            enabled=features.is_cuda and self._dtype == torch.float16,
        ):
            encoded, *outputs = self.encoder.forward_for_export(
                features, lengths, *values, lang_embedding=embedding
            )
        if self.language_fusion is not None:
            encoded = self.language_fusion(encoded, embedding)
        return encoded.float(), *outputs

    def _to_onnx(
        self,
        dir_path=".",
        dtype=torch.float32,
        future_context_ms=None,
        streaming=False,
        dynamic_context=None,
    ):
        if dynamic_context is None:
            dynamic_context = streaming
        self.encoder._onnx_context = future_context_ms
        self.encoder.export_language_dim = (
            self.language_embedding.weight.shape[0]
            if self.language_embedding is not None
            else 0
        )
        if streaming:
            return _export_streaming(self, dir_path, future_context_ms, dynamic_context)
        if (
            self.cfg.model_name
            in ("v4_ctc", "v4_rnnt", "multilingual_v2_rnnt", "v2_multilingual_rnnt")
            and future_context_ms is None
        ):
            graph = DualASRExport(self)
            example = self.encoder.input_example(batch_size=2, seqlen=96)
            onnx_converter(
                f"{self.cfg.model_name}_encoder",
                graph,
                dir_path,
                example,
                self.encoder.input_names(),
                graph.output_names,
                graph.dynamic_axes(),
                export_dtype=None,
            )
            if not graph.ctc:
                _export_advance(
                    self, dir_path, f"{self.cfg.model_name}_advance", example[0]
                )
            self.cfg.onnx_fc = list(self.heads)
            _save_onnx_config(self, dir_path)
            return
        super()._to_onnx(dir_path, None)
        _export_heads(self, dir_path)
        _save_onnx_config(self, dir_path, future_context_ms)

    def encode_language(self, language, batch_size):
        if self.language_embedding is None:
            return None
        language = self.default_language if language is None else language
        return self.language_embedding(language, batch_size)

    @torch.inference_mode()
    def decode(self, encoded: Tensor, lengths: Tensor) -> Dict[str, List[str]]:
        return {
            name: [item.text for item in items]
            for name, items in self._decode(encoded, lengths, None).items()
        }

    def _decode(
        self,
        encoded,
        encoded_lengths,
        wav_lengths,
        word_timestamps=False,
        return_confidence=False,
    ):
        return {
            name: self._decode_head(
                head,
                self.decodings[name],
                encoded,
                encoded_lengths,
                wav_lengths,
                word_timestamps,
                return_confidence,
            )
            for name, head in self.heads.items()
        }

    @torch.inference_mode()
    def transcribe(
        self,
        wav_file: str,
        word_timestamps: bool = False,
        *,
        language=None,
        future_context_ms=None,
        return_confidence: bool = False,
    ) -> Dict[str, TranscriptionResult]:
        """Transcribe each head independently, optionally with word timestamps and confidence."""
        wav, lengths = self.prepare_wav(wav_file, max_samples=LONGFORM_THRESHOLD)
        encoded, encoded_lengths = self.forward(
            wav, lengths, language, future_context_ms=future_context_ms
        )
        return {
            name: items[0]
            for name, items in self._decode(
                encoded, encoded_lengths, lengths, word_timestamps, return_confidence
            ).items()
        }


class GigaAMEmo(GigaAM):
    """
    Giga Acoustic Model for Emotion Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.id2name = cfg.id2name

    def get_probs(self, wav_file: str) -> Dict[str, float]:
        """
        Calculate probabilities for each emotion class based on the provided audio file.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, _ = self.forward(wav, length)
        encoded_pooled = nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1]
        ).squeeze(-1)

        logits = self.head(encoded_pooled)[0]
        probs = nn.functional.softmax(logits, dim=-1).detach().tolist()

        return {self.id2name[i]: probs[i] for i in range(len(self.id2name))}

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, _ = self.encoder(features, feature_lengths)
        enc_pooled = encoded.mean(dim=-1)
        return nn.functional.softmax(self.head(enc_pooled), dim=-1)

    def _to_onnx(self, dir_path: str = ".", dtype: torch.dtype = torch.float32) -> None:
        """
        Export onnx Emo model.
        """
        self.encoder.export_head = True
        self.encoder.export_emotion = True
        super()._to_onnx(dir_path, dtype)
