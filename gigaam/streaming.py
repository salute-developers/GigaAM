"""Stateful packet inference for mono audio. Call flush() at utterance end."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn

from .timestamps_utils import confidence_from_logprobs, tokens_to_transcription
from .types import TranscriptionResult


@dataclass
class ConformerLayerStreamState:
    attention: Optional[Tensor] = None
    convolution: Optional[Tensor] = None

    def tensors(self):
        return self.attention, self.convolution

    def restore(self, values: Iterator[Tensor]) -> None:
        self.attention = next(values)
        self.convolution = next(values)


@dataclass
class ZapformerLayerStreamState:
    query: Optional[Tensor] = None
    value: Optional[Tensor] = None
    mean: Optional[Tensor] = None
    convolution: Optional[Tensor] = None
    has_convolution: bool = True

    def tensors(self):
        if self.has_convolution:
            return self.query, self.value, self.mean, self.convolution
        return self.query, self.value

    def restore(self, values: Iterator[Tensor]) -> None:
        self.query = next(values)
        self.value = next(values)
        if self.has_convolution:
            self.mean = next(values)
            self.convolution = next(values)


LayerStreamState = Union[ConformerLayerStreamState, ZapformerLayerStreamState]


@dataclass
class EncoderStreamState:
    pending: Optional[Tensor] = None
    position: int = 0
    valid: Any = None
    chunk_size: int = 16
    left_context: int = 32
    frontend: List[Tensor] = field(default_factory=list)
    layers: List[LayerStreamState] = field(default_factory=list)

    def tensors(self):
        return tuple(value for layer in self.layers for value in layer.tensors())

    def restore(self, values: Iterable[Tensor]) -> None:
        values = iter(values)
        for layer in self.layers:
            layer.restore(values)

    def take_ready(self, x, size, final, axis=1):
        self.pending = x if self.pending is None else torch.cat((self.pending, x), axis)
        total = self.pending.shape[axis]
        count = total if final else (total // size * size if size else 0)
        ready = self.pending.narrow(axis, 0, count)
        self.pending = self.pending.narrow(axis, count, total - count).clone()
        return ready, count


@dataclass
class DecoderStreamState:
    previous: Optional[int] = None
    hidden: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None
    frame_offset: int = 0
    tokens: List[int] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    published: str = ""
    published_words: int = 0
    prediction: Optional[Tensor] = None
    prediction_hidden: Optional[Tuple[Tensor, Tensor]] = None
    token_logprobs: Optional[List[float]] = None
    published_tokens: int = 0


def decode_frames(head, decoding, encoded: Tensor, state: DecoderStreamState):
    lengths = encoded.new_tensor([encoded.shape[-1]], dtype=torch.long)
    if state.token_logprobs is None:
        decoding.decode(head, encoded, lengths, [state])
        return
    scores: List[List[float]] = []
    decoding.decode(head, encoded, lengths, [state], token_logprobs=scores)
    state.token_logprobs.extend(scores[0])


class GigaAMStreaming(nn.Module):
    """Transcribe one mono stream using a saved context in milliseconds."""

    def __init__(
        self,
        model: Any,
        future_context_ms: Optional[int] = None,
        language=None,
        word_timestamps: bool = False,
        return_confidence: bool = False,
    ):
        """Optionally publish timed words and confidence for each newly emitted piece."""
        super().__init__()
        self.model = model.eval()
        frontend = getattr(self.model.encoder, "frontend", None)
        if getattr(frontend, "subsampling_type", None) == "conv2d":
            raise ValueError("Streaming requires Conv1d subsampling.")
        if getattr(self.model.preprocessor, "center", False):
            raise ValueError("Streaming requires a center=False feature extractor.")
        self._language = language
        self.word_timestamps = word_timestamps
        self.return_confidence = return_confidence
        self._decoders = (
            {name: (model.heads[name], model.decodings[name]) for name in model.heads}
            if hasattr(model, "heads")
            else {"": (model.head, model.decoding)}
        )
        contexts = model.cfg.get("streaming", {}).get("contexts", [])
        self._contexts = {context["future_context_ms"]: context for context in contexts}
        self._configure_audio(model.cfg)
        self._configure_context(
            max(self._contexts) if future_context_ms is None else future_context_ms
        )
        self.reset_state()

    def _configure_audio(self, cfg):
        mel = self.model.preprocessor.featurizer[0]
        spec = mel.spectrogram
        self.sample_rate = int(cfg.preprocessor.sample_rate)
        self._nfft, self._hop = spec.n_fft, spec.hop_length
        self._frame_ms = self.model.encoder.frame_shift_ms

    def to_onnx(self, dir_path=".", dtype=torch.float32, dynamic_context=True):
        return self.model.to_onnx(
            dir_path,
            dtype,
            self.future_context_ms,
            streaming=True,
            dynamic_context=dynamic_context,
        )

    @property
    def future_context_ms(self):
        return int(self._chunk_size * self._frame_ms)

    @property
    def supported_future_context_ms(self):
        return tuple(sorted(self._contexts))

    def _configure_context(self, value):
        context = self._contexts[value]
        self._chunk_size = int(value / self._frame_ms)
        self._left_context = context["left_context"]
        self.model.encoder.chunk_size = self._chunk_size

    def set_future_context_ms(self, value: int):
        self._configure_context(value)
        self.reset_state()

    def set_language(self, language):
        self._language = language
        self.reset_state()

    def reset_state(self):
        self.encoder_state = self.model.encoder.create_stream_state(
            self._chunk_size, self._left_context
        )
        self.decoder_states = {
            name: DecoderStreamState(
                token_logprobs=[] if self.return_confidence else None
            )
            for name in self._decoders
        }
        with torch.inference_mode():
            self._lang_embedding = self.model.encode_language(self._language, 1)
        self._wave: Optional[Tensor] = None
        self._samples = 0
        self._finished = False

    @property
    def decoder_state(self):
        return self.decoder_states[""]

    def _features(self):
        frames = max(0, (self._wave.numel() - self._nfft) // self._hop + 1)
        if not frames:
            return self._wave.new_empty(1, self.model.cfg.preprocessor.features, 0)
        count = (frames - 1) * self._hop + self._nfft
        features, _ = self.model.preprocessor(
            self._wave[:count][None], self._wave.new_tensor([count], dtype=torch.long)
        )
        self._wave = self._wave[frames * self._hop :].clone()
        return features

    @torch.inference_mode()
    def transcribe_partial(
        self, waveform: Tensor, final: bool = False
    ) -> Union[TranscriptionResult, Dict[str, TranscriptionResult]]:
        """Return new text and words; their times are relative to utterance start."""
        if self._finished:
            empty = {
                name: TranscriptionResult("", [] if self.word_timestamps else None)
                for name in self._decoders
            }
            return empty[""] if "" in empty else empty
        features = self._packet_features(waveform)
        device = features.device
        lengths = torch.tensor([features.shape[-1]], device=device)
        dtype = next(self.model.encoder.parameters()).dtype
        with torch.autocast(
            device_type=device.type,
            dtype=dtype if device.type == "cuda" else torch.bfloat16,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            kwargs = (
                {"lang_embedding": self._lang_embedding}
                if self._lang_embedding is not None
                else {}
            )
            encoded, _, self.encoder_state = self.model.encoder(
                features.to(dtype),
                lengths,
                state=self.encoder_state,
                final=final,
                **kwargs,
            )
            if self.model.language_fusion is not None:
                encoded = cast(nn.Module, self.model.language_fusion)(
                    encoded, self._lang_embedding
                )
        if encoded.shape[-1]:
            for name, (head, decoding) in self._decoders.items():
                head_dtype = next(head.parameters(), encoded).dtype
                decode_frames(
                    head,
                    decoding,
                    encoded.to(head_dtype),
                    self.decoder_states[name],
                )
        return self._publish_packet(final)

    def _packet_features(self, waveform):
        device = self.model._device
        waveform = waveform.to(device=device, dtype=torch.float32)
        self._samples += waveform.numel()
        self._wave = (
            waveform if self._wave is None else torch.cat((self._wave, waveform))
        )
        features = self._features()
        return features

    def _publish_packet(self, final):
        self._finished = final
        pieces = {}
        for name, (_, decoding) in self._decoders.items():
            pieces[name] = self._publish(
                decoding.tokenizer, self.decoder_states[name], final
            )
        return pieces[""] if "" in pieces else pieces

    def _publish(self, tokenizer, state, final):
        count = len(state.tokens)
        if self.word_timestamps and not final:
            for index in range(count - 1, -1, -1):
                piece = tokenizer.id_to_str(state.tokens[index])
                if piece == " " or piece.startswith("▁"):
                    count = index + 1 if piece == " " else index
                    break
            else:
                count = 0
        result = tokens_to_transcription(
            tokenizer,
            state.tokens[:count],
            state.frames[:count],
            self._frame_ms / 1000 if self.word_timestamps else None,
            token_logprobs=(
                state.token_logprobs[:count]
                if self.word_timestamps and state.token_logprobs is not None
                else None
            ),
        )
        if result.words is not None:
            result.words = result.words[state.published_words :]
            state.published_words += len(result.words)
            if final:
                for word in result.words:
                    word.end = min(word.end, self._samples / self.sample_rate)
        elif not final and result.text.endswith("\ufffd"):
            result.text = result.text.rstrip("\ufffd")
            if state.token_logprobs is not None:
                # Keep unfinished UTF-8 byte tokens for the packet that publishes them.
                count -= 1
                while count > state.published_tokens and tokenizer.decode(
                    state.tokens[:count]
                ).endswith("\ufffd"):
                    count -= 1
        text = result.text
        result.text = text[len(state.published) :]
        state.published = text
        if state.token_logprobs is not None:
            result.confidence = (
                confidence_from_logprobs(
                    state.token_logprobs[state.published_tokens : count]
                )
                if result.text
                else None
            )
            if result.text:
                state.published_tokens = count
        return result

    def flush(self) -> Union[TranscriptionResult, Dict[str, TranscriptionResult]]:
        """Publish the remaining text and final word."""
        return self.transcribe_partial(torch.empty(0), final=True)
