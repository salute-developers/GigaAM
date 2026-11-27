import copy
import csv
import json
import os
import re
import shutil
import unicodedata
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import omegaconf
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.jit import TracerWarning

from .preprocess import SAMPLE_RATE, load_audio
from .types import AudioDatasetSample


def normalize_raw_text(text: str, language: Optional[str] = None) -> str:
    """
    Script-agnostic normalization: lowercase, collapse whitespace,
    replace punctuation with spaces, and drop other symbols.

    Word-internal apostrophes are preserved (e.g. ``don't`` stays ``don't``).
    Apostrophe variants are normalized to ASCII ``'``.
    Standalone quotes are replaced with spaces.
    """
    if language == "az":
        text = text.replace("I", "ı").replace("İ", "i")
    text = text.replace("ё", "е").replace("Ё", "Е").lower()
    for quote in ("’", "‘", "ʻ", "ʼ"):
        text = text.replace(quote, "'")
    out = []
    for i, c in enumerate(text):
        if c == "'":
            if (
                0 < i < len(text) - 1
                and text[i - 1].isalnum()
                and text[i + 1].isalnum()
            ):
                out.append(c)
            else:
                out.append(" ")
        elif c.isalnum() or c.isspace():
            out.append(c)
        elif unicodedata.category(c).startswith("P") and c not in "՛՜՞":
            out.append(" ")
    return " ".join("".join(out).split())


def _namespace_onnx_external_data(path):
    """Keep exported graphs from overwriting each other's external weights."""
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    graph = onnx.load(path, load_external_data=False)
    locations = {}
    for tensor in _get_all_tensors(graph):
        for entry in tensor.external_data:
            if entry.key == "location":
                location = entry.value
                if location not in locations:
                    target = Path(location).with_name(
                        f"{path.stem}_{Path(location).name}"
                    )
                    (path.parent / location).replace(path.parent / target)
                    locations[location] = str(target)
                entry.value = locations[location]
    if locations:
        onnx.save(graph, path)


def onnx_converter(
    model_name: str,
    module: torch.nn.Module,
    out_dir: str,
    inputs: Optional[Tuple[Tensor, ...]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[
        Union[Dict[str, List[int]], Dict[str, Dict[int, str]]]
    ] = None,
    opset_version: int = 17,
    export_dtype: Optional[torch.dtype] = torch.float32,
):
    """
    Export a submodule to ONNX: casts inputs and ``module`` to ``export_dtype`` for tracing,
    then restores the module to float32 via ``module.float()`` so the model stays usable.
    With ``export_dtype=None``, preserve existing module and input dtypes instead.
    """
    if inputs is None:
        inputs = module.input_example()  # type: ignore[operator]
    if input_names is None:
        input_names = module.input_names()  # type: ignore[operator]
    if output_names is None:
        output_names = module.output_names()  # type: ignore[operator]

    inputs = tuple(
        (
            x.to(export_dtype)
            if export_dtype is not None and x.dtype == torch.float32
            else x
        )
        for x in inputs
    )

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    out_path = str(Path(out_dir) / f"{model_name}.onnx")
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=TracerWarning)
        torch.onnx.export(
            module.to(export_dtype) if export_dtype is not None else module,
            inputs,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
        )
    _namespace_onnx_external_data(Path(out_path))
    print(f"Successfully ported onnx {model_name} to {out_path}.")
    # We force the whole module to float32 to avoid fp16 preprocessing issues
    if export_dtype is not None:
        module.float()


def _state_schema(tensors):
    return [
        dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix("torch."))
        for value in tensors
    ]


def export_onnx(
    model,
    dir_path=".",
    dtype=torch.float32,
    future_context_ms=None,
    streaming=False,
    dynamic_context=None,
):
    """Prepare an export-only model copy and delegate graph composition."""
    dual = hasattr(model, "heads")
    model = copy.deepcopy(model).float().eval()
    if dual:
        model.encoder.to(dtype=dtype)
        return model._to_onnx(
            dir_path, dtype, future_context_ms, streaming, dynamic_context
        )
    model._to_onnx(dir_path, dtype)
    omegaconf.OmegaConf.save(model.cfg, f"{dir_path}/{model.cfg.model_name}.yaml")


def _export_advance(model, directory, name, example):
    from .decoder import RNNTDecodeExport

    decoder = RNNTDecodeExport(model, example.float())
    onnx_converter(
        name,
        torch.jit.script(decoder),
        str(directory),
        decoder.input_example(example.float()),
        ["projected", "lengths"],
        ["tokens"],
        decoder.dynamic_axes(),
    )


def _export_heads(model, directory, model_name=None, *, frame_windows=False):
    for name, head in model.heads.items():
        prefix = f"{model_name or model.cfg.model_name}_{name}"
        parts = (
            {"ctc": head}
            if hasattr(head, "decoder_layers")
            else {"decoder": head.decoder, "joint": head.joint}
        )
        for part, module in parts.items():
            axes = module.dynamic_axes()
            if frame_windows and part == "joint":
                axes["enc"][2] = axes["joint"][1] = "frames"
            onnx_converter(
                f"{prefix}_{part}",
                module,
                str(directory),
                dynamic_axes=axes,
            )


def _save_onnx_config(model, directory, future_context_ms=None, model_name=None):
    directory = Path(directory)
    cfg = model.cfg
    cfg.model_name = model_name or model.cfg.model_name
    cfg.onnx_future_context_ms = future_context_ms
    cfg.onnx_default_language = model.default_language
    for name, decoding in model.decodings.items():
        path = cfg.decodings[name].get("model_path")
        if path:
            target = f"{cfg.model_name}_{name}_tokenizer.model"
            shutil.copyfile(path, directory / target)
            cfg.decodings[name].model_path = target
    omegaconf.OmegaConf.save(cfg, directory / f"{cfg.model_name}.yaml")


def _export_streaming(model, directory, future_context_ms, dynamic_context):
    contexts = {int(c.future_context_ms): c for c in model.cfg.streaming.contexts}
    milliseconds = max(contexts) if future_context_ms is None else future_context_ms
    context = contexts[milliseconds]
    factor = model.encoder.subsampling_factor
    frame_ms = model.encoder.frame_shift_ms
    chunk, left = int(milliseconds / frame_ms), int(context.left_context)

    encoder, device = model.encoder, model._device
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base_name = re.sub(r"_(?:\d+ms|dynamic)$", "", model.cfg.model_name)
    prefix = (
        f"{base_name}_dynamic" if dynamic_context else f"{base_name}_{milliseconds}ms"
    )
    manifest = dict(model_name=prefix, future_context_ms=milliseconds)
    warm = encoder.create_stream_state(chunk, left)
    example = torch.zeros(
        1,
        model.cfg.preprocessor.features,
        factor * (left + chunk),
        device=device,
        dtype=next(encoder.parameters()).dtype,
    )
    lengths = torch.tensor([example.shape[-1]], device=device)
    kwargs = {}
    if model.language_embedding is not None:
        kwargs["lang_embedding"] = model.language_embedding.weight.new_zeros(
            1, model.language_embedding.weight.shape[1]
        )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda"
            and next(encoder.parameters()).dtype == torch.float16,
        ),
    ):
        encoder(example, lengths, state=warm, final=True, **kwargs)
    states = [
        torch.zeros_like(value) for layer in warm.layers for value in layer.tensors()
    ]
    example = example[:, :, : chunk * factor]
    lengths = torch.tensor([example.shape[-1]], device=device)
    front = [torch.zeros_like(value) for value in warm.frontend]
    name = f"{prefix}_stream_frontend"
    saved_forward = encoder.forward
    encoder.forward = encoder.forward_frontend_for_export
    batch = getattr(encoder, "_triton_batch", False)
    native_ctc = (
        base_name == "v4_ctc"
        and batch
        and dynamic_context
        and getattr(encoder, "_triton_native", False)
    )
    if native_ctc:
        encoder.frontend._export_native_conv = True
    manifest["batched"] = batch

    def export(name, module, inputs, names, outputs, axes):
        if batch:
            inputs = tuple(
                (
                    value
                    if key == "context_frames"
                    else value.repeat((2,) + (1,) * (value.ndim - 1))
                )
                for key, value in zip(names, inputs, strict=True)
            )
            axes = {
                key: {0: "batch", **axes.get(key, {})}
                for key in (*names, *outputs)
                if key != "context_frames"
            }
        onnx_converter(
            name,
            module,
            str(directory),
            inputs,
            names,
            outputs,
            axes,
            export_dtype=None,
        )

    export(
        name,
        encoder,
        (example, lengths, *front),
        ["features", "feature_lengths", *[f"front_{i}" for i in range(len(front))]],
        [
            "pre_encoded",
            "encoded_lengths",
            *[f"next_front_{i}" for i in range(len(front))],
        ],
        {"features": {2: "mel_frames"}, "pre_encoded": {1: "frames"}},
    )
    encoder.forward = saved_forward
    frontend = dict(
        file=name + ".onnx",
        states=_state_schema(front),
    )
    encoder._onnx_context = (chunk, left, dynamic_context)
    encoder._onnx_caches = states
    model.forward = model.forward_for_export
    name = f"{prefix}_stream_step"
    if native_ctc:
        from .decoder import CTCStreamExport

        inputs, outputs = encoder.input_names(), encoder.output_names()
        export(
            name,
            CTCStreamExport(model, len(front)),
            (example, lengths, *encoder.input_example()[2:5], *front, *states),
            [*inputs[:5], *[f"front_{i}" for i in range(len(front))], *inputs[5:]],
            [
                *outputs[:4],
                *[f"next_front_{i}" for i in range(len(front))],
                *outputs[4:],
            ],
            {"features": {2: "mel_frames"}, "encoded": {2: "frames"}},
        )
        manifest["ctc_labels"] = list(model.heads)
        del encoder.frontend._export_native_conv
    else:
        export(
            name,
            model,
            encoder.input_example(),
            encoder.input_names(),
            encoder.output_names(),
            encoder.dynamic_axes(),
        )
    native_rnnt = getattr(encoder, "_triton_native", False) and base_name in (
        "v4_rnnt",
        "multilingual_v2_rnnt",
        "v2_multilingual_rnnt",
    )
    from .decoder import CTCHead, DualASRExport, RNNTHead

    encoder._onnx_context = dict(chunk=chunk, left_context=left)
    heads = tuple(getattr(model, "heads", {}).values())
    fused = bool(heads) and (
        all(isinstance(head, CTCHead) for head in heads)
        or (
            all(isinstance(head, RNNTHead) for head in heads)
            and getattr(encoder, "_triton_provider", None)
            not in ("TensorRT", "TensorRT-fp16")
        )
    )
    whole = DualASRExport(model) if fused else None
    if whole is not None:
        model.cfg.onnx_fc = list(model.heads)
    whole_name = f"{prefix}_encoder"
    onnx_converter(
        whole_name,
        whole if whole is not None else model,
        str(directory),
        encoder.input_example(batch_size=2, seqlen=96),
        encoder.input_names(),
        whole.output_names if whole is not None else encoder.output_names(),
        whole.dynamic_axes() if whole is not None else encoder.dynamic_axes(),
        export_dtype=None,
    )
    if whole is not None and not whole.ctc:
        _export_advance(model, directory, f"{prefix}_advance", example)
        manifest["advance"] = dict(file=f"{prefix}_advance.onnx")
    manifest["whole"] = dict(file=whole_name + ".onnx")
    manifest.update(
        subsampling_factor=factor,
        frame_shift_ms=frame_ms,
        encoder_dim=encoder.feat_out,
        frontend=frontend,
        step=dict(file=name + ".onnx"),
        states=_state_schema(states),
    )
    if native_rnnt:
        from .decoder import RNNTStreamDecodeExport

        decoder = RNNTStreamDecodeExport(model, example.float())
        name = f"{prefix}_packet_decode"
        onnx_converter(
            name,
            torch.jit.script(decoder),
            str(directory),
            decoder.input_example(example.float()),
            decoder.input_names,
            decoder.output_names,
            decoder.dynamic_axes(),
        )
        manifest.update(packet_decode=name + ".onnx", packet_heads=list(model.heads))
    for filename in (f"{prefix}.streaming.json", "streaming.json"):
        (directory / filename).write_text(json.dumps(manifest, indent=2) + "\n")
    provider = getattr(encoder, "_triton_provider", None)
    _export_heads(
        model,
        directory,
        prefix,
        frame_windows=provider not in ("TensorRT", "TensorRT-fp16"),
    )
    _save_onnx_config(model, directory, milliseconds, prefix)
    return prefix


def format_time(seconds: float) -> str:
    """
    Formats time in seconds to HH:MM:SS:mm format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)
    milliseconds = int((seconds - full_seconds) * 100)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"


def rtt_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=x1.ndim - 1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, offset: int = 0
) -> Tuple[Tensor, Tensor]:
    """
    Applies Rotary Position Embeddings to query and key tensors.
    """
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    return (q * cos) + (rtt_half(q) * sin), (k * cos) + (rtt_half(k) * sin)


def apply_masked_flash_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    h: int,
    d_k: int,
) -> Tensor:
    """
    Applies Flash Attention with padding masks.
    """

    from einops import rearrange
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    pad_mask = ~mask[:, 0, :]
    b, t = pad_mask.shape
    q = q.view(b, t, h * d_k)
    k = k.view(b, t, h * d_k)
    v = v.view(b, t, h * d_k)

    q_unpad, indices_q, _, max_seqlen_q = unpad_input(q, pad_mask)[:4]
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=h)

    k_unpad = unpad_input(k, pad_mask)[0]
    k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=h)

    v_unpad = unpad_input(v, pad_mask)[0]
    v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=h)

    lengths_q = pad_mask.sum(1).to(torch.int32).to(q.device)
    cu_seqlens_q = F.pad(lengths_q.cumsum(0), (1, 0), value=0).to(torch.int32)
    max_seqlen_q = torch.max(lengths_q)

    output_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_q,
    )

    scores = pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"),
        indices_q,
        b,
        t,
    )

    return scores


def download_short_audio() -> str:
    """Download test audio file if not exists"""
    audio_file = "example.wav"
    if not os.path.exists(audio_file):
        os.system(
            "wget -O example.wav "
            '"https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"'
        )
    assert os.path.exists(audio_file), "Short audio file not found"
    return audio_file


def download_long_audio() -> str:
    """Download test audio file if not exists"""
    audio_file = "long_example.wav"
    if not os.path.exists(audio_file):
        os.system(
            "wget -O long_example.wav "
            '"https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/long_example.wav"'
        )
    assert os.path.exists(audio_file), "Long audio file not found"
    return audio_file


class AudioDataset(torch.utils.data.Dataset):
    """
    Unified dataset class for training and inference.
    Supports loading from manifest file or an iterable of audio paths / waveforms.
    Provides min / max duration filtering, text normalization, and pre-tokenization.
    """

    def __init__(
        self,
        data: Union[str, Iterable[Union[str, np.ndarray, torch.Tensor]]],
        tokenizer=None,
        max_duration: Optional[float] = None,
        min_duration: float = 0.0,
        raw_text: bool = False,
        return_tokens: bool = False,
    ):
        self.raw_text = raw_text
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer
        self.samples: List[AudioDatasetSample] = []

        if return_tokens and tokenizer is None:
            raise ValueError("tokenizer is required when return_tokens=True")

        self.encode = self._make_encoder(tokenizer)

        if isinstance(data, str):
            self._load_manifest(data, min_duration, max_duration)
        elif isinstance(data, Iterable) and not isinstance(
            data, (str, bytes, bytearray)
        ):
            self._load_iterable(data, min_duration, max_duration)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if not self.samples:
            raise ValueError("No valid samples found after filtering")

    def _make_encoder(self, tokenizer):
        if tokenizer is None:
            return None

        if getattr(tokenizer, "charwise", False):
            c2i = {c: i for i, c in enumerate(tokenizer.vocab)}
            return lambda text: [c2i[c] for c in text if c in c2i]

        return tokenizer.model.encode

    def normalize_text(self, text: str) -> str:
        if not self.raw_text:
            return text

        text = normalize_raw_text(text)

        if self.tokenizer is not None and getattr(self.tokenizer, "charwise", False):
            vocab = set(self.tokenizer.vocab)
            return "".join(c for c in text if c in vocab)

        return text

    @staticmethod
    def _get_duration(item: Union[str, np.ndarray, Tensor]) -> float:
        if isinstance(item, str):
            with sf.SoundFile(item) as f:
                return f.frames / f.samplerate
        if isinstance(item, np.ndarray):
            return len(item) / SAMPLE_RATE
        if isinstance(item, torch.Tensor):
            return item.numel() / SAMPLE_RATE
        raise TypeError(f"Unexpected sample type: {type(item)}")

    def _duration_ok(
        self, duration: float, min_duration: float, max_duration: Optional[float]
    ) -> bool:
        if duration < min_duration:
            return False
        if max_duration is not None and duration > max_duration:
            return False
        return True

    @staticmethod
    def _print_filtered(
        n_total: int, dur_total: float, n_filt: int, dur_filt: float
    ) -> None:
        if n_filt == 0:
            return
        pn = 100.0 * n_filt / n_total
        pd = 100.0 * dur_filt / dur_total if dur_total > 0 else 0.0
        h_filt, h_total = dur_filt / 3600.0, dur_total / 3600.0
        print(
            f"filtered by duration: {n_filt}/{n_total} samples ({pn:.1f}%), "
            f"{h_filt:.2f}/{h_total:.2f} h ({pd:.1f}%)"
        )

    def _append_sample(
        self,
        item: Union[str, np.ndarray, Tensor],
        duration: float,
        text: Optional[str] = None,
    ) -> None:
        norm_text: Optional[str] = None
        tokens: Optional[List[int]] = None
        if text is not None:
            norm_text = self.normalize_text(text.strip())
            if self.return_tokens:
                assert self.encode is not None
                tokens = self.encode(norm_text)
        self.samples.append(
            AudioDatasetSample(
                item=item, duration=duration, text=norm_text, tokens=tokens
            )
        )

    def _load_manifest(
        self, manifest_path: str, min_duration: float, max_duration: Optional[float]
    ):
        data_dir = Path(manifest_path).resolve().parent
        n_total = n_filt = 0
        dur_total = dur_filt = 0.0

        with open(manifest_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                duration = float(row["duration"])
                n_total += 1
                dur_total += duration
                if not self._duration_ok(duration, min_duration, max_duration):
                    n_filt += 1
                    dur_filt += duration
                    continue

                pth = Path(row["path"])
                path = str((pth if pth.is_absolute() else data_dir / pth).resolve())
                text = row["transcription"] if "transcription" in row else None
                self._append_sample(path, duration, text=text)

        self._print_filtered(n_total, dur_total, n_filt, dur_filt)

    def _load_iterable(
        self,
        data: Iterable[Union[str, np.ndarray, torch.Tensor]],
        min_duration: float,
        max_duration: Optional[float],
    ):
        n_total = n_filt = 0
        dur_total = dur_filt = 0.0
        for item in data:
            if not isinstance(item, (str, np.ndarray, torch.Tensor)):
                raise TypeError(f"Unexpected dtype: {type(item)}")

            duration = self._get_duration(item)
            n_total += 1
            dur_total += duration
            if not self._duration_ok(duration, min_duration, max_duration):
                n_filt += 1
                dur_filt += duration
                continue

            self._append_sample(item, duration)

        self._print_filtered(n_total, dur_total, n_filt, dur_filt)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_audio(item: Union[str, np.ndarray, Tensor]) -> Tensor:
        if isinstance(item, str):
            try:
                with sf.SoundFile(item) as audio:
                    if audio.subtype == "PCM_16" and audio.channels == 1:
                        if audio.format == "WAV" and audio.samplerate == SAMPLE_RATE:
                            return torch.from_numpy(audio.read(dtype="float32"))
            except RuntimeError:
                pass
            return load_audio(item)
        if isinstance(item, np.ndarray):
            return torch.from_numpy(item)
        if isinstance(item, torch.Tensor):
            return item
        raise TypeError(f"Unexpected sample type: {type(item)}")

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        sample = self.samples[idx]
        wav = self._load_audio(sample.item)

        if self.return_tokens:
            assert sample.tokens is not None
            return wav, torch.tensor(sample.tokens, dtype=torch.long)

        return wav

    @staticmethod
    def collate(wavs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        lengths = torch.tensor([len(w) for w in wavs], dtype=torch.long)
        max_len = int(lengths.max().item())

        batch = torch.zeros(len(wavs), max_len, dtype=wavs[0].dtype)
        for i, wav in enumerate(wavs):
            batch[i, : wav.shape[-1]] = wav.squeeze()

        return batch, lengths

    def collate_fn(
        self, batch: List[Union[Tensor, Tuple[Tensor, Tensor]]]
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if not self.return_tokens:
            return self.collate(cast(List[Tensor], batch))

        wavs, tokens = zip(*cast(List[Tuple[Tensor, Tensor]], batch))
        wav_pad, wav_lens = self.collate(list(wavs))
        tok_pad, tok_lens = self.collate(list(tokens))

        return wav_pad, wav_lens, tok_pad, tok_lens
