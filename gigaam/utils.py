import csv
import os
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.jit import TracerWarning

from .preprocess import SAMPLE_RATE
from .types import AudioDatasetSample


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
    export_dtype: torch.dtype = torch.float32,
):
    """
    Export a submodule to ONNX: casts inputs and ``module`` to ``export_dtype`` for tracing,
    then restores the module to float32 via ``module.float()`` so the model stays usable.
    """
    if inputs is None:
        inputs = module.input_example()  # type: ignore[operator]
    if input_names is None:
        input_names = module.input_names()  # type: ignore[operator]
    if output_names is None:
        output_names = module.output_names()  # type: ignore[operator]

    inputs = tuple(
        x.to(export_dtype) if x.dtype == torch.float32 else x for x in inputs
    )

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    out_path = str(Path(out_dir) / f"{model_name}.onnx")
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=TracerWarning)
        torch.onnx.export(
            module.to(export_dtype),
            inputs,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
        )
    print(f"Successfully ported onnx {model_name} to {out_path}.")
    # We force the whole module to float32 to avoid fp16 preprocessing issues
    module.float()


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
            'wget -O example.wav "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"'
        )
    assert os.path.exists(audio_file), "Short audio file not found"
    return audio_file


def download_long_audio() -> str:
    """Download test audio file if not exists"""
    audio_file = "long_example.wav"
    if not os.path.exists(audio_file):
        os.system(
            'wget -O long_example.wav "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/long_example.wav"'
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

        text = text.replace("ё", "е").replace("Ё", "Е")
        text = " ".join(text.split())

        if self.tokenizer is not None and getattr(self.tokenizer, "charwise", False):
            vocab = set(self.tokenizer.vocab)
            return "".join(c for c in text.lower() if c in vocab)

        return text.lower()

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
        if n_total == 0:
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
            wav, sr = torchaudio.load(item)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            return wav
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
