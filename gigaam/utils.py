import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.jit import TracerWarning

from .preprocess import load_audio


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
):
    if inputs is None:
        inputs = module.input_example()  # type: ignore[operator]
    if input_names is None:
        input_names = module.input_names()  # type: ignore[operator]
    if output_names is None:
        output_names = module.output_names()  # type: ignore[operator]

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    out_path = str(Path(out_dir) / f"{model_name}.onnx")
    saved_dtype = next(module.parameters()).dtype
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=TracerWarning)
        torch.onnx.export(
            module.to(torch.float32),
            inputs,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    print(f"Successfully ported onnx {model_name} to {out_path}.")
    module.to(saved_dtype)


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


def download_short_audio():
    """Download test audio file if not exists"""
    audio_file = "example.wav"
    if not os.path.exists(audio_file):
        os.system(
            'wget -O example.wav "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"'
        )
    assert os.path.exists(audio_file), "Short audio file not found"
    return audio_file


def download_long_audio():
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
    Helper class for creating batched inputs
    """

    def __init__(self, lst: List[Union[str, np.ndarray, torch.Tensor]]):
        if len(lst) == 0:
            raise ValueError("AudioDataset cannot be initialized with an empty list")
        assert isinstance(
            lst[0], (str, np.ndarray, torch.Tensor)
        ), f"Unexpected dtype: {type(lst[0])}"
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        item = self.lst[idx]
        if isinstance(item, str):
            wav_tns = load_audio(item)
        elif isinstance(item, np.ndarray):
            wav_tns = torch.from_numpy(item)
        elif isinstance(item, torch.Tensor):
            wav_tns = item
        else:
            raise RuntimeError(f"Unexpected sample type: {type(item)} at idx={idx}")
        return wav_tns

    @staticmethod
    def collate(wavs):
        lengths = torch.tensor([len(wav) for wav in wavs])
        max_len = lengths.max().item()
        wav_tns = torch.zeros(len(wavs), max_len, dtype=wavs[0].dtype)
        for idx, wav in enumerate(wavs):
            wav_tns[idx, : wav.shape[-1]] = wav.squeeze()
        return wav_tns, lengths
