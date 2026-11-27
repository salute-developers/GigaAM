import math
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from .decoder import CTCHead
from .encoder import ConformerEncoder
from .preprocess import SAMPLE_RATE, FeatureExtractor, load_audio


def _kwargs(section: dict) -> dict:
    return {k: v for k, v in section.items() if k != "_target_"}


class GigaAMVAD(nn.Module):
    """Conformer VAD on shared ASR features, with independent 10-second windows."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = OmegaConf.create(cfg)
        self.preprocessor = FeatureExtractor(**_kwargs(cfg["preprocessor"]))
        self.encoder = ConformerEncoder(**_kwargs(cfg["encoder"]))
        self.head = CTCHead(**_kwargs(cfg["head"]))
        self.step_sec = (
            cfg["encoder"]["subsampling_factor"]
            * self.preprocessor.hop_length
            / SAMPLE_RATE
        )

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        encoded, _ = self.encoder(features, lengths)
        return self.head.decoder_layers(encoded)[:, 0]

    @torch.inference_mode()
    def speech_probs(self, audio: Tensor, batch_size: int = 32) -> Tensor:
        return speech_probabilities(
            audio, self.preprocessor, self, batch_size, self.cfg
        )

    def to_onnx(self, dir_path=".", dtype=torch.float32):
        from .utils import onnx_converter

        with self.encoder.onnx_export_mode():
            onnx_converter(
                "vad",
                self,
                dir_path,
                inputs=self.encoder.input_example(batch_size=2),
                input_names=["features", "lengths"],
                output_names=["logits"],
                dynamic_axes={
                    "features": {0: "batch", 2: "frames"},
                    "lengths": {0: "batch"},
                    "logits": {0: "batch", 1: "steps"},
                },
                export_dtype=dtype,
            )
        cfg = OmegaConf.create(self.cfg)
        cfg.model_name = "vad"
        OmegaConf.save(cfg, str(Path(dir_path) / "vad.yaml"))


@torch.inference_mode()
def speech_probabilities(audio, preprocessor, forward, batch_size=32, cfg=None):
    subsampling = cfg.encoder.subsampling_factor
    hop = preprocessor.hop_length
    step_samples = subsampling * hop
    # A centered STFT needs enough samples for its built-in reflection padding.
    min_samples = (
        preprocessor.n_fft // 2 + 1 if preprocessor.center else preprocessor.n_fft
    )
    if audio.numel() < min_samples:
        return torch.empty(0, dtype=torch.float32, device="cpu")

    device = next(preprocessor.buffers()).device
    full_features = None
    if preprocessor.center or preprocessor.n_fft != preprocessor.win_length:
        length = torch.tensor([audio.numel()], device=device)
        waveform = audio.to(device=device, dtype=torch.float32)
        full_features, _ = preprocessor(waveform[None], length)
        frames = full_features.size(-1)
    else:
        frames = int(preprocessor.out_len(torch.tensor([audio.numel()]))[0])

    def feature_frames(start, count):
        if full_features is not None:
            return full_features[:, :, start : start + count]
        sample_start = start * hop
        sample_count = (count - 1) * hop + preprocessor.n_fft
        chunk = audio[sample_start : sample_start + sample_count].to(
            device=device, dtype=torch.float32
        )
        lengths = torch.tensor([sample_count], device=device)
        features, _ = preprocessor(chunk[None], lengths)
        return features

    window_steps = round(cfg.vad.window_sec * SAMPLE_RATE / step_samples)
    window = window_steps * subsampling
    complete, tail_frames = divmod(frames, window)

    probs = []
    for begin in range(0, complete, batch_size):
        start = begin * window
        count = min(batch_size, complete - begin) * window
        features = feature_frames(start, count)
        batch = features[0].reshape(features.shape[1], -1, window).transpose(0, 1)
        lengths = torch.full([batch.shape[0]], window, device=device)
        probs.append(torch.sigmoid(forward(batch, lengths)).reshape(-1))
    if tail_frames:
        tail = feature_frames(complete * window, tail_frames)
        lengths = torch.tensor([tail.shape[-1]], device=device)
        probs.append(torch.sigmoid(forward(tail, lengths))[0])
    return torch.cat(probs).float().cpu()


@lru_cache(maxsize=4)
def get_vad_model(device, model_path=None):
    from . import load_model

    return load_model(model_path or "vad", device=device)


def speech_timestamps(
    probs: np.ndarray,
    step_sec: float,
    duration: float,
    threshold: float = 0.5,
    neg_threshold: float = 0.35,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.1,
) -> List[Tuple[float, float]]:
    """Speech intervals in seconds, using hysteresis and the native model grid."""
    min_silence_steps = math.ceil(min_silence_duration / step_sec - 1e-9)
    steps_per_sec = 1 / step_sec

    speeches: List[Tuple[float, float]] = []
    triggered, start, temp_end = False, 0, None
    for step, prob in enumerate(probs.tolist()):
        if prob >= threshold:
            temp_end = None
            if not triggered:
                triggered, start = True, step
            continue
        if prob < neg_threshold and triggered:
            if temp_end is None:
                temp_end = step
            if step - temp_end + 1 < min_silence_steps:
                continue
            if (temp_end - start) * step_sec > min_speech_duration:
                speeches.append((start / steps_per_sec, temp_end / steps_per_sec))
            temp_end, triggered = None, False
    if triggered and duration - start / steps_per_sec > min_speech_duration:
        speeches.append((start / steps_per_sec, duration))
    return speeches


def _min_prob_cuts(
    start: float,
    end: float,
    probs: np.ndarray,
    steps_per_sec: float,
    limit: float,
    margin: float,
) -> List[float]:
    """Split long speech at minimum-probability steps while respecting margins."""
    if end - start <= limit:
        return []
    parts = math.ceil((end - start) / limit)
    steps = [
        k
        for k in range(
            math.ceil((start + margin) * steps_per_sec - 1e-9),
            math.ceil((end - margin) * steps_per_sec - 1e-9),
        )
        if math.ceil((k / steps_per_sec - start) / limit)
        + math.ceil((end - k / steps_per_sec) / limit)
        == parts
    ]
    if not steps:
        return [start + (end - start) * q / parts for q in range(1, parts)]
    cut = min(steps, key=lambda k: probs[k]) / steps_per_sec
    return [
        *_min_prob_cuts(start, cut, probs, steps_per_sec, limit, margin),
        cut,
        *_min_prob_cuts(cut, end, probs, steps_per_sec, limit, margin),
    ]


def segment_audio_file(
    wav_file: str,
    sr: int,
    merge_gap: float = 0.7,
    pad: float = 0.2,
    strict_limit_duration: float = 30.0,
    device: torch.device = torch.device("cpu"),
    vad_batch_size: int = 32,
    vad_model_path: Optional[str] = None,
    vad=None,
    threshold: float = 0.5,
    neg_threshold: float = 0.35,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.1,
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """Merge padded speech and split long segments at pauses or minimum speech probability."""
    audio = load_audio(wav_file)
    if vad is None:
        model = get_vad_model(device, model_path=vad_model_path)
        probs = model.speech_probs(audio, batch_size=vad_batch_size).numpy()
        step_sec = model.step_sec
    else:
        probs = np.asarray(vad(audio))
        step_sec = 0.04
    total = audio.shape[0] / sr
    speech = speech_timestamps(
        probs,
        step_sec,
        total,
        threshold,
        neg_threshold,
        min_speech_duration,
        min_silence_duration,
    )
    inner = strict_limit_duration - max(2 * pad, merge_gap) - 1e-3  # fits a chunk

    margin = 0.1 * strict_limit_duration  # cuts not closer to segment edges

    segs: List[Tuple[float, float]] = []
    for start, end in speech:
        cuts = _min_prob_cuts(start, end, probs, 1 / step_sec, inner, margin)
        edges = [start, *cuts, end]
        segs += list(zip(edges, edges[1:]))

    groups: List[List[int]] = []  # pieces: [first, last] segs, pauses <= merge_gap
    for q in range(len(segs)):
        if groups and segs[q][0] - segs[q - 1][1] <= merge_gap + 1e-9:
            groups[-1][1] = q
        else:
            groups.append([q, q])

    boundaries: List[Tuple[float, float]] = []
    for i, j in groups:
        start, end = max(0.0, segs[i][0] - pad), min(total, segs[j][1] + pad)
        k = i
        while end - start > strict_limit_duration + 1e-9:
            cands = []  # (pause length, pause middle, seg before it)
            for m in range(k, j):
                mid = (segs[m][1] + segs[m + 1][0]) / 2
                if mid - start > strict_limit_duration + 1e-9:
                    break
                cands.append((segs[m + 1][0] - segs[m][1], mid, m))
            _, mid, m = max(cands)
            boundaries.append((start, mid))
            start, k = mid, m + 1
        boundaries.append((start, end))

    segments = [audio[round(s * sr) : round(e * sr)] for s, e in boundaries]
    return segments, boundaries
