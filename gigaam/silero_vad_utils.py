from typing import List, Optional, Tuple

import torch

from .chunking_utils import ChunkingConfig, merge_speech_spans
from .preprocess import load_audio

try:
    from silero_vad import get_speech_timestamps, load_silero_vad
except ImportError as exc:
    raise ImportError(
        "The 'silero' VAD backend requires the silero-vad package. "
        "Install it with: pip install gigaam[silero]"
    ) from exc

_MODEL: Optional[torch.nn.Module] = None


def get_silero_model() -> torch.nn.Module:
    """
    Loads the Silero VAD model with weights bundled in the `silero-vad` package.
    The model is loaded only once and reused across subsequent calls.
    It runs fully offline and does not require a Hugging Face token.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = load_silero_vad()
    return _MODEL


def segment_audio_file(
    wav_file: str,
    sr: int,
    config: Optional[ChunkingConfig] = None,
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    Silero VAD counterpart of `vad_utils.segment_audio_file`.
    Silero VAD always runs on CPU, so no device argument is needed.
    """

    audio = load_audio(wav_file)
    model = get_silero_model()
    timestamps = get_speech_timestamps(audio, model, sampling_rate=sr)
    speech_spans = [(ts["start"] / sr, ts["end"] / sr) for ts in timestamps]
    return merge_speech_spans(audio, speech_spans, sr, config or ChunkingConfig())
