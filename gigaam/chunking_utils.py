from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Tuning for merging VAD speech spans into ASR-sized chunks.
    """

    max_duration: float = 22.0
    min_duration: float = 15.0
    strict_limit_duration: float = 30.0
    new_chunk_threshold: float = 0.2


def merge_speech_spans(
    audio: torch.Tensor,
    speech_spans: List[Tuple[float, float]],
    sr: int,
    config: ChunkingConfig,
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    Merges VAD speech spans into audio chunks suitable for ASR inference.
    Spans are concatenated into chunks according to config.max_duration and
    config.min_duration; chunks longer than config.strict_limit_duration are
    split manually.
    """
    max_duration = config.max_duration
    min_duration = config.min_duration
    strict_limit_duration = config.strict_limit_duration
    new_chunk_threshold = config.new_chunk_threshold
    segments: List[torch.Tensor] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries: List[Tuple[float, float]] = []

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float):
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    for span_start, span_end in speech_spans:
        start = max(0, span_start)
        end = min(audio.shape[0] / sr, span_end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    return segments, boundaries
