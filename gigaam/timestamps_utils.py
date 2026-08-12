import math
from typing import List, Optional, Sequence

from .decoding import Tokenizer
from .preprocess import SAMPLE_RATE
from .types import TranscriptionResult, Word


def confidence_from_logprobs(logprobs: Optional[Sequence[float]]) -> Optional[float]:
    """Geometric mean of emitted token probabilities, or None without scores."""
    if logprobs is None or len(logprobs) == 0:
        return None
    return math.exp(sum(logprobs) / len(logprobs))


def tokens_to_transcription(
    tokenizer, token_ids, token_frames, frame_shift=None, token_logprobs=None
):
    """Decode tokens with optional word timestamps and token-based confidence."""
    return TranscriptionResult(
        text=tokenizer.decode(token_ids),
        words=(
            frames_to_words(
                tokenizer, token_ids, token_frames, frame_shift, token_logprobs
            )
            if frame_shift is not None
            else None
        ),
        confidence=confidence_from_logprobs(token_logprobs),
    )


def compute_frame_shift(audio_length_samples: int, seq_len: int) -> float:
    """Compute frame shift (seconds per encoder frame)."""
    return audio_length_samples / SAMPLE_RATE / seq_len


def frames_to_words(
    tokenizer: Tokenizer,
    token_ids: List[int],
    token_frames: List[int],
    frame_shift: float,
    token_logprobs: Optional[Sequence[float]] = None,
) -> List[Word]:
    """
    Convert token-level frame indices to word-level timestamps.
    Groups tokens into words at word boundaries (space or sentencepiece '▁' prefix).
    Word confidence uses the emitted tokens contributing text to that word.
    """
    words: List[Word] = []
    current_chars: List[str] = []
    current_frames: List[int] = []
    current_logprobs: List[float] = []

    def commit():
        text = "".join(current_chars).strip()
        if text:
            words.append(
                Word(
                    text=text,
                    start=current_frames[0] * frame_shift,
                    end=(current_frames[-1] + 1) * frame_shift,
                    confidence=confidence_from_logprobs(current_logprobs),
                )
            )
        current_chars.clear()
        current_frames.clear()
        current_logprobs.clear()

    for i, (token_id, frame) in enumerate(zip(token_ids, token_frames)):
        char = tokenizer.id_to_str(token_id)
        if char.startswith("▁"):
            commit()
            char = char[1:]
        elif char == " ":
            commit()
            continue
        current_chars.append(char)
        current_frames.append(frame)
        if char and token_logprobs is not None and len(token_logprobs):
            current_logprobs.append(token_logprobs[i])

    commit()
    return words
