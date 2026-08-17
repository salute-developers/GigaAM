import math
from typing import List, Optional, Sequence

from .decoding import Tokenizer
from .preprocess import SAMPLE_RATE
from .types import Word


def compute_frame_shift(audio_length_samples: int, seq_len: int) -> float:
    """Compute frame shift (seconds per encoder frame)."""
    return audio_length_samples / SAMPLE_RATE / seq_len


def aggregate_confidence(token_logprobs: Sequence[float]) -> Optional[float]:
    """
    Collapse per-token log-probabilities into a single confidence in (0, 1].

    Uses the length-normalized geometric mean ``exp(mean(log p))`` so that long
    words are not penalized simply for consisting of more tokens.
    Returns None for an empty sequence.
    """
    if not token_logprobs:
        return None
    return math.exp(sum(token_logprobs) / len(token_logprobs))


def frames_to_words(
    tokenizer: Tokenizer,
    token_ids: List[int],
    token_frames: List[int],
    frame_shift: float,
    token_logprobs: Optional[List[float]] = None,
) -> List[Word]:
    """
    Convert token-level frame indices to word-level timestamps.
    Groups tokens into words at word boundaries (space or sentencepiece '▁' prefix).

    When ``token_logprobs`` is given, each word also carries a confidence score
    aggregated over the tokens it was built from (see ``aggregate_confidence``).
    """
    words: List[Word] = []
    current_chars: List[str] = []
    current_frames: List[int] = []
    current_logprobs: List[float] = []

    def commit():
        if not current_chars:
            return
        text = "".join(current_chars).strip()
        if not text:
            current_chars.clear()
            current_frames.clear()
            current_logprobs.clear()
            return
        start = current_frames[0] * frame_shift
        end = (current_frames[-1] + 1) * frame_shift
        words.append(
            Word(
                text=text,
                start=start,
                end=end,
                confidence=aggregate_confidence(current_logprobs),
            )
        )
        current_chars.clear()
        current_frames.clear()
        current_logprobs.clear()

    logprobs = token_logprobs if token_logprobs is not None else [None] * len(token_ids)
    for token_id, frame, logprob in zip(token_ids, token_frames, logprobs):
        char = tokenizer.id_to_str(token_id)
        if char.startswith("▁"):
            commit()
            char = char[1:]
        elif char == " ":
            commit()
            continue
        current_chars.append(char)
        current_frames.append(frame)
        if logprob is not None:
            current_logprobs.append(logprob)

    commit()
    return words
