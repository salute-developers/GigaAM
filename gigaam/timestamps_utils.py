from typing import List

from .decoding import Tokenizer
from .preprocess import SAMPLE_RATE
from .types import Word


def compute_frame_shift(audio_length_samples: int, seq_len: int) -> float:
    """Compute frame shift (seconds per encoder frame)."""
    return audio_length_samples / SAMPLE_RATE / seq_len


def frames_to_words(
    tokenizer: Tokenizer,
    token_ids: List[int],
    token_frames: List[int],
    frame_shift: float,
) -> List[Word]:
    """
    Convert token-level frame indices to word-level timestamps.
    Groups tokens into words at word boundaries (space or sentencepiece '▁' prefix).
    """
    words: List[Word] = []
    current_chars: List[str] = []
    current_frames: List[int] = []

    def commit():
        if not current_chars:
            return
        text = "".join(current_chars).strip()
        if not text:
            current_chars.clear()
            current_frames.clear()
            return
        start = current_frames[0] * frame_shift
        end = (current_frames[-1] + 1) * frame_shift
        words.append(Word(text=text, start=start, end=end))
        current_chars.clear()
        current_frames.clear()

    for token_id, frame in zip(token_ids, token_frames):
        char = tokenizer.id_to_str(token_id)
        if char.startswith("▁"):
            commit()
            char = char[1:]
        elif char == " ":
            commit()
            continue
        current_chars.append(char)
        current_frames.append(frame)

    commit()
    return words
