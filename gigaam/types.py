from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    words: Optional[List[Word]] = None

    def __str__(self) -> str:
        return self.text


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: Optional[List[Word]] = None


@dataclass
class LongformTranscriptionResult:
    segments: List[Segment]

    @property
    def words(self) -> List[Word]:
        """Flatten all words from all segments."""
        result = []
        for seg in self.segments:
            if seg.words:
                result.extend(seg.words)
        return result

    @property
    def has_word_timestamps(self) -> bool:
        return bool(self.segments) and self.segments[0].words is not None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    def __str__(self) -> str:
        return self.text

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)
