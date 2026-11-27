import math

import pytest
import torch

from gigaam.decoding import CTCGreedyDecoding, Tokenizer
from gigaam.streaming import DecoderStreamState
from gigaam.timestamps_utils import tokens_to_transcription


def test_ctc_scores_follow_emissions_across_packets():
    decoder = CTCGreedyDecoding(["a", "b"])
    # Each row gives P(a), P(b), P(blank). Frame 1 repeats frame 0,
    # frame 3 starts with another "a" after a blank, and frame 5 is padding.
    logprobs = torch.tensor(
        [
            [0.60, 0.15, 0.25],
            [0.99, 0.005, 0.005],
            [0.10, 0.10, 0.80],
            [0.50, 0.25, 0.25],
            [0.20, 0.70, 0.10],
            [0.80, 0.10, 0.10],
        ]
    ).log()
    encoded = torch.stack([logprobs, logprobs])
    lengths = torch.tensor([5, 0])

    def head(encoder_output):
        return encoder_output

    plain = decoder.decode(head, encoded, lengths)
    assert plain == [("aab", [0, 0, 1], [0, 3, 4]), ("", [], [])]
    scores = []
    assert decoder.decode(head, encoded, lengths, token_logprobs=scores) == plain
    expected = [math.log(p) for p in (0.60, 0.50, 0.70)]
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == []

    states = [DecoderStreamState(), DecoderStreamState()]
    scores = []
    for start, end in [(0, 1), (1, 6)]:
        decoder.decode(
            head,
            encoded[:, start:end],
            (lengths - start).clamp(0, end - start),
            states,
            token_logprobs=scores,
        )
    assert scores[0] + scores[2] == pytest.approx(expected)
    assert scores[1] == scores[3] == []
    assert states[0].frames == [0, 3, 4]

    labels = encoded.argmax(-1)
    assert decoder.decode(head, labels, lengths) == plain
    with pytest.raises(ValueError, match="labels"):
        decoder.decode(head, labels, lengths, token_logprobs=[])


def test_word_and_utterance_score_different_groups_of_tokens():
    tokenizer = Tokenizer(["▁a", "b", " ", "▁", "c"])
    ids = frames = [0, 1, 2, 3, 4]
    probabilities = [0.25, 1.0, 0.1, 0.2, 0.8]
    scores = [math.log(p) for p in probabilities]
    scored = tokens_to_transcription(tokenizer, ids, frames, 0.04, scores)
    assert [w.text for w in scored.words] == ["ab", "c"]
    assert [w.confidence for w in scored.words] == pytest.approx([0.5, 0.8])
    assert scored.confidence == pytest.approx(math.prod(probabilities) ** 0.2)
    assert (
        tokens_to_transcription(tokenizer, [], [], token_logprobs=[]).confidence is None
    )
