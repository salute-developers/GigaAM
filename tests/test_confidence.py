import logging
import math

import pytest
import soundfile as sf
import torch

import gigaam
from gigaam.decoding import Hypothesis
from gigaam.preprocess import SAMPLE_RATE, load_audio
from gigaam.timestamps_utils import aggregate_confidence, frames_to_words
from gigaam.utils import download_long_audio, download_short_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REVISIONS = ["v3_ctc", "v3_e2e_rnnt"]


@pytest.fixture(scope="session")
def test_audio():
    return download_short_audio()


@pytest.fixture(scope="session")
def long_audio():
    return download_long_audio()


@pytest.fixture(scope="session")
def noisy_audio(tmp_path_factory, test_audio):
    """The same utterance buried in white noise at 0 dB SNR."""
    wav = load_audio(test_audio)
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(wav.shape, generator=generator)
    noise *= wav.pow(2).mean().sqrt() / noise.pow(2).mean().sqrt()
    noisy = ((wav + noise) / 2).clamp(-1.0, 1.0)

    path = tmp_path_factory.mktemp("confidence") / "noisy.wav"
    sf.write(str(path), noisy.numpy(), SAMPLE_RATE)
    return str(path)


# ---------------------------------------------------------------- unit tests


def test_aggregate_confidence_is_geometric_mean():
    """Confidence is exp(mean(log p)), i.e. length-normalized."""
    assert aggregate_confidence([]) is None
    assert aggregate_confidence([0.0]) == pytest.approx(1.0)
    assert aggregate_confidence([math.log(0.5)]) == pytest.approx(0.5)
    # Two identical tokens must score the same as one: no length penalty.
    assert aggregate_confidence([math.log(0.5)] * 2) == pytest.approx(0.5)
    assert aggregate_confidence([math.log(0.25), math.log(1.0)]) == pytest.approx(0.5)


def test_hypothesis_is_tuple_compatible():
    """Hypothesis keeps the old positional layout of the decoder output."""
    hyp = Hypothesis("ok", [1, 2], [0, 3], [-0.1, -0.2])
    text, token_ids, token_frames, token_logprobs = hyp
    assert (text, token_ids, token_frames) == (hyp[0], hyp[1], hyp[2])
    assert token_logprobs == hyp.token_logprobs


def test_frames_to_words_confidence_is_optional():
    """Without log-probs the word confidence stays None."""

    class _CharTokenizer:
        def id_to_str(self, token_id):
            return "abc "[token_id]

    words = frames_to_words(_CharTokenizer(), [0, 1, 2], [0, 1, 2], 0.04)
    assert len(words) == 1
    assert words[0].confidence is None

    words = frames_to_words(
        _CharTokenizer(), [0, 1, 2], [0, 1, 2], 0.04, [math.log(0.5)] * 3
    )
    assert words[0].confidence == pytest.approx(0.5)


# --------------------------------------------------------------- model tests


@pytest.mark.parametrize("revision", REVISIONS)
def test_transcribe_returns_confidence(revision, test_audio):
    """Utterance confidence is reported even without word timestamps."""
    model = gigaam.load_model(revision, device="cpu")
    result = model.transcribe(test_audio)

    assert result.confidence is not None, "Utterance confidence should be set"
    assert 0.0 < result.confidence <= 1.0, f"Out of range: {result.confidence}"
    logger.info(f"{revision}: utterance confidence={result.confidence:.4f}")


@pytest.mark.parametrize("revision", REVISIONS)
def test_word_confidence_values(revision, test_audio):
    """Every word carries a confidence in (0, 1]."""
    model = gigaam.load_model(revision, device="cpu")
    result = model.transcribe(test_audio, word_timestamps=True)

    assert result.words, "Should have words"
    for word in result.words:
        assert word.confidence is not None, f"No confidence: {word}"
        assert 0.0 < word.confidence <= 1.0, f"Out of range: {word}"

    worst = min(result.words, key=lambda w: w.confidence)
    logger.info(
        f"{revision}: {len(result.words)} words, "
        f"least confident '{worst.text}'={worst.confidence:.4f}"
    )


@pytest.mark.parametrize("revision", REVISIONS)
def test_confidence_drops_on_noisy_audio(revision, test_audio, noisy_audio):
    """
    The score must be informative: the same utterance at 0 dB SNR has to score
    lower than the clean one.
    """
    model = gigaam.load_model(revision, device="cpu")
    clean = model.transcribe(test_audio).confidence
    noisy = model.transcribe(noisy_audio).confidence

    logger.info(f"{revision}: clean={clean:.4f} noisy={noisy:.4f}")
    assert noisy < clean, f"Noise did not lower confidence: {noisy} >= {clean}"


@pytest.mark.parametrize("revision", REVISIONS)
def test_longform_confidence(revision, long_audio):
    """Longform segments and their words both carry confidence."""
    model = gigaam.load_model(revision, device="cpu")
    result = model.transcribe_longform(long_audio, word_timestamps=True)

    assert result.segments, "Should have segments"
    for segment in result.segments:
        assert segment.confidence is not None, f"No confidence: {segment.text[:40]}"
        assert 0.0 < segment.confidence <= 1.0, f"Out of range: {segment.confidence}"
        for word in segment.words or []:
            assert word.confidence is not None, f"No confidence: {word}"
            assert 0.0 < word.confidence <= 1.0, f"Out of range: {word}"

    logger.info(
        f"{revision} longform: {len(result.segments)} segments, "
        f"min confidence={min(s.confidence for s in result.segments):.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
