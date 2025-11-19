import logging
import os

import pytest

import gigaam
from gigaam.utils import download_short_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictions = {
    "emo": {
        "angry": 7.70451661082916e-05,
        "sad": 0.002205904107540846,
        "neutral": 0.9233596324920654,
        "positive": 0.07435736805200577,
    },
    "asr": "ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый",  # noqa: E501
    "v3_e2e_ctc": "Ничьих, не требуя похвал, счастлив уж я надеждой сладкой, Что дева с трепетом любви посмотрит, может быть украдкой На песни грешные мои. У лукоморья дуб зелёный.",  # noqa: E501
    "v3_e2e_rnnt": "Ничьих не требуя похвал, Счастлив уж я надеждой сладкой, Что дева с трепетом любви Посмотрит, может быть, украдкой На песни грешные мои. У лукоморья дуб зелёный.",  # noqa: E501
}


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


def run_model_method(model, revision, test_audio):
    if "ssl" in revision:
        result = model.embed_audio(test_audio)
        assert result is not None, "SSL embedding failed"
        logger.info(f"{revision}: SSL embedding completed")

    elif "emo" in revision:
        result = model.get_probs(test_audio)
        assert all(
            abs(result[em] - _predictions["emo"][em]) < 1e-3 for em in result
        ), f"Emotion probs failed: {result}"
        logger.info(f"{revision}: Emotion probs obtained")

    else:
        result = model.transcribe(test_audio)
        if "e2e" in revision:
            assert (
                _predictions[revision] == result
            ), f"Transcription failed ({revision}): {result}"
        else:
            assert (
                _predictions["asr"] == result
            ), f"Transcription failed ({revision}): {result}"
        logger.info(f"{revision}: Transcription completed")


@pytest.mark.parametrize(
    "revision",
    [
        "emo",
        "v1_ctc",
        "v1_rnnt",
        "v1_ssl",
        "v2_ctc",
        "v2_rnnt",
        "v2_ssl",
        "v3_ctc",
        "v3_rnnt",
        "v3_e2e_ctc",
        "v3_e2e_rnnt",
        "v3_ssl",
    ],
)
@pytest.mark.full
def test_model_revision_full(revision, test_audio):
    """Test specific model revision loads and processes audio (full models only)"""
    model = gigaam.load_model(revision)
    run_model_method(model, revision, test_audio)
    os.remove(os.path.join(gigaam._CACHE_DIR, f"{revision}.ckpt"))


@pytest.mark.parametrize("revision", ["emo", "v2_ssl", "v3_ctc", "v3_e2e_rnnt"])
@pytest.mark.partial
def test_model_revision_partial(revision, test_audio):
    """Test specific model revision loads and processes audio (partial models enabled)"""
    model = gigaam.load_model(revision)
    run_model_method(model, revision, test_audio)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "partial"])
