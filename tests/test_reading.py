import logging

import librosa
import pytest
import torch
from torch.nn.functional import softmax

import gigaam
from gigaam.preprocess import SAMPLE_RATE
from gigaam.utils import download_short_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


@pytest.mark.parametrize("revision", ["emo", "v2_ctc", "v3_e2e_rnnt"])
def test_librosa_loading(revision, test_audio):
    """Test the outputs with librosa.load are close to load_audio"""
    model = gigaam.load_model(revision)
    wav_tns = torch.from_numpy(librosa.load(test_audio, sr=SAMPLE_RATE)[0])
    lengths = torch.full([1], wav_tns.shape[-1], device=model._device)
    with torch.no_grad():
        encoded, encoded_len = model(
            wav_tns.unsqueeze(0).to(model._device).to(model._dtype), lengths
        )
        if "emo" in revision:
            orig_probs = model.get_probs(test_audio)
            pred_probs = (
                softmax(model.head(encoded.mean(dim=-1)), dim=-1)
                .squeeze()
                .cpu()
                .tolist()
            )
            pred_probs = {
                model.id2name[i]: pred_probs[i] for i in range(len(model.id2name))
            }
            are_close = (
                max(abs(pred_probs[k] - orig_probs[k]) for k in orig_probs) < 1e-3
            )
            assert (
                are_close
            ), f"Emotions with librosa failed: {orig_probs} != {pred_probs}"
        else:
            orig_text = model.transcribe(test_audio)
            pred_text = model.decoding.decode(model.head, encoded, encoded_len)[0]
            assert (
                orig_text == pred_text
            ), f"Transcribe with librosa failed: {orig_text} != {pred_text}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
