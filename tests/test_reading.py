import logging

import pytest
import torch
from torch.nn.functional import softmax

import gigaam
from gigaam.utils import AudioDataset, download_short_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


@pytest.mark.parametrize("revision", ["emo"])
def test_torchaudio_loading(revision, test_audio):
    """Torchaudio-loaded waveform should match get_probs(path) (ffmpeg load_audio)."""
    model = gigaam.load_model(revision)
    wav_tns = AudioDataset([test_audio])[0]
    lengths = torch.full([1], wav_tns.shape[-1], device=model._device)
    with torch.no_grad():
        encoded, _ = model(
            wav_tns.unsqueeze(0).to(model._device).to(model._dtype), lengths
        )
        orig_probs = model.get_probs(test_audio)
        pred_probs = (
            softmax(model.head(encoded.mean(dim=-1)), dim=-1).squeeze().cpu().tolist()
        )
        pred_probs = {
            model.id2name[i]: pred_probs[i] for i in range(len(model.id2name))
        }
        are_close = max(abs(pred_probs[k] - orig_probs[k]) for k in orig_probs) < 1e-3
        assert (
            are_close
        ), f"Emotions with torchaudio failed: {orig_probs} != {pred_probs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
