import logging
import shutil

import numpy as np
import pytest

import gigaam
from gigaam.onnx_utils import infer_onnx, load_onnx
from gigaam.utils import download_short_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


@pytest.mark.parametrize("revision", ["emo", "v2_ssl", "v3_ctc", "v3_e2e_rnnt"])
def test_onnx_converting(revision, test_audio):
    """Test specific model revision loads and processes audio (partial models enabled)"""
    onnx_dir = "test_onnx_tmp"
    model = gigaam.load_model(revision, fp16_encoder=False)
    model.to_onnx(dir_path=onnx_dir)
    sessions, model_cfg = load_onnx(onnx_dir, revision)
    result = infer_onnx(test_audio, model_cfg, sessions)
    shutil.rmtree(onnx_dir)

    if "ssl" in revision:
        orig_embed = model.embed_audio(test_audio)[0].detach().cpu().numpy()
        diff = np.abs(orig_embed - result).max()
        assert diff < 0.01, f"{revision}: ONNX embed failed with diff {diff}"

    elif "emo" in revision:
        orig_probs = model.get_probs(test_audio)
        pred_probs = {model.id2name[i]: result[0, i] for i in range(len(model.id2name))}
        assert all(
            abs(orig_probs[em] - pred_probs[em]) < 1e-3 for em in orig_probs
        ), f"{revision}: ONNX emotions probs failed: {pred_probs}"

    else:
        orig_text = model.transcribe(test_audio)
        assert orig_text == result, f"{revision}: ONNX transcribe failed: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "partial"])
