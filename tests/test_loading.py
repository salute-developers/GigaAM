import logging
import os

import pytest
import torch
from omegaconf import OmegaConf

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
        if isinstance(model, gigaam.GigaAMDualASR):
            assert set(result) == {"raw", "e2e"}
            assert all(item.text.strip() for item in result.values())
        elif "e2e" in revision:
            assert _predictions[revision] == str(
                result
            ), f"Transcription failed ({revision}): {str(result)}"
        else:
            assert _predictions["asr"] == str(
                result
            ), f"Transcription failed ({revision}): {str(result)}"
        logger.info(f"{revision}: Transcription completed")


def test_local_lightning_checkpoint_ignores_stale_sibling_yaml(tmp_path):
    from test_streaming import tiny_model

    model = tiny_model()
    model_cfg = OmegaConf.to_container(model.cfg, resolve=True)
    checkpoint = tmp_path / "fine_tuned.ckpt"
    torch.save(
        {
            "hyper_parameters": {"model_name": "tiny_ctc", "model_cfg": model_cfg},
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    OmegaConf.save(
        OmegaConf.create({"model_name": "stale"}), checkpoint.with_suffix(".yaml")
    )

    loaded = gigaam.load_model(str(checkpoint), device="cpu", fp16_encoder=False)
    assert isinstance(loaded, gigaam.GigaAMASR)
    assert loaded.cfg.model_name == "tiny_ctc"


def test_local_portable_vad_checkpoint_uses_embedded_config(tmp_path):
    from gigaam.vad_utils import GigaAMVAD

    cfg = OmegaConf.create(
        {
            "model_name": "vad",
            "preprocessor": {
                "sample_rate": 16000,
                "features": 8,
                "n_fft": 32,
                "win_length": 32,
                "hop_length": 8,
                "center": False,
            },
            "encoder": {
                "feat_in": 8,
                "n_layers": 1,
                "d_model": 8,
                "subsampling": "conv1d",
                "subs_kernel_size": 3,
                "subsampling_factor": 2,
                "n_heads": 2,
                "pos_emb_max_len": 64,
                "conv_kernel_size": 3,
            },
            "head": {"feat_in": 8, "num_classes": 2},
            "vad": {"window_sec": 1.0},
        }
    )
    model = GigaAMVAD(cfg)
    checkpoint = tmp_path / "portable_vad.ckpt"
    torch.save(
        {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    OmegaConf.save(
        OmegaConf.create({"model_name": "stale"}), checkpoint.with_suffix(".yaml")
    )

    loaded = gigaam.load_model(str(checkpoint), device="cpu", fp16_encoder=False)
    assert isinstance(loaded, GigaAMVAD)
    assert loaded.cfg.model_name == "vad"


def test_local_bare_weights_still_use_sibling_yaml(tmp_path):
    from test_streaming import tiny_model

    model = tiny_model()
    checkpoint = tmp_path / "bare_weights.ckpt"
    torch.save(model.state_dict(), checkpoint)
    OmegaConf.save(model.cfg, checkpoint.with_suffix(".yaml"))

    loaded = gigaam.load_model(str(checkpoint), device="cpu", fp16_encoder=False)
    assert isinstance(loaded, gigaam.GigaAMASR)
    assert loaded.cfg.model_name == "tiny_ctc"


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
        "multilingual_ctc",
        "multilingual_ssl",
        "multilingual_large_ctc",
        "multilingual_large_ssl",
    ],
)
@pytest.mark.full
def test_model_revision_full(revision, test_audio):
    """Test specific model revision loads and processes audio (full models only)"""
    model = gigaam.load_model(revision)
    run_model_method(model, revision, test_audio)
    os.remove(os.path.join(gigaam._CACHE_DIR, f"{model.cfg.model_name}.ckpt"))


@pytest.mark.parametrize("revision", ["v4_rnnt", "v3_ctc", "v2_ssl", "emo"])
@pytest.mark.partial
def test_model_revision_partial(revision, test_audio):
    """Test specific model revision loads and processes audio (partial models enabled)"""
    model = gigaam.load_model(revision)
    run_model_method(model, revision, test_audio)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "partial"])
