import logging

import numpy as np
import pytest
import torch
from scipy import signal

import gigaam
from gigaam.utils import AudioDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio(duration=3.0, sr=16000):
    """Generate synthetic test audio with different characteristics"""
    t = np.linspace(0, duration, int(sr * duration))
    audio = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 660 * t)
        + 0.1 * np.random.normal(0, 0.1, len(t))
    )
    envelope = signal.windows.tukey(len(audio), alpha=0.1)
    return (audio * envelope).astype(np.float32)


def create_test_batches(batch_size=4, max_duration=3.0, sr=16000):
    """Create test batches with different lengths"""
    durations = np.linspace(max_duration * 0.5, max_duration, batch_size)
    ds = AudioDataset([generate_test_audio(duration=dr, sr=sr) for dr in durations])
    return ds.collate([ds[i] for i in range(len(ds))])


def custom_forward(model, features, feature_lengths):
    """Custom forward pass for batching comparison"""
    out_feat, out_lns = [], []
    for i in range(len(features)):
        of, ol = model.preprocessor(
            features[i : i + 1, : feature_lengths[i]], feature_lengths[i : i + 1]
        )
        if model._device.type == "cpu":
            of, ol = model.encoder.pre_encode(of.transpose(1, 2), ol)
        else:
            with torch.autocast(device_type="cuda"):
                of, ol = model.encoder.pre_encode(of.transpose(1, 2), ol)
        out_feat.append(of.transpose(1, 2))
        out_lns.append(ol.item())

    features_padded = torch.zeros(len(features), out_feat[0].shape[1], max(out_lns)).to(
        features.device
    )
    for i in range(len(features)):
        features_padded[i, :, : out_lns[i]] = out_feat[i][:, :, : out_lns[i]]
    feature_lengths_tensor = torch.tensor(out_lns).to(features.device)

    subs_forward = model.encoder.pre_encode.forward
    model.encoder.pre_encode.forward = lambda x, lengths: (x, lengths)

    if model._device.type == "cpu":
        enc_out = model.encoder(features_padded, feature_lengths_tensor)
    else:
        with torch.autocast(device_type=model._device.type, dtype=torch.float16):
            enc_out = model.encoder(features_padded, feature_lengths_tensor)

    model.encoder.pre_encode.forward = subs_forward
    return enc_out


def compare_outputs(output1, output2, atol=0.03):
    """Compare two model outputs with tolerance"""
    feat1, lens1 = output1
    feat2, lens2 = output2
    assert (lens1 == lens2).all(), f"Length mismatch: {lens1} != {lens2}"
    min_len = min(lens1.item(), lens2.item())
    assert (
        feat1.shape[:2] == feat2.shape[:2]
    ), f"Shape mismatch: {feat1.shape} vs {feat2.shape}"

    feat1, feat2 = feat1[:, :, :min_len], feat2[:, :, :min_len]
    abs_diff = torch.abs(feat1 - feat2).max().item()
    close = abs_diff < atol
    return close, {
        "max_absolute_difference": abs_diff,
    }


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_model_batching(revision, batch_size):
    """Test batching correctness for different models and batch sizes"""
    torch.manual_seed(0)
    model = gigaam.load_model(revision)
    model.eval()
    device = next(model.parameters()).device

    features, feature_lengths = create_test_batches(
        batch_size=batch_size, max_duration=5.0
    )
    features, feature_lengths = features.to(device), feature_lengths.to(device)

    with torch.no_grad():
        batch_output = custom_forward(model, features, feature_lengths)

    # Compare with individual processing
    for i in range(batch_size):
        single_features = features[i : i + 1][:, : feature_lengths[i]]
        single_lengths = feature_lengths[i : i + 1]

        with torch.no_grad():
            single_output = custom_forward(model, single_features, single_lengths)

        if batch_size == 1:
            close, metrics = compare_outputs(batch_output, single_output)
        else:
            batch_element = (batch_output[0][i : i + 1], batch_output[1][i : i + 1])
            close, metrics = compare_outputs(batch_element, single_output)

        assert close, (
            f"Batch inconsistency for sample {i}: "
            f"abs_diff={metrics['max_absolute_difference']:.4f}, "
        )

    logger.info(f"Batching test passed: {revision} batch_size={batch_size}")


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_batching_edge_cases(revision):
    """Test batching with edge cases"""
    model = gigaam.load_model(revision)
    model.eval()
    device = next(model.parameters()).device

    # Test with very short sequences
    features = torch.randn(2, 5000).to(device)
    feature_lengths = torch.tensor([3200, 5000], dtype=torch.int32).to(device)

    with torch.no_grad():
        output = custom_forward(model, features, feature_lengths)

    assert output[0].shape[0] == 2, "Should handle variable length sequences"
    logger.info(f"Edge case test passed: {revision}")


@pytest.mark.parametrize("max_duration", [0.5, 1.0])
def test_different_audio_lengths(max_duration):
    """Test batching with different audio durations"""
    model = gigaam.load_model("v3_e2e_ctc")
    model.eval()
    device = next(model.parameters()).device

    features, feature_lengths = create_test_batches(
        batch_size=2, max_duration=max_duration
    )
    features, feature_lengths = features.to(device), feature_lengths.to(device)

    with torch.no_grad():
        output = custom_forward(model, features, feature_lengths)

    assert output[0].shape[0] == 2, f"Failed for max_duration={max_duration}"
    logger.info(f"Audio length test passed: max_duration={max_duration}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
