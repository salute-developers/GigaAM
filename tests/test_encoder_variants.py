import hydra
import pytest
import torch
from test_language import tiny_conditioned

from gigaam.encoder import (
    OrthogonalLinear,
    RotaryPositionalEmbedding,
    RotaryPositionMultiHeadAttention,
    StridingSubsampling,
    ZapformerAttention,
    ZapformerEncoder,
)
from gigaam.streaming import EncoderStreamState
from gigaam.utils import apply_rotary_pos_emb


@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("sdpa", [True, False])
def test_rotary_projection_order(before, sdpa):
    torch.manual_seed(7)
    model = RotaryPositionMultiHeadAttention(
        2, 8, torch_sdpa_attn=sdpa, apply_rope_before_projections=before
    ).eval()
    x = torch.randn(2, 5, 8)
    position = RotaryPositionalEmbedding(4, 10000)
    position.extend_pe(5, x.device)
    _, (cos, sin) = position(x)
    qk = x.transpose(0, 1).reshape(5, 2, 2, 4)
    q, k = apply_rotary_pos_emb(qk, qk, cos, sin) if before else (qk, qk)
    q, k, v = model.forward_qkv(
        q.reshape(5, 2, 8).transpose(0, 1), k.reshape(5, 2, 8).transpose(0, 1), x
    )
    if not before:
        q, k = apply_rotary_pos_emb(
            q.permute(2, 0, 1, 3), k.permute(2, 0, 1, 3), cos, sin
        )
        q, k = q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3)
    expected = model.linear_out(
        ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 8)
    )
    torch.testing.assert_close(model(x, x, x, [cos, sin]), expected)
    default = RotaryPositionMultiHeadAttention(2, 8)
    assert default.apply_rope_before_projections


def test_zapformer_padding_and_gradients():
    torch.manual_seed(7)
    model = hydra.utils.instantiate(tiny_conditioned().cfg.encoder).eval()
    x, lengths = torch.randn(2, 8, 35), torch.tensor([35, 29])
    output, actual_lengths = model(x, lengths)
    assert output.shape == (2, 8, 9)
    assert actual_lengths.tolist() == [9, 8]
    altered = x.clone()
    altered[1, :, 29:] = 100
    torch.testing.assert_close(model(altered, lengths)[0], output)
    output.square().mean().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_streaming_subsampling_zero_radius_does_not_replay_frames():
    torch.manual_seed(11)
    model = StridingSubsampling("conv1d", 2, 2, 8, 8, 8).eval()
    features = torch.randn(1, 8, 16)
    lengths = torch.tensor([16])
    state = EncoderStreamState()
    first, first_len = model(
        features[..., :8].transpose(1, 2), lengths.new_tensor([8]), state
    )
    second, second_len = model(
        features[..., 8:].transpose(1, 2), lengths.new_tensor([5]), state
    )
    whole, whole_len = model(features.transpose(1, 2), lengths.new_tensor([13]))
    torch.testing.assert_close(torch.cat((first, second), dim=1), whole)
    assert torch.cat((first_len, second_len)).sum().item() == whole_len.item()
    assert all(cache.shape[-1] == 0 for cache in state.frontend)


@pytest.mark.parametrize("offset", [0, 2])
def test_zapformer_sdpa_matches_math(offset):
    torch.manual_seed(19)
    attention = ZapformerAttention(8, 2, 4, 3, 0.1).eval()
    x_qk, x_vg = torch.randn(2, 7, 8), torch.randn(2, 7, 8)
    cos, sin = torch.randn(1, 1, 7, 2), torch.randn(1, 1, 7, 2)
    mask = torch.ones(2, 7 - offset, 7, dtype=torch.bool).triu(1 + offset)
    mask[1, :, 5:] = True
    mask[1, -1] = True
    attention.torch_sdpa_attn = False
    expected = attention(x_qk, x_vg, cos, sin, mask, offset)
    attention.torch_sdpa_attn = True
    torch.testing.assert_close(attention(x_qk, x_vg, cos, sin, mask, offset), expected)


def test_zapformer_rejects_flash():
    from gigaam import _apply_flash_policy

    with pytest.raises(NotImplementedError, match="not supported for Zapformer"):
        ZapformerEncoder(flash_attn=True)
    with pytest.raises(NotImplementedError, match="not supported for Zapformer"):
        _apply_flash_policy(tiny_conditioned().cfg, True, torch.device("cpu"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@torch.inference_mode()
def test_zapformer_transposed_projection_matches_cpu_on_mps():
    torch.manual_seed(7)
    projection = OrthogonalLinear(768, 768).eval()
    x = torch.randn(1, 158, 768)
    expected = projection(x, transpose=True)
    actual = projection.to("mps")(x.to("mps"), transpose=True).cpu()
    torch.testing.assert_close(actual, expected)
