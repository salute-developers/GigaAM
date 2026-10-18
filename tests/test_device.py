import pytest
import torch
from omegaconf import OmegaConf

import gigaam


def _patch(monkeypatch, cuda: bool, xpu):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if xpu is None:
        monkeypatch.delattr(torch, "xpu", raising=False)
    else:
        monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)


@pytest.mark.parametrize(
    "cuda, xpu, expected",
    [
        (True, True, "cuda"),
        (False, True, "xpu"),
        (False, False, "cpu"),
        (False, None, "cpu"),  # a torch build without the xpu module
    ],
)
def test_default_device(monkeypatch, cuda, xpu, expected):
    _patch(monkeypatch, cuda, xpu)
    assert gigaam._normalize_device(None) == torch.device(expected)


def test_explicit_device_is_kept():
    assert gigaam._normalize_device("xpu:1") == torch.device("xpu:1")


@pytest.mark.parametrize(
    "device, enabled",
    [("cuda", True), ("cpu", False), ("xpu", False), ("mps", False)],
)
def test_flash_attn_only_on_cuda(device, enabled):
    cfg = OmegaConf.create({"encoder": {"flash_attn": True}})
    gigaam._apply_flash_policy(cfg, None, torch.device(device))
    assert cfg.encoder.flash_attn is enabled
