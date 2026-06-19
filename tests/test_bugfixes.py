import numpy as np
import torch
from omegaconf import OmegaConf

from gigaam.decoding import DEFAULT_MAX_SYMBOLS_PER_STEP, RNNTGreedyDecoding, Tokenizer
from gigaam.onnx_utils import (
    _decode_ctc_batch,
    _decode_rnnt_batch,
    _max_symbols_per_step,
    _split_state as split_numpy_state,
)
from gigaam.preprocess import SpecScaler


def test_rnnt_split_state_copies_torch_slices():
    h = torch.zeros(2, 3, 4)
    c = torch.zeros(2, 3, 4)

    split = RNNTGreedyDecoding._split_state((h, c))
    split[1][0].fill_(1.0)
    split[1][1].fill_(1.0)

    assert h.sum().item() == 0.0
    assert c.sum().item() == 0.0


def test_rnnt_split_state_copies_numpy_slices():
    h = np.zeros((2, 3, 4), dtype=np.float32)
    c = np.zeros((2, 3, 4), dtype=np.float32)

    split = split_numpy_state((h, c))
    split[1][0].fill(1.0)
    split[1][1].fill(1.0)

    assert not np.shares_memory(split[1][0], h)
    assert not np.shares_memory(split[1][1], c)
    assert h.sum() == 0.0
    assert c.sum() == 0.0


def test_onnx_rnnt_max_symbols_per_step_comes_from_config():
    cfg = OmegaConf.create({"decoding": {"max_symbols_per_step": 7}})

    assert _max_symbols_per_step(cfg) == 7
    assert _max_symbols_per_step(OmegaConf.create({})) == DEFAULT_MAX_SYMBOLS_PER_STEP


def test_onnx_ctc_hypotheses_include_token_frames():
    tokenizer = Tokenizer(["a", "b"])
    labels = np.array([[0, 0, 2, 1, 2]], dtype=np.int64)
    lengths = np.array([5], dtype=np.int64)

    decoded = _decode_ctc_batch(
        labels,
        lengths,
        tokenizer,
        return_hypotheses=True,
    )

    assert decoded == [("ab", [0, 1], [0, 3])]


def test_onnx_rnnt_hypotheses_include_token_frames():
    class Node:
        def __init__(self, name, node_type="tensor(float)"):
            self.name = name
            self.type = node_type

    class PredSession:
        def get_inputs(self):
            return [
                Node("labels", "tensor(int64)"),
                Node("h"),
                Node("c"),
            ]

        def get_outputs(self):
            return [Node("dec"), Node("h"), Node("c")]

        def run(self, _, inputs):
            batch = inputs["labels"].shape[0]
            h = np.zeros((1, batch, 2), dtype=np.float32)
            c = np.zeros((1, batch, 2), dtype=np.float32)
            dec = np.zeros((batch, 1, 2), dtype=np.float32)
            return [dec, h, c]

    class JointSession:
        def __init__(self):
            self.calls = 0

        def get_inputs(self):
            return [Node("enc"), Node("dec")]

        def get_outputs(self):
            return [Node("joint")]

        def run(self, _, inputs):
            batch = inputs["enc"].shape[0]
            logits = np.zeros((batch, 1, 1, 2), dtype=np.float32)
            logits[:, 0, 0, 0 if self.calls == 0 else 1] = 1.0
            self.calls += 1
            return [logits]

    cfg = OmegaConf.create(
        {
            "head": {"decoder": {"pred_hidden": 2, "pred_rnn_layers": 1}},
            "decoding": {"max_symbols_per_step": 2},
        }
    )
    tokenizer = Tokenizer(["a"])
    decoded = _decode_rnnt_batch(
        np.zeros((1, 2, 2), dtype=np.float32),
        np.array([2], dtype=np.int64),
        cfg,
        [None, PredSession(), JointSession()],
        tokenizer,
        return_hypotheses=True,
    )

    assert decoded == [("a", [0], [0])]


def test_spec_scaler_does_not_mutate_input():
    x = torch.tensor([0.0, 2.0, 1e10])
    original = x.clone()

    SpecScaler()(x)

    assert torch.equal(x, original)
