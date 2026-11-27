from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


class CTCHead(nn.Module):
    """
    CTC Head module for Connectionist Temporal Classification.
    """

    def __init__(self, feat_in: int, num_classes: int):
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(feat_in, num_classes, kernel_size=1)
        )

    def forward(self, encoder_output: Tensor) -> Tensor:
        return torch.nn.functional.log_softmax(
            self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
        )

    def input_example(self):
        conv = self.decoder_layers[0]
        return (conv.weight.new_zeros(2, conv.in_channels, 24),)

    def input_names(self):
        return ["encoded"]

    def output_names(self):
        return ["log_probs"]

    def dynamic_axes(self):
        return {
            "encoded": {0: "batch", 2: "frames"},
            "log_probs": {0: "batch", 1: "frames"},
        }


class DualASRExport(nn.Module):
    output_names = ["encoded_len", "head_outputs"]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ctc = all(isinstance(head, CTCHead) for head in model.heads.values())

    def forward(self, features, lengths, *language):
        encoded, lengths = self.model.forward_for_export(features, lengths, *language)
        encoded = encoded.transpose(1, 2)
        heads = self.model.heads.values()
        if self.ctc:
            outputs = _ctc_export_labels(encoded, heads)
        else:
            outputs = [head.joint.enc(encoded) for head in heads]
        return lengths, torch.stack(outputs)

    def dynamic_axes(self):
        axes = self.model.encoder.dynamic_axes()
        del axes["encoded"]
        axes["head_outputs"] = {1: "batch", 2: "frames"}
        return axes


def _ctc_export_labels(encoded, heads):
    return [
        nn.functional.linear(encoded, conv.weight[..., 0], conv.bias).argmax(-1)
        for conv in (head.decoder_layers[0] for head in heads)
    ]


class CTCStreamExport(nn.Module):
    def __init__(self, model, frontend_count):
        super().__init__()
        self.model = model
        self.frontend_count = frontend_count

    def forward(self, features, lengths, context, position, valid, *caches):
        split = self.frontend_count
        features, lengths, *frontend = self.model.encoder.forward_frontend_for_export(
            features, lengths, *caches[:split]
        )
        encoded, lengths, position, valid, *encoder = self.model.forward_for_export(
            features, lengths, context, position, valid, *caches[split:]
        )
        labels = _ctc_export_labels(encoded.transpose(1, 2), self.model.heads.values())
        return torch.stack(labels, dim=1), lengths, position, valid, *frontend, *encoder


class RNNTJoint(nn.Module):
    """
    RNN-Transducer Joint Network Module.
    This module combines the outputs of the encoder and the prediction network using
    a linear transformation followed by ReLU activation and another linear projection.
    """

    supports_frame_windows = True

    def __init__(
        self, enc_hidden: int, pred_hidden: int, joint_hidden: int, num_classes: int
    ):
        super().__init__()
        self.enc_hidden = enc_hidden
        self.pred_hidden = pred_hidden
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(joint_hidden, num_classes))

    def joint(self, encoder_out: Tensor, decoder_out: Tensor) -> Tensor:
        """
        Combine the encoder and prediction network outputs into a joint representation.
        """
        enc = self.enc(encoder_out).unsqueeze(2)
        pred = self.pred(decoder_out).unsqueeze(1)
        return self.joint_net(enc + pred).log_softmax(-1)

    def input_example(self, batch_size: int = 8) -> Tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        enc = torch.zeros(batch_size, self.enc_hidden, 1)
        dec = torch.zeros(batch_size, self.pred_hidden, 1)
        return enc.float().to(device), dec.float().to(device)

    def input_names(self) -> List[str]:
        return ["enc", "dec"]

    def output_names(self) -> List[str]:
        return ["joint"]

    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "enc": {0: "batch_size"},
            "dec": {0: "batch_size"},
            "joint": {0: "batch_size"},
        }

    def forward(self, enc: Tensor, dec: Tensor) -> Tensor:
        return self.joint(enc.transpose(1, 2), dec.transpose(1, 2))


class RNNTDecoder(nn.Module):
    """
    RNN-Transducer Decoder Module.
    This module handles the prediction network part of the RNN-Transducer architecture.
    """

    def __init__(self, pred_hidden: int, pred_rnn_layers: int, num_classes: int):
        super().__init__()
        self.blank_id = num_classes - 1
        self.pred_hidden = pred_hidden
        self.embed = nn.Embedding(num_classes, pred_hidden, padding_idx=self.blank_id)
        self.lstm = nn.LSTM(pred_hidden, pred_hidden, pred_rnn_layers)

    def predict(
        self,
        x: Optional[Tensor],
        state: Optional[Tensor],
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make predictions based on the current input and previous states.
        If no input is provided, use zeros as the initial input.
        """
        if x is not None:
            emb: Tensor = self.embed(x)
        else:
            emb = torch.zeros(
                (batch_size, 1, self.pred_hidden), device=next(self.parameters()).device
            )
        g, hid = self.lstm(emb.transpose(0, 1), state)
        return g.transpose(0, 1), hid

    def input_example(self, batch_size: int = 8) -> Tuple[Tensor, Tensor, Tensor]:
        device = next(self.parameters()).device
        label = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
        hidden_h = torch.zeros(self.lstm.num_layers, batch_size, self.pred_hidden).to(
            device
        )
        hidden_c = torch.zeros(self.lstm.num_layers, batch_size, self.pred_hidden).to(
            device
        )
        return label, hidden_h, hidden_c

    def input_names(self) -> List[str]:
        return ["x", "hi", "ci"]

    def output_names(self) -> List[str]:
        return ["dec", "ho", "co"]

    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "x": {0: "batch_size"},
            "hi": {1: "batch_size"},
            "ci": {1: "batch_size"},
            "dec": {0: "batch_size"},
            "ho": {1: "batch_size"},
            "co": {1: "batch_size"},
        }

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        ONNX-specific forward with x, state = (h, c) -> x, h, c.
        """
        emb = self.embed(x)
        g, (h, c) = self.lstm(emb.transpose(0, 1), (h, c))
        return g.transpose(0, 1), h, c


class RNNTHead(nn.Module):
    """
    RNN-Transducer Head Module.
    This module combines the decoder and joint network components of the RNN-Transducer architecture.
    """

    def __init__(self, decoder: Dict[str, int], joint: Dict[str, int]):
        super().__init__()
        self.decoder = RNNTDecoder(**decoder)
        self.joint = RNNTJoint(**joint)


class RNNTAdvance(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.heads = model.heads
        self.vocab_size = max(
            head.decoder.embed.num_embeddings for head in self.heads.values()
        )
        self.max_symbols = min(dec.max_symbols for dec in model.decodings.values())

    def _pad_vocab(self, value, fill=0.0):
        return nn.functional.pad(
            value, (0, self.vocab_size - value.size(-1)), value=fill
        )

    def forward(self, projected, lengths, position, previous, repeats, hidden, cell):
        lstms = [head.decoder.lstm for head in self.heads.values()]
        joints = [head.joint for head in self.heads.values()]
        weights = torch.stack(
            [torch.cat((m.weight_ih_l0, m.weight_hh_l0), 1).T for m in lstms]
        )
        bias = torch.stack([m.bias_ih_l0 + m.bias_hh_l0 for m in lstms])[:, None]
        embeddings = torch.stack(
            [self._pad_vocab(h.decoder.embed.weight.T).T for h in self.heads.values()]
        )
        pred_weights = torch.stack([joint.pred.weight.T for joint in joints])
        pred_bias = torch.stack([joint.pred.bias for joint in joints])[:, None]
        output = [joint.joint_net[1] for joint in joints]
        join_weights = torch.stack(
            [self._pad_vocab(layer.weight.T) for layer in output]
        )
        join_bias = torch.stack(
            [self._pad_vocab(layer.bias, -torch.inf) for layer in output]
        )[:, None]
        blank = position.new_tensor(
            [head.decoder.blank_id for head in self.heads.values()]
        )[:, None]
        offsets = torch.arange(8, device=position.device)
        tokens = []
        for _ in range(4):
            frames = position[..., None] + offsets
            index = torch.minimum(frames, (lengths - 1).clamp_min(0)[None, :, None])
            window = projected.gather(
                2, index[..., None].expand(-1, -1, -1, projected.size(-1))
            )
            embed = embeddings.gather(
                1, previous[..., None].expand(-1, -1, embeddings.size(-1))
            )
            embed = embed.masked_fill((previous == blank)[..., None], 0)
            i, f, g, o = (torch.cat((embed, hidden), -1) @ weights + bias).split(
                [weights.shape[-1] // 4] * 4, dim=-1
            )
            next_cell = f.sigmoid() * cell + i.sigmoid() * g.tanh()
            next_hidden = o.sigmoid() * next_cell.tanh()
            pred = next_hidden @ pred_weights + pred_bias
            joined = (
                (window + pred.unsqueeze(2))
                .relu()
                .reshape(projected.size(0), -1, pred.size(-1))
            )
            labels = (joined @ join_weights + join_bias).argmax(-1).reshape_as(index)
            valid = (labels != blank[..., None]) & (frames < lengths[None, :, None])
            emitted = valid.any(-1)
            delta = torch.where(valid, 8 - offsets, 0).argmax(-1)
            symbol = labels.gather(-1, delta[..., None])[..., 0]
            repeats = torch.where(delta == 0, repeats + 1, 1)
            advance = repeats >= self.max_symbols
            position = torch.where(
                emitted, position + delta + advance.long(), position + 8
            )
            repeats = torch.where(emitted & ~advance, repeats, 0)
            previous = torch.where(emitted, symbol, previous)
            hidden = torch.where(emitted[..., None], next_hidden, hidden)
            cell = torch.where(emitted[..., None], next_cell, cell)
            tokens.append(torch.where(emitted, symbol, -1))
        return position, previous, repeats, hidden, cell, torch.stack(tokens)


class RNNTDecodeExport(nn.Module):
    def __init__(self, model, example):
        super().__init__()
        head = next(iter(model.heads.values()))
        self.register_buffer(
            "blank",
            example.new_tensor(
                [[head.decoder.blank_id] for head in model.heads.values()],
                dtype=torch.long,
            ),
        )
        self.hidden_size = head.decoder.pred_hidden
        self.joint_size = head.joint.enc.out_features
        self.num_heads = len(model.heads)
        step = RNNTAdvance(model)
        self.max_symbols = step.max_symbols
        projected, lengths = self.input_example(example)
        self.step = torch.jit.trace(
            step,
            (projected, lengths, *self.initial_state(projected)),
            check_trace=False,
        )

    def initial_state(self, projected):
        position = torch.zeros_like(projected[:, :, 0, 0], dtype=torch.long)
        hidden = projected.new_zeros(
            (position.size(0), position.size(1), self.hidden_size)
        )
        return position, position + self.blank, position, hidden, hidden

    def forward(self, projected, lengths):
        _, previous, _, hidden, cell = self.initial_state(projected)
        return self.decode(projected, lengths, previous, hidden, cell)[0]

    def decode(self, projected, lengths, previous, hidden, cell):
        position = torch.zeros_like(previous)
        repeats = position
        limit = projected.size(2) * (self.max_symbols + 1)
        tokens = position.new_full((limit + 3, position.size(0), position.size(1)), -1)
        start = 0
        while start < limit and bool(torch.any(position < lengths)):
            position, previous, repeats, hidden, cell, labels = self.step(
                projected, lengths, position, previous, repeats, hidden, cell
            )
            indices = (torch.arange(4, device=position.device) + start).view(4, 1, 1)
            tokens = tokens.scatter(0, indices.expand_as(labels), labels.long())
            start += 4
        return tokens, previous, hidden, cell

    def input_example(self, example):
        return example.new_zeros(
            (self.num_heads, 2, 16, self.joint_size)
        ), example.new_full((2,), 16, dtype=torch.long)

    def dynamic_axes(self):
        return {
            "projected": {1: "batch", 2: "frames"},
            "lengths": {0: "batch"},
            "tokens": {0: "steps", 2: "batch"},
        }


class RNNTStreamDecodeExport(nn.Module):
    input_names = ["encoded", "lengths", "previous", "hidden", "cell"]
    output_names = ["tokens", "next_previous", "next_hidden", "next_cell"]

    def __init__(self, model, example):
        super().__init__()
        self.decoder = RNNTDecodeExport(model, example)
        self.projections = nn.ModuleList(
            head.joint.enc for head in model.heads.values()
        )
        self.encoder_dim = model.encoder.feat_out

    def forward(self, encoded, lengths, previous, hidden, cell):
        encoded = encoded.transpose(1, 2)
        projected = torch.stack(
            [projection(encoded) for projection in self.projections]
        )
        return self.decoder.decode(projected, lengths, previous, hidden, cell)

    def input_example(self, example):
        projected, lengths = self.decoder.input_example(example)
        _, previous, _, hidden, cell = self.decoder.initial_state(projected)
        encoded = example.new_zeros(
            lengths.numel(), self.encoder_dim, projected.size(2)
        )
        return encoded, lengths, previous, hidden, cell

    def dynamic_axes(self):
        axes = self.decoder.dynamic_axes()
        del axes["projected"]
        axes["encoded"] = {0: "batch", 2: "frames"}
        for name in (*self.input_names[2:], *self.output_names[1:]):
            axes[name] = {1: "batch"}
        return axes
