from typing import Dict, Optional, Tuple

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


class RNNTJoint(nn.Module):
    """
    RNN-Transducer Joint Network Module.
    This module combines the outputs of the encoder and the prediction network using
    a linear transformation followed by ReLU activation and another linear projection.
    """

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

    def input_example(self):
        device = next(self.parameters()).device
        enc = torch.zeros(1, self.enc_hidden, 1)
        dec = torch.zeros(1, self.pred_hidden, 1)
        return enc.float().to(device), dec.float().to(device)

    def input_names(self):
        return ["enc", "dec"]

    def output_names(self):
        return ["joint"]

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

    def input_example(self):
        device = next(self.parameters()).device
        label = torch.tensor([[0]]).to(device)
        hidden_h = torch.zeros(1, 1, self.pred_hidden).to(device)
        hidden_c = torch.zeros(1, 1, self.pred_hidden).to(device)
        return label, hidden_h, hidden_c

    def input_names(self):
        return ["x", "h", "c"]

    def output_names(self):
        return ["dec", "h", "c"]

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
