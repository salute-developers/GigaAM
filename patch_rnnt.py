with open("gigaam_mlx.py", "r") as f:
    content = f.read()

rnnt_code = """
class RNNTJoint(nn.Module):
    def __init__(self, enc_hidden: int, pred_hidden: int, joint_hidden: int, num_classes: int):
        super().__init__()
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.joint_net_linear = nn.Linear(joint_hidden, num_classes)

    def __call__(self, encoder_out: mx.array, decoder_out: mx.array) -> mx.array:
        enc = mx.expand_dims(self.enc(encoder_out), 2)
        pred = mx.expand_dims(self.pred(decoder_out), 1)
        joint_rep = mx.maximum(enc + pred, 0.0)
        return self.joint_net_linear(joint_rep)

class RNNTDecoder(nn.Module):
    def __init__(self, num_classes: int, pred_hidden: int):
        super().__init__()
        self.blank_id = num_classes - 1
        self.pred_hidden = pred_hidden
        self.embed = nn.Embedding(num_classes, pred_hidden)
        self.lstm = nn.LSTM(pred_hidden, pred_hidden)

    def predict(self, x: Optional[int], state: Optional[Tuple[mx.array, mx.array]], batch_size: int = 1):
        if x is not None:
            emb = self.embed(mx.array([[x]]))
        else:
            emb = mx.zeros((batch_size, 1, self.pred_hidden))

        h, c = state if state is not None else (None, None)
        out, cell = self.lstm(emb, hidden=h, cell=c)
        return out, (out[:, -1, :], cell[:, -1, :])

class RNNTHead(nn.Module):
    def __init__(self, cfg: GigaAMConfig):
        super().__init__()
        self.decoder = RNNTDecoder(cfg.num_classes, cfg.rnnt_pred_hidden)
        self.joint = RNNTJoint(cfg.d_model, cfg.rnnt_pred_hidden, cfg.rnnt_joint_hidden, cfg.num_classes)

"""

# Insert before CTCHead
content = content.replace("class CTCHead(nn.Module):", rnnt_code + "\nclass CTCHead(nn.Module):")

with open("gigaam_mlx.py", "w") as f:
    f.write(content)
