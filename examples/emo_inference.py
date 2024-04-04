import argparse
from typing import List, Union

import hydra
import soundfile
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


class SpecScaler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class GigaAMEmo(torch.nn.Module):
    def __init__(self, conf: Union[DictConfig, ListConfig]):
        super().__init__()
        self.id2name = conf.id2name
        self.feature_extractor = hydra.utils.instantiate(conf.feature_extractor)
        self.conformer = hydra.utils.instantiate(conf.encoder)
        self.linear_head = hydra.utils.instantiate(conf.classification_head)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, features, features_length=None):
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if not features_length:
            features_length = torch.ones(features.shape[0], device=self.device) * features.shape[-1]
        encoded, _ = self.conformer(audio_signal=features, length=features_length)
        encoded_pooled = torch.nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1]
        ).squeeze(-1)

        logits = self.linear_head(encoded_pooled)
        return logits

    def get_probs(self, audio_path: str) -> List[List[float]]:
        audio_signal, _ = soundfile.read(audio_path, dtype="float32")
        audio_tensor = torch.tensor(audio_signal).float().to(self.device)
        features = self.feature_extractor(audio_tensor)
        logits = self.forward(features)
        probs = torch.nn.functional.softmax(logits, dim=1).detach().tolist()
        return probs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using GigaAM-Emo checkpoint"
    )
    parser.add_argument("--model_config", help="Path to GigaAM-Emo config file (.yaml)")
    parser.add_argument(
        "--model_weights", help="Path to GigaAM-Emo checkpoint file (.ckpt)"
    )
    parser.add_argument("--audio_path", help="Path to audio signal")
    parser.add_argument("--device", help="Device: cpu / cuda")
    return parser.parse_args()


def main(model_config: str, model_weights: str, device: str, audio_path: str):
    conf = OmegaConf.load(model_config)
    model = GigaAMEmo(conf)
    ckpt = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        probs = model.get_probs(audio_path)[0]
    print(", ".join([f"{model.id2name[i]}: {p:.3f}" for i, p in enumerate(probs)]))


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_config=args.model_config,
        model_weights=args.model_weights,
        device=args.device,
        audio_path=args.audio_path,
    )
