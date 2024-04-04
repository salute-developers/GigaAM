import argparse

import hydra
import soundfile
import torch
from omegaconf import OmegaConf


class SpecScaler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using GigaAM checkpoint"
    )
    parser.add_argument("--encoder_config", help="Path to GigaAM config file (.yaml)")
    parser.add_argument(
        "--model_weights", help="Path to GigaAM checkpoint file (.ckpt)"
    )
    parser.add_argument("--audio_path", help="Path to audio signal")
    parser.add_argument("--device", help="Device: cpu / cuda")
    return parser.parse_args()


def main(encoder_config: str, model_weights: str, device: str, audio_path: str):
    conf = OmegaConf.load(encoder_config)

    encoder = hydra.utils.instantiate(conf.encoder)
    ckpt = torch.load(model_weights, map_location="cpu")
    encoder.load_state_dict(ckpt, strict=True)
    encoder.to(device)

    feature_extractor = hydra.utils.instantiate(conf.feature_extractor)

    audio_signal, _ = soundfile.read(audio_path, dtype="float32")
    features = feature_extractor(torch.tensor(audio_signal).float())
    features = features.to(device)

    encoded, _ = encoder.forward(
        audio_signal=features.unsqueeze(0),
        length=torch.tensor([features.shape[-1]]).to(device),
    )
    print(f"encoded signal shape: {encoded.shape}")


if __name__ == "__main__":
    args = _parse_args()
    main(
        encoder_config=args.encoder_config,
        model_weights=args.model_weights,
        device=args.device,
        audio_path=args.audio_path,
    )
