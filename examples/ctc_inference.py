import argparse

import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=kwargs["nfilt"],
            window_fn=self.torch_windows[kwargs["window"]],
            mel_scale=mel_scale,
            norm=kwargs["mel_norm"],
            n_fft=kwargs["n_fft"],
            f_max=kwargs.get("highfreq", None),
            f_min=kwargs.get("lowfreq", 0),
            wkwargs=wkwargs,
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
            mel_scale=mel_scale, **kwargs,
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using GigaAM-CTC checkpoint"
    )
    parser.add_argument("--model_config", help="Path to GigaAM-CTC config file (.yaml)")
    parser.add_argument(
        "--model_weights", help="Path to GigaAM-CTC checkpoint file (.ckpt)"
    )
    parser.add_argument("--audio_path", help="Path to audio signal")
    parser.add_argument("--device", help="Device: cpu / cuda")
    return parser.parse_args()


def main(model_config: str, model_weights: str, device: str, audio_path: str):

    model = EncDecCTCModel.from_config_file(model_config)

    ckpt = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()

    transcription = model.transcribe([audio_path])[0]
    print(f"transcription: {transcription}")


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_config=args.model_config,
        model_weights=args.model_weights,
        device=args.device,
        audio_path=args.audio_path,
    )
