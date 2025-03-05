from subprocess import CalledProcessError, run
from typing import Tuple, Union

import torch
import torchaudio
import numpy as np
from torch import Tensor, nn
from numpy.typing import NDArray

SAMPLE_RATE = 16000


def load_audio_file(path: str, sample_rate: int):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        audio = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError("Failed to load audio") from exc
    
    return audio


def load_audio(
    audio: Union[str, NDArray],
    sample_rate: int = SAMPLE_RATE,
    return_format: str = "float",
) -> Tensor:
    """
    Load an audio file and resample it to the specified sample rate.
    """
    if isinstance(audio, str):
        audio_bytes = load_audio_file(audio, sample_rate)
    elif isinstance(audio, np.ndarray):
        audio = (audio.flatten() * 32767).astype(np.int16)
        audio_bytes = audio.tobytes()
    else:
        raise TypeError("Argument 'audio' must be type of str | np.ndarray")

    if return_format == "float":
        return torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0

    return torch.frombuffer(audio_bytes, dtype=torch.int16)


class SpecScaler(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """

    def __init__(self, sample_rate: int, features: int):
        super().__init__()
        self.hop_length = sample_rate // 100
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=sample_rate // 40,
                win_length=sample_rate // 40,
                hop_length=self.hop_length,
                n_mels=features,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def forward(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)
