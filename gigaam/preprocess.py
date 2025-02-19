from subprocess import CalledProcessError, run
from typing import Iterable, Tuple, Union

import torch
import torchaudio
from torch import Tensor, nn, from_numpy, mean, float32, int16
from numpy import ndarray, asarray

SAMPLE_RATE = 16_000


def load_audio(
    audio: Union[str, Tensor, ndarray, Iterable],
    sample_rate: int = SAMPLE_RATE,
    return_format: str = "float",
    result_sample_rate: int = SAMPLE_RATE,
) -> Tensor:
    """
    Load an audio file and resample it to the specified sample rate (16kHz by default).
    If the input is not a string (path), it is assumed to be an iterable object
    containing audio samples with provided sample rate.
    """
    if isinstance(audio, str):
        return load_audio_from_path(audio, result_sample_rate, return_format)

    if not isinstance(audio, Tensor):
        if not isinstance(audio, ndarray):
            try:
                audio = asarray(audio)
            except ValueError as exc:
                raise ValueError(
                    "Passed audio content is not convertible to numpy.ndarray!"
                    f"Expected Iterable Python object or numpy.ndarray or torch.Tensor, got {type(audio)}"
                ) from exc

        assert "float" in audio.dtype.name or "int" in audio.dtype.name, f"Audio should be a float or int array, got {audio.dtype} array"

        audio = from_numpy(audio)

    if not audio.dtype.is_floating_point and return_format == "float" and audio.abs().max() > 1:
        audio = audio.float() / 32768.0
    elif audio.dtype.is_floating_point and return_format == "int" and audio.abs().max() <= 1.0:
        audio = audio.mul(32768.0)

    if audio.ndim != 1:
        audio = mean(audio, dim=0)

    audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=result_sample_rate)

    if audio.dtype != float32 and return_format == "float":
        audio = audio.to(dtype=float32)
    elif audio.dtype != int16 and return_format == "int":
        audio = audio.round().to(dtype=int16)

    return audio


def load_audio_from_path(
    audio_path: str, sample_rate: int = SAMPLE_RATE, return_format: str = "float"
) -> Tensor:
    """
    Load an audio file and resample it to the specified sample rate.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
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

    if return_format == "float":
        return torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0

    return torch.frombuffer(audio, dtype=torch.int16)


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
