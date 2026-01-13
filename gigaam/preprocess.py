import warnings
from subprocess import CalledProcessError, run
from typing import List, Tuple, Union

import torch
import torchaudio
from torch import Tensor, nn

SAMPLE_RATE = 16000


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> Tensor:
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0


def load_multichannel_audio(
    audio_input: Union[str, List[str]], 
    sample_rate: int = SAMPLE_RATE
) -> Tuple[List[Tensor], int]:
    """
    Load multichannel audio from either:
    - A single stereo/multichannel file (str)
    - Multiple separate audio files (List[str])
    
    Returns:
        Tuple of (list of channel tensors, max_length)
    """
    if isinstance(audio_input, str):
        # Try to load with torchaudio first (more reliable for multichannel)
        try:
            import torchaudio
            waveform, file_sr = torchaudio.load(audio_input)
            
            # Resample if needed
            if file_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(file_sr, sample_rate)
                waveform = resampler(waveform)
            
            # Convert to list of channel tensors
            num_channels = waveform.shape[0]
            channels = [waveform[i] for i in range(num_channels)]
            
            max_length = max(len(ch) for ch in channels)
            return channels, max_length
        except Exception:
            # Fallback to ffmpeg approach
            pass
        
        # Fallback: Load multichannel file with ffmpeg
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            audio_input,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-",
        ]
        try:
            audio_bytes = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as exc:
            raise RuntimeError(f"Failed to load audio from {audio_input}") from exc
        
        # Try to determine number of channels from file metadata
        # Default to stereo (2 channels) for common cases
        num_channels = 2  # Default assumption
        
        # Try ffprobe if available
        cmd_probe = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_input
        ]
        try:
            result = run(cmd_probe, capture_output=True, check=True)
            num_channels = int(result.stdout.strip().split()[0])
        except (CalledProcessError, ValueError, IndexError):
            # If ffprobe fails, try to infer from data size
            # This is a heuristic - may not always work
            pass
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            audio_data = torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0
        
        # Reshape to channels
        if num_channels > 1 and len(audio_data) % num_channels == 0:
            audio_data = audio_data.view(-1, num_channels).transpose(0, 1)
            channels = [audio_data[i] for i in range(num_channels)]
        else:
            # Single channel or couldn't determine
            channels = [audio_data]
        
        max_length = max(len(ch) for ch in channels)
        return channels, max_length
    
    else:
        # Load multiple separate files
        channels = []
        for path in audio_input:
            channels.append(load_audio(path, sample_rate))
        
        max_length = max(len(ch) for ch in channels)
        return channels, max_length


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

    def __init__(self, sample_rate: int, features: int, **kwargs):
        super().__init__()
        self.hop_length = kwargs.get("hop_length", sample_rate // 100)
        self.win_length = kwargs.get("win_length", sample_rate // 40)
        self.n_fft = kwargs.get("n_fft", sample_rate // 40)
        self.center = kwargs.get("center", True)
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=features,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                center=self.center,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        if self.center:
            return (
                input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
            )
        else:
            return (
                (input_lengths - self.win_length)
                .div(self.hop_length, rounding_mode="floor")
                .add(1)
                .long()
            )

    def forward(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)
