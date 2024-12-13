import os
from io import BytesIO
from typing import List, Tuple, Union

import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment

_PIPELINE = None


def get_pipeline(device: Union[str, torch.device]) -> Pipeline:
    """
    Retrieves a PyAnnote voice activity detection pipeline and move it to the specified device.
    The pipeline is loaded only once and reused across subsequent calls.
    It requires the Hugging Face API token to be set in the HF_TOKEN environment variable.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE.to(device)

    try:
        hf_token = os.environ["HF_TOKEN"]
    except KeyError as exc:
        raise ValueError("HF_TOKEN environment variable is not set") from exc

    _PIPELINE = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection", use_auth_token=hf_token
    )

    return _PIPELINE.to(device)


def audiosegment_to_tensor(audiosegment: AudioSegment) -> torch.Tensor:
    """
    Converts an AudioSegment object to a PyTorch tensor.
    """
    samples = torch.tensor(audiosegment.get_array_of_samples(), dtype=torch.float32)
    if audiosegment.channels == 2:
        samples = samples.view(-1, 2)

    samples = samples / 32768.0  # Normalize to [-1, 1] range
    return samples


def segment_audio(
    wav_tensor: torch.Tensor,
    sample_rate: int,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    new_chunk_threshold: float = 0.2,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    Segments an audio waveform into smaller chunks based on speech activity.
    The segmentation is performed using a PyAnnote voice activity detection pipeline.
    """

    audio = AudioSegment(
        wav_tensor.numpy().tobytes(),
        frame_rate=sample_rate,
        sample_width=wav_tensor.dtype.itemsize,
        channels=1,
    )
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    # Process audio with pipeline to obtain segments with speech activity
    pipeline = get_pipeline(device)
    sad_segments = pipeline({"uri": "filename", "audio": audio_bytes})

    segments: List[torch.Tensor] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries: List[Tuple[float, float]] = []

    # Concat segments from pipeline into chunks for asr according to max/min duration
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(len(audio) / 1000, segment.end)
        if (
            curr_duration > min_duration and start - curr_end > new_chunk_threshold
        ) or (curr_duration + (end - curr_end) > max_duration):

            start_ms = int(curr_start * 1000)
            end_ms = int(curr_end * 1000)
            segments.append(audiosegment_to_tensor(audio[start_ms:end_ms]))
            boundaries.append((curr_start, curr_end))
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration != 0:
        start_ms = int(curr_start * 1000)
        end_ms = int(curr_end * 1000)
        segments.append(audiosegment_to_tensor(audio[start_ms:end_ms]))
        boundaries.append((curr_start, curr_end))

    return segments, boundaries
