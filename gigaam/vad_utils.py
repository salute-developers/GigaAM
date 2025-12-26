import os
from typing import List, Tuple

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pyannote.audio import Model, Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines import VoiceActivityDetection
from torch.torch_version import TorchVersion

from .preprocess import load_audio

_PIPELINE = None


def resolve_local_segmentation_path(model_id: str) -> str:
    """
    Finds the local path to the segmentation model.
    """
    try:
        return snapshot_download(
            repo_id=model_id,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            f"Model {model_id} was not found locally, "
            f"and no HF_TOKEN was provided to download it."
        )

    return snapshot_download(
        repo_id=model_id,
        token=hf_token,
    )


def load_segmentation_model(model_id: str) -> Model:
    """
    Loads the segmentation model from a local snapshot.
    If it doesn’t exist, it first creates (downloads) the snapshot.
    """
    local_path = resolve_local_segmentation_path(model_id=model_id)

    with torch.serialization.safe_globals(
        [
            TorchVersion,
            Problem,
            Specifications,
            Resolution,
        ]
    ):
        return Model.from_pretrained(local_path)


def get_pipeline(
    device: torch.device, model_id: str = "pyannote/segmentation-3.0"
) -> Pipeline:
    """
    Retrieves a PyAnnote voice activity detection pipeline and moves it to the specified device.
    The pipeline is loaded only once and reused across subsequent calls.
    It requires the Hugging Face API token to be set in the HF_TOKEN environment variable.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE.to(device)

    model = load_segmentation_model(model_id=model_id)

    _PIPELINE = VoiceActivityDetection(segmentation=model)
    _PIPELINE.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    return _PIPELINE.to(device)


def segment_audio_file(
    wav_file: str,
    sr: int,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    Segments an audio waveform into smaller chunks based on speech activity.
    The segmentation is performed using a PyAnnote voice activity detection pipeline.
    """

    audio = load_audio(wav_file)
    pipeline = get_pipeline(device)
    sad_segments = pipeline(wav_file)

    segments: List[torch.Tensor] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries: List[Tuple[float, float]] = []

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float):
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    # Concat segments from pipeline into chunks for asr according to max/min duration
    # Segments longer than strict_limit_duration are split manually
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(audio.shape[0] / sr, segment.end)
        if curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    return segments, boundaries
