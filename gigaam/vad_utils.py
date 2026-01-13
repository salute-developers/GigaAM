import os
from itertools import chain, groupby
from typing import Dict, List, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pyannote.audio import Model, Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines import VoiceActivityDetection
from torch.torch_version import TorchVersion

from .preprocess import load_audio, load_multichannel_audio, SAMPLE_RATE

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


def segment_multichannel_audio(
    audio_input: Union[str, List[str]],
    sr: int = SAMPLE_RATE,
    pause_threshold: float = 2.0,
    strict_limit_duration: float = 30.0,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, Union[int, torch.Tensor, Tuple[float, float]]]]:
    """
    Segments multichannel audio with synchronized diarization.
    
    Simple approach:
    1. Segment each channel with pause_threshold (2 sec by default)
    2. Sort all segments by start_time
    3. Merge segments from same channel (up to strict_limit_duration)
    4. Reduce long pauses to 1 sec to save GPU resources
    
    Returns:
        List of segment dicts with keys: 'channel', 'audio', 'boundaries' (start, end)
        boundaries contain REAL start/end times (not affected by pause reduction)
    """
    # Load multichannel audio
    channels, max_length = load_multichannel_audio(audio_input, sr)
    num_channels = len(channels)
    
    # Move channels to device and pad all channels to same length
    # Do this in one pass to minimize CPU-GPU transfers
    for i in range(num_channels):
        channels[i] = channels[i].to(device)  # Move to GPU first
        if len(channels[i]) < max_length:
            padding = torch.zeros(max_length - len(channels[i]), device=device, dtype=channels[i].dtype)
            channels[i] = torch.cat([channels[i], padding])  # Already on GPU, no transfer needed
    
    pipeline = get_pipeline(device)
    
    # Step 1: Get ALL small VAD segments for ALL channels (don't merge yet!)
    all_segments: List[Dict[str, Union[int, torch.Tensor, Tuple[float, float], float]]] = []
    
    for channel_idx, channel_audio in enumerate(channels):
        # Track last segment info for THIS channel only (in channel scope!)
        prev_end: float = None
        prev_global_start: float = None
        
        # Use pipeline with tensor directly - NO DISK I/O!
        channel_audio_tensor = channel_audio.unsqueeze(0)  # (1, num_samples)
        input_dict = {"waveform": channel_audio_tensor, "sample_rate": sr}
        sad_segments = pipeline(input_dict)
        
        # Get all small VAD segments for this channel (each contains several words)
        # Don't merge yet - we need to sort ALL segments from ALL channels first!
        for segment in sad_segments.get_timeline().support():
            start = max(0, segment.start)
            end = min(max_length / sr, segment.end)
            
            # Calculate global_start: if pause < pause_threshold from previous segment of THIS channel,
            # use previous global_start, otherwise use current start
            if prev_end is not None:
                pause_from_prev = start - prev_end
                if pause_from_prev < pause_threshold:
                    global_start = prev_global_start
                else:
                    global_start = start
            else:
                global_start = start
            
            # Extract audio tensor for this segment (already on GPU)
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            seg_audio = channels[channel_idx][start_idx:end_idx].clone()  # Clone to avoid keeping reference to large tensor
            
            all_segments.append({
                "channel": channel_idx,
                "audio": seg_audio,
                "boundaries": (start, end),  # Real boundaries
                "global_start": global_start,  # Start of the group this segment belongs to
            })
            
            # Update last segment info for THIS channel
            prev_end = end
            prev_global_start = global_start
    
    # Step 2: Sort ALL segments from ALL channels by global_start, then by start_time
    all_segments.sort(key=lambda x: (x["global_start"], x["boundaries"][0]))
    
    # Step 3: Group by channel, then merge segments in each group (up to strict_limit_duration)
    
    def merge_channel_segments(channel_segments: List[Dict]) -> List[Dict]:
        """Merge segments in a channel group, splitting by strict_limit_duration windows"""
        if not channel_segments:
            return []
        
        # Assign window_idx to each segment based on accumulated audio duration
        accumulated_duration = 0.0
        segments_with_window = []
        for seg in channel_segments:
            seg_duration = len(seg["audio"]) / sr
            window_idx = int(accumulated_duration / strict_limit_duration)
            accumulated_duration += seg_duration
            segments_with_window.append((window_idx, seg))
        
        # Sort by window_idx, then group by window_idx, then map merge function
        def merge_window_segments(window_group):
            window_idx, window_segs_iter = window_group
            # Extract seg from (window_idx, seg) tuples
            window_segs = [seg for _, seg in window_segs_iter]
            channel_idx = window_segs[0]["channel"]
            
            # Concatenate all audio in window (no pauses)
            all_audio = [seg["audio"] for seg in window_segs]
            merged_audio = torch.cat(all_audio)
            
            # Get min start and max end
            all_starts = [seg["boundaries"][0] for seg in window_segs]
            all_ends = [seg["boundaries"][1] for seg in window_segs]
            merged_start = min(all_starts)
            merged_end = max(all_ends)
            
            return {
                "channel": channel_idx,
                "audio": merged_audio,
                "boundaries": (merged_start, merged_end),
            }
        
        # Sort -> groupby -> map
        sorted_segments = sorted(segments_with_window, key=lambda x: x[0])
        grouped_by_window = groupby(sorted_segments, key=lambda x: x[0])
        merged_results = list(map(merge_window_segments, grouped_by_window))
        
        return merged_results
    
    # Step 3: Group by channel, then merge segments in each group (up to strict_limit_duration)
    merged_groups = map(
        lambda channel_group: merge_channel_segments(list(channel_group[1])),
        groupby(all_segments, key=lambda x: x["channel"])
    )
    final_segments = list(chain.from_iterable(merged_groups))
    
    return final_segments
