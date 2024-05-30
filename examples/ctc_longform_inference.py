import argparse
from io import BytesIO
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)
from pyannote.audio import Pipeline
from pydub import AudioSegment


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
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )


def audiosegment_to_numpy(audiosegment: AudioSegment) -> np.ndarray:
    """Convert AudioSegment to numpy array."""
    samples = np.array(audiosegment.get_array_of_samples())
    if audiosegment.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32, order="C") / 32768.0
    return samples


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)
    milliseconds = int((seconds - full_seconds) * 100)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    else:
        return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"


def segment_audio(
    audio_path: str,
    pipeline: Pipeline,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    new_chunk_threshold: float = 0.2,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    # Prepare audio for pyannote vad pipeline
    audio = AudioSegment.from_wav(audio_path)
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    # Process audio with pipeline to obtain segments with speech activity
    sad_segments = pipeline({"uri": "filename", "audio": audio_bytes})

    segments = []
    curr_duration = 0
    curr_start = 0
    curr_end = 0
    boundaries = []

    # Concat segments from pipeline into chunks for asr according to max/min duration
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(len(audio) / 1000, segment.end)
        if (
            curr_duration > min_duration and start - curr_end > new_chunk_threshold
        ) or (curr_duration + (end - curr_end) > max_duration):
            audio_segment = audiosegment_to_numpy(
                audio[curr_start * 1000 : curr_end * 1000]
            )
            segments.append(audio_segment)
            boundaries.append([curr_start, curr_end])
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration != 0:
        audio_segment = audiosegment_to_numpy(
            audio[curr_start * 1000 : curr_end * 1000]
        )
        segments.append(audio_segment)
        boundaries.append([curr_start, curr_end])

    return segments, boundaries


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run long-form inference using GigaAM-CTC checkpoint"
    )
    parser.add_argument("--model_config", help="Path to GigaAM-CTC config file (.yaml)")
    parser.add_argument(
        "--model_weights", help="Path to GigaAM-CTC checkpoint file (.ckpt)"
    )
    parser.add_argument("--audio_path", help="Path to audio signal")
    parser.add_argument(
        "--hf_token", help="HuggingFace token for using pyannote Pipeline"
    )
    parser.add_argument("--device", help="Device: cpu / cuda")
    parser.add_argument("--fp16", help="Run in FP16 mode", default=True)
    parser.add_argument(
        "--batch_size", help="Batch size for acoustic model inference", default=10
    )
    return parser.parse_args()


def main(
    model_config: str,
    model_weights: str,
    device: str,
    audio_path: str,
    hf_token: str,
    fp16: bool,
    batch_size: int = 10,
):
    # Initialize model
    model = EncDecCTCModel.from_config_file(model_config)

    ckpt = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    if device != "cpu" and fp16:
        model = model.half()
        model.preprocessor = model.preprocessor.float()
    model.eval()

    # Initialize pyannote pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection", use_auth_token=hf_token
    )
    pipeline = pipeline.to(torch.device(device))

    # Segment audio
    segments, boundaries = segment_audio(audio_path, pipeline)

    # Transcribe segments
    transcriptions = []
    if device != "cpu" and fp16:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            transcriptions = model.transcribe(segments, batch_size=batch_size)
    else:
        transcriptions = model.transcribe(segments, batch_size=batch_size)

    for transcription, boundary in zip(transcriptions, boundaries):
        print(
            f"[{format_time(boundary[0])} - {format_time(boundary[1])}]: {transcription}\n"
        )


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_config=args.model_config,
        model_weights=args.model_weights,
        device=args.device,
        audio_path=args.audio_path,
        hf_token=args.hf_token,
        fp16=args.fp16,
        batch_size=args.batch_size,
    )
