import difflib
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from torch.utils.data import DataLoader

import gigaam
from gigaam.utils import AudioDataset

from mac_transcriber.glossary import apply_glossary


SR = 16000
FRAME_S = 0.10
FRAME = int(SR * FRAME_S)

MIN_SEGMENT_S = 0.28
MERGE_GAP_S = 0.45
PAD_S = 0.28
TARGET_SPLIT_S = 18.0
MIN_SPLIT_S = 10.0
MAX_SPLIT_S = 23.2
DIARIZATION_MODEL = os.environ.get(
    "MAC_TRANSCRIBER_DIARIZATION_MODEL",
    "pyannote/speaker-diarization-community-1",
)
DIARIZATION_PAD_S = float(os.environ.get("MAC_TRANSCRIBER_DIARIZATION_PAD_S", "0.06"))
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
FILENAME_DATE_SUFFIX_RE = re.compile(
    r"(?:\s*[-–—_]\s*)?20\d{2}[-_.]\d{2}[-_.]\d{2}"
    r"(?:\s+\d{2}[-_:]\d{2}(?:\s*UTC)?)?$",
    re.IGNORECASE,
)

_MODEL = None
_MODEL_LOCK = Lock()
_DIARIZATION_PIPELINE = None
_DIARIZATION_MODEL_NAME = None
_DIARIZATION_LOCK = Lock()


@dataclass
class TrackSpec:
    path: Path
    speaker: str


@dataclass
class Segment:
    speaker: str
    track: str
    start: float
    end: float
    wav: torch.Tensor | None = None
    text: str = ""


@dataclass
class TranscriptBlock:
    speaker: str
    start: float
    end: float
    text: str
    segment_ids: list[str]


@dataclass(frozen=True)
class DiarizedTurn:
    speaker: str
    start: float
    end: float


class DiarizationUnavailable(RuntimeError):
    pass


def transcribe_meeting(
    *,
    meeting_dir: Path,
    model_name: str,
    cache_dir: str,
    device: str,
    batch_size: int,
    status_callback: Callable[..., None] | None = None,
) -> dict[str, object]:
    started = time.time()
    input_dir = meeting_dir / "input"
    output_dir = meeting_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(
        status_callback,
        "processing",
        phase="preparing",
        progress=0.05,
        message="Reading meeting metadata",
    )
    metadata = _load_json(input_dir / "metadata.json")
    tracks = _track_specs(input_dir=input_dir, metadata=metadata)
    _emit_progress(
        status_callback,
        "processing",
        phase="segmenting",
        progress=0.12,
        message="Detecting speech segments",
        tracks_total=len(tracks),
    )

    def on_segmenting_progress(done: int, total: int, segments_count: int) -> None:
        _emit_progress(
            status_callback,
            "processing",
            phase="segmenting",
            progress=round(0.12 + (0.08 * done / max(total, 1)), 3),
            message="Detecting speech segments",
            tracks_done=done,
            tracks_total=total,
            segments_total=segments_count,
        )

    segments, tracks = build_input_segments(
        input_dir=input_dir,
        metadata=metadata,
        device=device,
        progress_callback=on_segmenting_progress,
    )
    _emit_progress(
        status_callback,
        "processing",
        phase="transcribing",
        progress=0.20 if segments else 0.82,
        message="Transcribing speech segments" if segments else "No speech segments found",
        segments_done=0,
        segments_total=len(segments),
        tracks_total=len(tracks),
    )

    def on_segment_progress(done: int, total: int) -> None:
        _emit_progress(
            status_callback,
            "processing",
            phase="transcribing",
            progress=round(0.20 + (0.68 * done / max(total, 1)), 3),
            message="Transcribing speech segments",
            segments_done=done,
            segments_total=total,
            tracks_total=len(tracks),
        )

    segments = transcribe_segments(
        segments,
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        progress_callback=on_segment_progress,
    )
    _emit_progress(
        status_callback,
        "processing",
        phase="writing_artifacts",
        progress=0.92,
        message="Writing transcript artifacts",
        segments_done=len(segments),
        segments_total=len(segments),
        tracks_total=len(tracks),
    )
    write_artifacts(
        output_dir=output_dir,
        meeting_id=metadata.get("meeting_id") or meeting_dir.name,
        title=resolve_meeting_title(metadata, meeting_dir=meeting_dir),
        source_filename=metadata.get("source_filename") or _source_filename(tracks),
        model_name=model_name,
        tracks=tracks,
        segments=segments,
        elapsed_seconds=time.time() - started,
    )
    result = {
        "segments": len(segments),
        "tracks": len(tracks),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _emit_progress(
        status_callback,
        "completed",
        phase="completed",
        progress=1.0,
        message="Transcription complete",
        result=result,
    )
    return result


def resolve_meeting_title(metadata: dict, *, meeting_dir: Path | None = None) -> str:
    for key in ("title", "meeting_title", "topic", "zoom_topic"):
        title = _clean_meeting_title(metadata.get(key))
        if title and not UUID_RE.fullmatch(title):
            return title

    filename_title = title_from_source_filename(metadata.get("source_filename"))
    if filename_title:
        return filename_title

    fallback = _clean_meeting_title(metadata.get("meeting_id"))
    if not fallback and meeting_dir is not None:
        fallback = _clean_meeting_title(meeting_dir.name)
    if fallback and not UUID_RE.fullmatch(fallback):
        return fallback
    return "Протокол встречи"


def title_from_source_filename(value: object) -> str:
    if not isinstance(value, str):
        return ""
    title = Path(value).stem.strip()
    title = FILENAME_DATE_SUFFIX_RE.sub("", title).strip()
    return _clean_meeting_title(title)


def _clean_meeting_title(value: object) -> str:
    if not isinstance(value, str):
        return ""
    title = value.strip()
    title = re.sub(r"\s*[-–—]\s*[-–—]\s*", " - ", title)
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"^\s*[-–—_]+\s*", "", title)
    title = re.sub(r"\s*[-–—_]+\s*$", "", title)
    return title.strip()


def _emit_progress(callback: Callable[..., None] | None, status: str, **payload: object) -> None:
    if callback is not None:
        callback(status, **payload)


def build_input_segments(
    *,
    input_dir: Path,
    metadata: dict,
    device: str,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[list[Segment], list[TrackSpec]]:
    tracks = _track_specs(input_dir=input_dir, metadata=metadata)
    if _is_single_mixed_file(input_dir=input_dir, tracks=tracks):
        if progress_callback is not None:
            progress_callback(0, 1, 0)
        try:
            segments = build_diarized_segments(
                tracks[0].path,
                device=device,
                metadata=metadata,
            )
        except DiarizationUnavailable:
            segments = []
        if segments:
            if progress_callback is not None:
                progress_callback(1, 1, len(segments))
            return segments, speaker_tracks_from_segments(tracks[0].path, segments)

    return build_segments(tracks, progress_callback=progress_callback), tracks


def build_diarized_segments(
    audio_path: Path,
    *,
    device: str,
    metadata: dict,
) -> list[Segment]:
    turns = diarize_audio(audio_path, device=device, metadata=metadata)
    if not turns:
        return []

    wav = gigaam.load_audio(str(audio_path)).cpu()
    _mask, rms, _threshold = activity_mask(wav)
    speaker_names = stable_speaker_names(turns)
    segments: list[Segment] = []
    min_len = int(MIN_SEGMENT_S * SR)
    pad = int(DIARIZATION_PAD_S * SR)

    for turn in turns:
        start_sample = max(0, int(turn.start * SR) - pad)
        end_sample = min(wav.numel(), math.ceil(turn.end * SR) + pad)
        if end_sample - start_sample < min_len:
            continue
        for part_start, part_end in split_interval(start_sample, end_sample, rms):
            segments.append(
                Segment(
                    speaker=speaker_names[turn.speaker],
                    track=audio_path.name,
                    start=part_start / SR,
                    end=part_end / SR,
                    wav=wav[part_start:part_end].clone(),
                )
            )

    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker))
    return segments


def diarize_audio(audio_path: Path, *, device: str, metadata: dict) -> list[DiarizedTurn]:
    if not diarization_enabled():
        raise DiarizationUnavailable("diarization disabled")

    pipeline = load_diarization_pipeline(device=device)
    prepared_path = prepare_diarization_audio(audio_path)
    try:
        output = pipeline(str(prepared_path), **diarization_options(metadata))
    except Exception as exc:  # noqa: BLE001
        raise DiarizationUnavailable(f"diarization failed: {exc}") from exc

    try:
        turns = diarization_output_to_turns(output)
    except Exception as exc:  # noqa: BLE001
        raise DiarizationUnavailable(f"cannot read diarization output: {exc}") from exc
    return merge_diarized_turns(turns)


def prepare_diarization_audio(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    prepared_path = audio_path.with_suffix(".diarization.wav")
    try:
        source_mtime = audio_path.stat().st_mtime
        if prepared_path.exists() and prepared_path.stat().st_mtime >= source_mtime:
            return prepared_path
    except OSError as exc:
        raise DiarizationUnavailable(f"cannot inspect audio file: {exc}") from exc

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-vn",
        str(prepared_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise DiarizationUnavailable("ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise DiarizationUnavailable(f"ffmpeg conversion failed: {message}") from exc

    return prepared_path


def load_diarization_pipeline(*, device: str):
    global _DIARIZATION_PIPELINE, _DIARIZATION_MODEL_NAME
    pipeline_device = os.environ.get("MAC_TRANSCRIBER_DIARIZATION_DEVICE") or device
    with _DIARIZATION_LOCK:
        if _DIARIZATION_PIPELINE is None or _DIARIZATION_MODEL_NAME != DIARIZATION_MODEL:
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise DiarizationUnavailable("pyannote.audio is not installed") from exc

            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            kwargs = {"token": token} if token else {}
            try:
                _DIARIZATION_PIPELINE = Pipeline.from_pretrained(DIARIZATION_MODEL, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise DiarizationUnavailable(f"cannot load diarization model: {exc}") from exc
            if _DIARIZATION_PIPELINE is None:
                raise DiarizationUnavailable("cannot load diarization model")
            _DIARIZATION_MODEL_NAME = DIARIZATION_MODEL

        try:
            return _DIARIZATION_PIPELINE.to(torch.device(pipeline_device))
        except Exception as exc:  # noqa: BLE001
            raise DiarizationUnavailable(
                f"cannot move diarization model to {pipeline_device}: {exc}"
            ) from exc


def diarization_enabled() -> bool:
    value = os.environ.get("MAC_TRANSCRIBER_DIARIZATION", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def diarization_options(metadata: dict) -> dict[str, int]:
    options: dict[str, int] = {}
    participants = metadata.get("participants")
    if isinstance(participants, list) and len(participants) >= 2:
        options["num_speakers"] = len(participants)

    num_speakers = positive_int_env("MAC_TRANSCRIBER_DIARIZATION_NUM_SPEAKERS")
    min_speakers = positive_int_env("MAC_TRANSCRIBER_DIARIZATION_MIN_SPEAKERS")
    max_speakers = positive_int_env("MAC_TRANSCRIBER_DIARIZATION_MAX_SPEAKERS")
    if num_speakers is not None:
        options = {"num_speakers": num_speakers}
    else:
        if min_speakers is not None:
            options["min_speakers"] = min_speakers
        if max_speakers is not None:
            options["max_speakers"] = max_speakers
    return options


def positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def diarization_output_to_turns(output: object) -> list[DiarizedTurn]:
    annotation = (
        getattr(output, "exclusive_speaker_diarization", None)
        or getattr(output, "speaker_diarization", None)
        or output
    )
    turns: list[DiarizedTurn] = []

    if hasattr(annotation, "itertracks"):
        for turn, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                DiarizedTurn(
                    speaker=str(speaker),
                    start=float(turn.start),
                    end=float(turn.end),
                )
            )
        return turns

    for item in annotation:
        if len(item) == 2:
            turn, speaker = item
        elif len(item) == 3:
            turn, _track, speaker = item
        else:
            continue
        turns.append(
            DiarizedTurn(
                speaker=str(speaker),
                start=float(turn.start),
                end=float(turn.end),
            )
        )
    return turns


def merge_diarized_turns(turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
    merged: list[DiarizedTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        if turn.end <= turn.start or turn.end - turn.start < MIN_SEGMENT_S:
            continue
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and turn.start - merged[-1].end <= MERGE_GAP_S
        ):
            merged[-1] = DiarizedTurn(
                speaker=merged[-1].speaker,
                start=merged[-1].start,
                end=max(merged[-1].end, turn.end),
            )
        else:
            merged.append(turn)
    return merged


def stable_speaker_names(turns: list[DiarizedTurn]) -> dict[str, str]:
    names: dict[str, str] = {}
    for turn in turns:
        if turn.speaker not in names:
            names[turn.speaker] = f"Speaker {len(names) + 1}"
    return names


def speaker_tracks_from_segments(audio_path: Path, segments: list[Segment]) -> list[TrackSpec]:
    speakers: list[str] = []
    for segment in segments:
        if segment.speaker not in speakers:
            speakers.append(segment.speaker)
    return [TrackSpec(path=audio_path, speaker=speaker) for speaker in speakers]


def _is_single_mixed_file(*, input_dir: Path, tracks: list[TrackSpec]) -> bool:
    return len(tracks) == 1 and tracks[0].path == input_dir / "audio.m4a"


def build_segments(
    tracks: list[TrackSpec],
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> list[Segment]:
    segments: list[Segment] = []
    total = len(tracks)
    for index, track in enumerate(tracks, start=1):
        wav = gigaam.load_audio(str(track.path)).cpu()
        mask, rms, _threshold = activity_mask(wav)
        for start, end in mask_to_intervals(mask, wav.numel()):
            for part_start, part_end in split_interval(start, end, rms):
                segments.append(
                    Segment(
                        speaker=track.speaker,
                        track=track.path.name,
                        start=part_start / SR,
                        end=part_end / SR,
                        wav=wav[part_start:part_end].clone(),
                    )
                )
        if progress_callback is not None:
            progress_callback(index, total, len(segments))
    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker))
    return segments


def activity_mask(wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    wav = wav.abs().cpu()
    n_frames = math.ceil(wav.numel() / FRAME)
    padded = torch.nn.functional.pad(wav, (0, n_frames * FRAME - wav.numel()))
    rms = (padded.view(n_frames, FRAME).pow(2).mean(dim=1) + 1e-12).sqrt()
    nonzero = rms[rms > 1e-5]
    if len(nonzero) == 0:
        threshold = 1e-4
    else:
        q20 = float(torch.quantile(nonzero, 0.20))
        q95 = float(torch.quantile(nonzero, 0.95))
        threshold = max(q20 * 3.0, q95 * 0.08, 0.0015)
    return rms > threshold, rms, threshold


def mask_to_intervals(mask: torch.Tensor, total_samples: int) -> list[tuple[int, int]]:
    values = mask.tolist()
    raw: list[tuple[int, int]] = []
    start = None
    for i, active in enumerate(values + [False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            raw.append((start, i))
            start = None

    merge_gap = int(MERGE_GAP_S / FRAME_S)
    merged: list[tuple[int, int]] = []
    for start_f, end_f in raw:
        if not merged or start_f - merged[-1][1] > merge_gap:
            merged.append((start_f, end_f))
        else:
            merged[-1] = (merged[-1][0], end_f)

    intervals: list[tuple[int, int]] = []
    pad = int(PAD_S * SR)
    min_len = int(MIN_SEGMENT_S * SR)
    for start_f, end_f in merged:
        start_sample = max(0, start_f * FRAME - pad)
        end_sample = min(total_samples, end_f * FRAME + pad)
        if end_sample - start_sample >= min_len:
            intervals.append((start_sample, end_sample))
    return intervals


def split_interval(start: int, end: int, rms: torch.Tensor) -> list[tuple[int, int]]:
    max_len = int(MAX_SPLIT_S * SR)
    if end - start <= max_len:
        return [(start, end)]

    pieces: list[tuple[int, int]] = []
    current = start
    while end - current > max_len:
        lo = current + int(MIN_SPLIT_S * SR)
        hi = min(end, current + max_len)
        lo_f = max(0, lo // FRAME)
        hi_f = min(len(rms), max(lo_f + 1, hi // FRAME))
        if hi_f <= lo_f:
            cut = min(end, current + int(TARGET_SPLIT_S * SR))
        else:
            cut_f = lo_f + int(torch.argmin(rms[lo_f:hi_f]).item())
            cut = max(current + int(MIN_SPLIT_S * SR), min(end, cut_f * FRAME))
        pieces.append((current, cut))
        current = cut
    if end - current >= int(MIN_SEGMENT_S * SR):
        pieces.append((current, end))
    return pieces


def transcribe_segments(
    segments: list[Segment],
    *,
    model_name: str,
    cache_dir: str,
    device: str,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Segment]:
    if not segments:
        return []
    model = load_model(model_name=model_name, cache_dir=cache_dir, device=device)
    dataset = AudioDataset([segment.wav for segment in segments if segment.wav is not None])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=0,
    )
    index = 0
    total = len(segments)
    with torch.inference_mode():
        for wav_pad, wav_lens in loader:
            encoded, encoded_len = model.forward(wav_pad, wav_lens)
            decoded = model._decode(encoded, encoded_len, wav_lens, word_timestamps=False)
            for text, _words in decoded:
                segments[index].text = apply_glossary(normalize_text(text))
                segments[index].wav = None
                index += 1
            if progress_callback is not None:
                progress_callback(index, total)
    return dedupe_segments(segments)


def load_model(*, model_name: str, cache_dir: str, device: str):
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = gigaam.load_model(model_name, download_root=cache_dir, device=device)
            _MODEL.eval()
        return _MODEL


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def dedupe_segments(segments: list[Segment]) -> list[Segment]:
    kept: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        if not segment.text:
            continue
        duplicate = False
        for previous in kept[-8:]:
            if previous.speaker != segment.speaker:
                continue
            overlap = min(previous.end, segment.end) - max(previous.start, segment.start)
            if overlap <= 0:
                continue
            shorter = max(0.001, min(previous.end - previous.start, segment.end - segment.start))
            ratio = difflib.SequenceMatcher(None, previous.text, segment.text).ratio()
            if overlap / shorter > 0.55 and ratio > 0.62:
                duplicate = True
                break
        if not duplicate:
            kept.append(segment)
    return kept


def write_artifacts(
    *,
    output_dir: Path,
    meeting_id: str,
    title: str,
    source_filename: str,
    model_name: str,
    tracks: list[TrackSpec],
    segments: list[Segment],
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_segments = sorted(segments, key=lambda item: (item.start, item.end))
    transcript_md = render_transcript_markdown(
        meeting_id=meeting_id,
        title=title,
        model_name=model_name,
        tracks=tracks,
        segments=segments,
    )
    (output_dir / "transcript.md").write_text(transcript_md, encoding="utf-8")
    from mac_transcriber.reporting import write_report_artifacts
    from mac_transcriber.memory_db import load_report_context_pack, sync_meeting_memory

    context_pack, context_pack_error = load_report_context_pack(
        meeting_id=meeting_id,
        title=title,
        source_filename=source_filename,
    )
    if context_pack is not None:
        (output_dir / "context_pack.json").write_text(
            json.dumps(context_pack, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    report_artifacts = write_report_artifacts(
        output_dir=output_dir,
        meeting_id=meeting_id,
        title=title,
        source_filename=source_filename,
        model_name=model_name,
        segments=segments,
        use_ai=_report_ai_enabled(),
        make_pdf=_env_flag("MAC_TRANSCRIBER_REPORT_PDF", default=_report_ai_enabled()),
        report_model=os.environ.get("MAC_TRANSCRIBER_REPORT_MODEL"),
        context_pack=context_pack,
    )
    summary_payload = {
        "meeting_id": meeting_id,
        "title": title,
        "source_filename": source_filename,
        "model_name": model_name,
        "summary": f"Transcription completed: {len(segments)} segments, {len(tracks)} tracks.",
        "segments": len(segments),
        "tracks": len(tracks),
        "segment_count": len(segments),
        "track_count": len(tracks),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "report_generator": report_artifacts.generated_by,
        "report_status": report_artifacts.status,
        "report_alerts": report_artifacts.alerts,
        "report_html": str(report_artifacts.html_path),
        "report_health": str(report_artifacts.health_path),
        "slack": {
            "text": report_artifacts.slack_text,
            "files": report_artifacts.slack_files,
        },
        "coverage_json": str(report_artifacts.coverage_path),
        "report_pdf": str(report_artifacts.pdf_path) if report_artifacts.pdf_path else None,
        "report_pdf_error": report_artifacts.pdf_error,
    }
    if context_pack is not None:
        summary_payload["context_pack_json"] = str(output_dir / "context_pack.json")
    if context_pack_error:
        summary_payload["context_pack_error"] = context_pack_error
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary_payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "transcript.json").write_text(
        json.dumps(
            [
                {
                    "segment_id": f"S{index:04d}",
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "speaker": segment.speaker,
                    "track": segment.track,
                    "text": segment.text,
                }
                for index, segment in enumerate(ordered_segments, start=1)
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "segments.tsv").write_text(
        render_segments_tsv(ordered_segments),
        encoding="utf-8",
    )
    (output_dir / "speaker_track_map.tsv").write_text(render_track_map(tracks), encoding="utf-8")

    manifest_metadata = {
        **_load_optional_json_dict(output_dir.parent / "input" / "metadata.json"),
        "meeting_id": meeting_id,
        "title": title,
        "source_filename": source_filename,
        "model_name": model_name,
        "segment_count": len(segments),
        "track_count": len(tracks),
    }
    from mac_transcriber.archive import write_meeting_manifest

    meeting_dir = output_dir.parent
    manifest_path = write_meeting_manifest(meeting_dir, manifest_metadata, summary_payload)
    memory_sync_error = sync_meeting_memory(meeting_dir, manifest_path)
    if memory_sync_error:
        summary_payload["memory_sync_error"] = memory_sync_error
        (output_dir / "summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_meeting_manifest(meeting_dir, manifest_metadata, summary_payload)


def _source_filename(tracks: list[TrackSpec]) -> str:
    if not tracks:
        return ""
    return tracks[0].path.name


def _report_ai_enabled() -> bool:
    return os.environ.get("MAC_TRANSCRIBER_REPORT_MODE", "ai").strip().lower() == "ai"


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def render_transcript_markdown(
    *,
    meeting_id: str,
    title: str | None = None,
    model_name: str,
    tracks: list[TrackSpec],
    segments: list[Segment],
) -> str:
    heading = title or f"Zoom transcript: {meeting_id}"
    lines = [
        f"# {heading}",
        "",
        f"- ID: `{meeting_id}`",
        f"- Model: {model_name}",
        f"- Segments: {len(segments)}",
        "",
        "## Speaker Track Map",
        "",
    ]
    for track in tracks:
        lines.append(f"- {track.speaker}: {track.path.name}")
    lines.extend(["", "## Transcript", ""])
    for block in merge_transcript_blocks(segments):
        lines.append(
            f"[{format_segment_id_range(block.segment_ids)}] "
            f"[{fmt_time(block.start)} - {fmt_time(block.end)}] "
            f"**{block.speaker}:** {block.text}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_report_markdown(meeting_id: str, segments: list[Segment]) -> str:
    speakers = sorted({segment.speaker for segment in segments})
    return (
        f"# Meeting report: {meeting_id}\n\n"
        f"- Transcript segments: {len(segments)}\n"
        f"- Speakers: {', '.join(speakers) if speakers else 'none'}\n\n"
        "Automatic summary generation is not enabled yet; see transcript.md.\n"
    )


def render_segments_tsv(segments: list[Segment]) -> str:
    lines = ["start\tend\tspeaker\ttrack\ttext"]
    for segment in segments:
        lines.append(
            "\t".join(
                [
                    f"{segment.start:.3f}",
                    f"{segment.end:.3f}",
                    segment.speaker,
                    segment.track,
                    segment.text.replace("\t", " "),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def render_track_map(tracks: list[TrackSpec]) -> str:
    lines = ["track\tspeaker"]
    for track in tracks:
        lines.append(f"{track.path.name}\t{track.speaker}")
    return "\n".join(lines) + "\n"


def merge_blocks(segments: list[Segment]) -> list[Segment]:
    blocks: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        if (
            blocks
            and blocks[-1].speaker == segment.speaker
            and segment.start - blocks[-1].end <= 1.25
        ):
            blocks[-1].end = max(blocks[-1].end, segment.end)
            blocks[-1].text = f"{blocks[-1].text} {segment.text}".strip()
        else:
            blocks.append(
                Segment(
                    speaker=segment.speaker,
                    track=segment.track,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
            )
    return blocks


def merge_transcript_blocks(segments: list[Segment]) -> list[TranscriptBlock]:
    blocks: list[TranscriptBlock] = []
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    for index, segment in enumerate(ordered, start=1):
        segment_id = f"S{index:04d}"
        if (
            blocks
            and blocks[-1].speaker == segment.speaker
            and segment.start - blocks[-1].end <= 1.25
        ):
            blocks[-1].end = max(blocks[-1].end, segment.end)
            blocks[-1].text = f"{blocks[-1].text} {segment.text}".strip()
            blocks[-1].segment_ids.append(segment_id)
        else:
            blocks.append(
                TranscriptBlock(
                    speaker=segment.speaker,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    segment_ids=[segment_id],
                )
            )
    return blocks


def format_segment_id_range(segment_ids: list[str]) -> str:
    if not segment_ids:
        return "S0000"
    if len(segment_ids) == 1:
        return segment_ids[0]
    return f"{segment_ids[0]}-{segment_ids[-1]}"


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _track_specs(*, input_dir: Path, metadata: dict) -> list[TrackSpec]:
    participant_files = sorted((input_dir / "participants").glob("*.m4a"))
    if not participant_files:
        if _requires_zoom_participant_tracks(metadata):
            raise ValueError("Zoom participant tracks are required for this meeting")
        return [TrackSpec(path=input_dir / "audio.m4a", speaker="Speaker")]

    tracks_metadata = metadata.get("zoom_participant_tracks") or []
    if _requires_zoom_participant_tracks(metadata) and len(tracks_metadata) != len(participant_files):
        raise ValueError("Zoom participant track metadata does not match downloaded files")
    specs: list[TrackSpec] = []
    for index, path in enumerate(participant_files):
        speaker = None
        if index < len(tracks_metadata):
            speaker = tracks_metadata[index].get("speaker_name")
        specs.append(TrackSpec(path=path, speaker=speaker or f"Zoom participant {index + 1}"))
    return specs


def _requires_zoom_participant_tracks(metadata: dict) -> bool:
    if metadata.get("processing_mode") == "zoom_participant_tracks":
        return True
    if metadata.get("source") == "zoom_cloud_backfill":
        return True
    tracks_metadata = metadata.get("zoom_participant_tracks")
    return isinstance(tracks_metadata, list) and bool(tracks_metadata)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
