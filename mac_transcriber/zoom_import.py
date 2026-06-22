from __future__ import annotations

import json
import re
from pathlib import Path
from time import monotonic
from typing import Iterable, Mapping

from mac_transcriber import asr
from mac_transcriber.asr import Segment, TrackSpec, resolve_meeting_title


TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})(?:\s+.*)?$"
)
SECRET_METADATA_KEYS = {
    "download_url",
    "play_url",
    "recording_play_passcode",
    "password",
    "passcode",
}


def parse_zoom_vtt(vtt_text: str) -> list[Segment]:
    segments: list[Segment] = []
    current_start: float | None = None
    current_end: float | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_lines
        if current_start is None or current_end is None:
            current_lines = []
            return
        text = " ".join(line.strip() for line in current_lines if line.strip()).strip()
        if text:
            speaker, clean_text = split_zoom_speaker(text)
            segments.append(
                Segment(
                    speaker=speaker,
                    track="zoom_transcript",
                    start=current_start,
                    end=current_end,
                    text=clean_text,
                )
            )
        current_start = None
        current_end = None
        current_lines = []

    for raw_line in vtt_text.splitlines():
        line = raw_line.strip()
        if not line or line == "WEBVTT" or line.isdigit() or line.startswith("NOTE"):
            if not line:
                flush()
            continue
        match = TIMESTAMP_RE.match(line)
        if match:
            flush()
            current_start = parse_vtt_timestamp(match.group("start"))
            current_end = parse_vtt_timestamp(match.group("end"))
            current_lines = []
            continue
        current_lines.append(line)

    flush()
    return segments


def segments_from_zoom_timeline(timeline: Iterable[Mapping[str, object]]) -> list[Segment]:
    segments: list[Segment] = []
    for item in timeline:
        text = _string_or_empty(item.get("text"))
        if not text:
            continue
        start = parse_vtt_timestamp(_string_or_empty(item.get("ts")))
        end_value = _string_or_empty(item.get("end_ts"))
        end = parse_vtt_timestamp(end_value) if end_value else start
        speaker = _string_or_empty(item.get("display_name")) or "Zoom"
        segments.append(
            Segment(
                speaker=speaker,
                track="zoom_connector",
                start=start,
                end=end,
                text=text,
            )
        )
    return segments


def split_zoom_speaker(text: str) -> tuple[str, str]:
    speaker, separator, rest = text.partition(":")
    if separator and speaker.strip() and len(speaker.strip()) <= 80 and rest.strip():
        return speaker.strip(), rest.strip()
    return "Zoom", text.strip()


def parse_vtt_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def sanitize_zoom_recording_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if key in SECRET_METADATA_KEYS:
            continue
        if key in {"recording_files", "participant_audio_files"} and isinstance(value, list):
            sanitized[key] = [
                sanitize_zoom_recording_metadata(item)
                for item in value
                if isinstance(item, Mapping)
            ]
        elif isinstance(value, Mapping):
            sanitized[key] = sanitize_zoom_recording_metadata(value)
        else:
            sanitized[key] = value
    return sanitized


def import_zoom_vtt(
    *,
    meeting_dir: Path,
    metadata: Mapping[str, object],
    vtt_text: str,
) -> dict[str, object]:
    started = monotonic()
    input_dir = meeting_dir / "input"
    output_dir = meeting_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_metadata = sanitize_zoom_recording_metadata(metadata)
    meeting_id = str(clean_metadata.get("uuid") or clean_metadata.get("meeting_id") or meeting_dir.name)
    clean_metadata["meeting_id"] = meeting_id
    clean_metadata["source_filename"] = "zoom_transcript.vtt"
    clean_metadata["title"] = resolve_meeting_title(dict(clean_metadata), meeting_dir=meeting_dir)

    (input_dir / "metadata.json").write_text(
        json.dumps(clean_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (input_dir / "zoom_transcript.vtt").write_text(vtt_text, encoding="utf-8")

    segments = parse_zoom_vtt(vtt_text)
    tracks = [TrackSpec(path=input_dir / "zoom_transcript.vtt", speaker="Zoom Transcript")]
    elapsed_seconds = round(monotonic() - started, 3)
    asr.write_artifacts(
        output_dir=output_dir,
        meeting_id=meeting_id,
        title=str(clean_metadata["title"]),
        source_filename="zoom_transcript.vtt",
        model_name="zoom_transcript",
        tracks=tracks,
        segments=segments,
        elapsed_seconds=elapsed_seconds,
    )
    return {
        "segments": len(segments),
        "tracks": len(tracks),
        "elapsed_seconds": elapsed_seconds,
    }


def import_zoom_timeline(
    *,
    meeting_dir: Path,
    metadata: Mapping[str, object],
    timeline: list[Mapping[str, object]],
) -> dict[str, object]:
    started = monotonic()
    input_dir = meeting_dir / "input"
    output_dir = meeting_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_metadata = sanitize_zoom_recording_metadata(metadata)
    meeting_id = str(clean_metadata.get("uuid") or clean_metadata.get("meeting_id") or meeting_dir.name)
    clean_metadata["meeting_id"] = meeting_id
    clean_metadata["source_filename"] = "zoom_timeline.json"
    clean_metadata["title"] = resolve_meeting_title(dict(clean_metadata), meeting_dir=meeting_dir)

    (input_dir / "metadata.json").write_text(
        json.dumps(clean_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (input_dir / "zoom_timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    segments = segments_from_zoom_timeline(timeline)
    tracks = [TrackSpec(path=input_dir / "zoom_timeline.json", speaker="Zoom Connector")]
    elapsed_seconds = round(monotonic() - started, 3)
    asr.write_artifacts(
        output_dir=output_dir,
        meeting_id=meeting_id,
        title=str(clean_metadata["title"]),
        source_filename="zoom_timeline.json",
        model_name="zoom_connector",
        tracks=tracks,
        segments=segments,
        elapsed_seconds=elapsed_seconds,
    )
    return {
        "segments": len(segments),
        "tracks": len(tracks),
        "elapsed_seconds": elapsed_seconds,
    }


def _string_or_empty(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()
