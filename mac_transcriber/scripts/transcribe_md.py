#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mac_transcriber.asr import transcribe_meeting


DEFAULT_MODEL = "v3_e2e_rnnt"
DEFAULT_CACHE_DIR = "/tmp/gigaam-cache"
DEFAULT_WORK_ROOT = ".local/manual_transcripts"
DEFAULT_OUTPUT_DIR = "transcripts"


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    if not source.exists():
        print(f"Input file does not exist: {source}", file=sys.stderr)
        return 2

    title = args.title or source.stem.strip() or "Transcript"
    meeting_id = args.meeting_id or unique_meeting_id(source)
    work_root = args.work_root.expanduser().resolve()
    meeting_dir = work_root / meeting_id
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(source, args.output_dir.expanduser().resolve())
    )

    prepare_meeting_dir(
        meeting_dir=meeting_dir,
        source=source,
        meeting_id=meeting_id,
        title=title,
        force=args.force,
    )

    os.environ["MAC_TRANSCRIBER_DIARIZATION"] = "1" if args.diarization else "0"
    if args.ai_report:
        os.environ["MAC_TRANSCRIBER_REPORT_MODE"] = "ai"
        os.environ["MAC_TRANSCRIBER_REPORT_MODEL"] = args.report_model
    if args.pdf or args.pdf_output:
        os.environ["MAC_TRANSCRIBER_REPORT_PDF"] = "1"

    started = time.time()
    result = transcribe_meeting(
        meeting_dir=meeting_dir,
        model_name=args.model,
        cache_dir=str(args.cache_dir.expanduser()),
        device=args.device,
        batch_size=args.batch_size,
        status_callback=progress_printer(),
    )

    artifact_path = meeting_dir / "artifacts" / "transcript.md"
    transcript = artifact_path.read_text(encoding="utf-8")
    transcript = normalize_markdown(
        transcript=transcript,
        title=title,
        source=source,
        model=args.model,
        duration=probe_duration(source),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcript, encoding="utf-8")

    artifact_dir = meeting_dir / "artifacts"
    report_output_path = None
    html_output_path = None
    typst_output_path = None
    health_output_path = None
    coverage_output_path = None
    pdf_output_path = None
    should_copy_report = args.report or args.ai_report or args.report_output or args.pdf
    if should_copy_report:
        report_output_path = (
            args.report_output.expanduser().resolve()
            if args.report_output
            else default_report_output_path(source, args.output_dir.expanduser().resolve(), suffix=".md")
        )
        copy_artifact(artifact_dir / "report.md", report_output_path)
        html_output_path = default_report_output_path(
            source,
            args.output_dir.expanduser().resolve(),
            suffix=".html",
        )
        copy_artifact(artifact_dir / "report.html", html_output_path)
        typst_output_path = default_report_output_path(
            source,
            args.output_dir.expanduser().resolve(),
            suffix=".typ",
        )
        copy_artifact(artifact_dir / "report.typ", typst_output_path)
        health_output_path = default_report_output_path(
            source,
            args.output_dir.expanduser().resolve(),
            suffix="_health.json",
        )
        copy_artifact(artifact_dir / "report_health.json", health_output_path)
        coverage_output_path = default_report_output_path(
            source,
            args.output_dir.expanduser().resolve(),
            suffix="_coverage.json",
        )
        copy_artifact(artifact_dir / "coverage.json", coverage_output_path)

    if args.pdf or args.pdf_output:
        pdf_output_path = (
            args.pdf_output.expanduser().resolve()
            if args.pdf_output
            else default_report_output_path(source, args.output_dir.expanduser().resolve(), suffix=".pdf")
        )
        if (artifact_dir / "report.pdf").exists():
            copy_artifact(artifact_dir / "report.pdf", pdf_output_path)
        else:
            pdf_output_path = None

    print(
        json.dumps(
            {
                "markdown": str(output_path),
                "report_markdown": str(report_output_path) if report_output_path else None,
                "report_html": str(html_output_path) if html_output_path else None,
                "report_typst": str(typst_output_path) if typst_output_path else None,
                "report_health": str(health_output_path) if health_output_path else None,
                "coverage_json": str(coverage_output_path) if coverage_output_path else None,
                "report_pdf": str(pdf_output_path) if pdf_output_path else None,
                "artifact_dir": str(artifact_dir),
                "segments": result.get("segments"),
                "tracks": result.get("tracks"),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Markdown transcript for an audio/video file with GigaAM."
    )
    parser.add_argument("input", type=Path, help="Audio or video file to transcribe")
    parser.add_argument("--title", help="Markdown title")
    parser.add_argument("--output", type=Path, help="Markdown output path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Default output directory when --output is not set",
    )
    parser.add_argument("--meeting-id", help="Stable id for the working artifact folder")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(DEFAULT_WORK_ROOT),
        help="Directory for intermediate transcription artifacts",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GigaAM model name")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(DEFAULT_CACHE_DIR),
        help="GigaAM model cache directory",
    )
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=4, help="ASR batch size")
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable pyannote diarization for a single mixed recording",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing working artifact folder for this input",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a structured Markdown report next to the transcript",
    )
    parser.add_argument(
        "--ai-report",
        action="store_true",
        help="Use OpenAI to generate a richer cited report from the transcript",
    )
    parser.add_argument(
        "--report-model",
        default=os.environ.get("MAC_TRANSCRIBER_REPORT_MODEL", "gpt-5.5"),
        help="OpenAI model for --ai-report",
    )
    parser.add_argument("--report-output", type=Path, help="Markdown report output path")
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Print the HTML report to PDF with Playwright when available",
    )
    parser.add_argument("--pdf-output", type=Path, help="PDF report output path")
    return parser.parse_args()


def prepare_meeting_dir(
    *,
    meeting_dir: Path,
    source: Path,
    meeting_id: str,
    title: str,
    force: bool,
) -> None:
    if meeting_dir.exists():
        if not force:
            raise SystemExit(
                f"Working directory already exists: {meeting_dir}\n"
                "Pass --force to rerun and overwrite its intermediate artifacts."
            )
        shutil.rmtree(meeting_dir)

    input_dir = meeting_dir / "input"
    participants_dir = input_dir / "participants"
    participants_dir.mkdir(parents=True, exist_ok=True)

    link_path = input_dir / "audio.m4a"
    try:
        link_path.symlink_to(source)
    except OSError:
        shutil.copy2(source, link_path)

    metadata = {
        "meeting_id": meeting_id,
        "title": title,
        "source_filename": source.name,
        "source_path": str(source),
        "language_code": "ru",
        "processing_mode": "simple",
        "participants": [],
        "zoom_participant_tracks": [],
    }
    (input_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def progress_printer():
    last: dict[str, object] = {"phase": None, "done_bucket": None}

    def print_progress(status: str, **payload: object) -> None:
        phase = payload.get("phase") or status
        done = payload.get("segments_done")
        total = payload.get("segments_total")
        progress = float(payload.get("progress") or 0)
        message = payload.get("message") or phase
        done_bucket = None if done is None else int(done) // 10
        if phase == last["phase"] and done_bucket == last["done_bucket"]:
            return
        last["phase"] = phase
        last["done_bucket"] = done_bucket
        if done is not None and total is not None:
            print(f"[{progress:.0%}] {message}: {done}/{total}", flush=True)
        else:
            print(f"[{progress:.0%}] {message}", flush=True)

    return print_progress


def normalize_markdown(
    *,
    transcript: str,
    title: str,
    source: Path,
    model: str,
    duration: float | None,
) -> str:
    lines = transcript.splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = f"# {title}"
    else:
        lines.insert(0, f"# {title}")
    body = lines[1:]
    first_section_index = next(
        (index for index, line in enumerate(body) if line.startswith("## ")),
        len(body),
    )
    body = [
        line
        for index, line in enumerate(body)
        if not (index < first_section_index and line.startswith("- Model:"))
    ]

    header = [
        "",
        "<!-- Automatic GigaAM transcription. Recognition mistakes are possible. -->",
        "",
        f"- Source: `{source.name}`",
        f"- Model: `{model}`",
    ]
    if duration is not None:
        header.append(f"- Duration: `{format_hms(duration)}`")

    while body and body[0] == "":
        body.pop(0)
    return "\n".join([lines[0], *header, *body]).rstrip() + "\n"


def probe_duration(source: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def default_output_path(source: Path, output_dir: Path) -> Path:
    return output_dir / f"{slugify(source.stem)}_transcript.md"


def default_report_output_path(source: Path, output_dir: Path, *, suffix: str) -> Path:
    return output_dir / f"{slugify(source.stem)}_report{suffix}"


def copy_artifact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def unique_meeting_id(source: Path) -> str:
    return f"{slugify(source.stem)}_{time.strftime('%Y%m%d_%H%M%S')}"


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "transcript"


def format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
