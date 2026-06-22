#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mac_transcriber import zoom_import


DEFAULT_WORK_ROOT = ".local/mac_transcriber/meetings"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vtt_path = args.vtt.expanduser().resolve()
    if not vtt_path.exists():
        print(f"VTT file does not exist: {vtt_path}", file=sys.stderr)
        return 2

    metadata = load_metadata(args.metadata)
    meeting_id = args.meeting_id or str(metadata.get("uuid") or metadata.get("meeting_id") or vtt_path.stem)
    if args.title:
        metadata["title"] = args.title
    metadata["meeting_id"] = meeting_id

    meeting_dir = args.work_root.expanduser().resolve() / safe_meeting_dir_name(meeting_id)
    if meeting_dir.exists() and args.force:
        shutil.rmtree(meeting_dir)
    elif meeting_dir.exists():
        print(f"Meeting directory already exists: {meeting_dir} (use --force)", file=sys.stderr)
        return 2

    result = zoom_import.import_zoom_vtt(
        meeting_dir=meeting_dir,
        metadata=metadata,
        vtt_text=vtt_path.read_text(encoding="utf-8"),
    )
    print(
        json.dumps(
            {
                "meeting_dir": str(meeting_dir),
                "artifact_dir": str(meeting_dir / "artifacts"),
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import an existing Zoom VTT transcript.")
    parser.add_argument("vtt", type=Path, help="Zoom transcript .vtt file")
    parser.add_argument("--metadata", type=Path, help="Zoom recording metadata JSON")
    parser.add_argument("--meeting-id", help="Stable local meeting id")
    parser.add_argument("--title", help="Override meeting title")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(DEFAULT_WORK_ROOT),
        help="Root directory for local meeting artifacts",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing meeting directory")
    return parser.parse_args(argv)


def load_metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    loaded = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def safe_meeting_dir_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip())
    return safe or "zoom_meeting"


if __name__ == "__main__":
    raise SystemExit(main())
