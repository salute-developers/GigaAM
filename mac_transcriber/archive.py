import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_inventory(meeting_dir: Path) -> list[dict[str, object]]:
    files: list[Path] = []

    input_dir = meeting_dir / "input"
    artifacts_dir = meeting_dir / "artifacts"
    if input_dir.exists():
        files.extend(path for path in input_dir.rglob("*") if path.is_file())
    if artifacts_dir.exists():
        files.extend(path for path in artifacts_dir.rglob("*") if path.is_file())

    status_path = meeting_dir / "status.json"
    if status_path.is_file():
        files.append(status_path)

    inventory = []
    for path in sorted(files, key=lambda item: item.relative_to(meeting_dir).as_posix()):
        relative_path = path.relative_to(meeting_dir).as_posix()
        if relative_path == "artifacts/manifest.json":
            continue

        stat = path.stat()
        inventory.append(
            {
                "relative_path": relative_path,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "kind": _file_kind(path, relative_path),
            }
        )

    return inventory


def write_meeting_manifest(
    meeting_dir: Path, metadata: dict, result: dict | None = None
) -> Path:
    manifest_path = meeting_dir / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "meeting_id": metadata.get("meeting_id"),
        "title": metadata.get("title"),
        "source_filename": metadata.get("source_filename"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "result": result,
        "files": build_file_inventory(meeting_dir),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _file_kind(path: Path, relative_path: str) -> str:
    if relative_path == "input/audio.m4a":
        return "source_audio"
    if relative_path == "input/metadata.json":
        return "metadata"
    if relative_path == "status.json":
        return "status"
    if relative_path.startswith("input/participants/"):
        return "participant_audio"
    if relative_path.startswith("artifacts/"):
        return path.stem
    return "file"
