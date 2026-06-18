import json
import os
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from mac_transcriber.archive import write_meeting_manifest
from mac_transcriber.asr import resolve_meeting_title, transcribe_meeting
from mac_transcriber.memory_db import sync_meeting_memory
from mac_transcriber.reporting import ReportQuotaError


def _load_local_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_local_env_file(Path.cwd() / ".env.local")

ROOT = Path(os.environ.get("MAC_TRANSCRIBER_ROOT", ".local/mac_transcriber")).resolve()
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
API_KEY = os.environ.get("MAC_TRANSCRIBER_API_KEY", "")
MODEL_NAME = os.environ.get("MAC_TRANSCRIBER_MODEL", "v3_e2e_rnnt")
CACHE_DIR = os.environ.get("MAC_TRANSCRIBER_CACHE", "/tmp/gigaam-cache")
DEVICE = os.environ.get("MAC_TRANSCRIBER_DEVICE", "cpu")
BATCH_SIZE = int(os.environ.get("MAC_TRANSCRIBER_BATCH_SIZE", "4"))
_PROCESS_LOCK = Lock()

ARTIFACTS = {
    "manifest_json": ("manifest.json", "application/json"),
    "context_pack_json": ("context_pack.json", "application/json"),
    "transcript_md": ("transcript.md", "text/markdown; charset=utf-8"),
    "transcript_json": ("transcript.json", "application/json"),
    "summary_json": ("summary.json", "application/json"),
    "report_md": ("report.md", "text/markdown; charset=utf-8"),
    "report_json": ("report.json", "application/json"),
    "report_html": ("report.html", "text/html; charset=utf-8"),
    "report_health": ("report_health.json", "application/json"),
    "coverage_json": ("coverage.json", "application/json"),
    "report_typ": ("report.typ", "text/plain; charset=utf-8"),
    "report_pdf": ("report.pdf", "application/pdf"),
    "segments_tsv": ("segments.tsv", "text/tab-separated-values; charset=utf-8"),
    "speaker_track_map": (
        "speaker_track_map.tsv",
        "text/tab-separated-values; charset=utf-8",
    ),
}

app = FastAPI(title="GigaAM Mac Transcriber")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui_index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/ui/config")
def ui_config() -> dict[str, object]:
    return {
        # Не отдаём сам секрет наружу: UI запрашивает ключ у оператора, когда auth_required.
        "auth_required": bool(API_KEY),
        "model": MODEL_NAME,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "root": str(ROOT),
    }


@app.post("/meetings", dependencies=[Depends(require_auth)])
async def create_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    zoom_participant_files: list[UploadFile] | None = File(default=None),
    language_code: str = Form(default="ru"),
    processing_mode: str = Form(default="simple"),
    title: str | None = Form(default=None),
    meeting_title: str | None = Form(default=None),
    topic: str | None = Form(default=None),
    participants: str | None = Form(default=None),
    zoom_participant_tracks_json: str | None = Form(default=None),
) -> dict[str, str]:
    meeting_id = str(uuid.uuid4())
    meeting_dir = ROOT / "meetings" / meeting_id
    input_dir = meeting_dir / "input"
    participants_dir = input_dir / "participants"
    participants_dir.mkdir(parents=True, exist_ok=False)

    await _save_upload(file, input_dir / "audio.m4a")
    for index, upload in enumerate(zoom_participant_files or [], start=1):
        await _save_upload(upload, participants_dir / f"{index:02d}.m4a")

    metadata = {
        "meeting_id": meeting_id,
        "source_filename": file.filename,
        "language_code": language_code,
        "processing_mode": processing_mode,
        "participants": _split_csv(participants),
        "zoom_participant_tracks": _parse_tracks(zoom_participant_tracks_json),
    }
    metadata["title"] = resolve_meeting_title(
        {
            **metadata,
            "title": title,
            "meeting_title": meeting_title,
            "topic": topic,
        },
        meeting_dir=meeting_dir,
    )
    _write_status(
        meeting_dir,
        "uploaded",
        phase="uploaded",
        progress=0.01,
        message="Upload complete",
        metadata=metadata,
    )
    (input_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    background_tasks.add_task(_process_meeting, meeting_id)
    return {"id": meeting_id, "status": "uploaded"}


@app.get("/meetings", dependencies=[Depends(require_auth)])
def list_meetings() -> dict[str, list[dict]]:
    meetings_dir = ROOT / "meetings"
    if not meetings_dir.exists():
        return {"meetings": []}

    items = []
    for meeting_dir in sorted(
        (path for path in meetings_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        status = _read_status(meeting_dir)
        if not status:
            continue
        _attach_report_health(status, meeting_dir)
        status["artifacts"] = _available_artifacts(meeting_dir)
        items.append(status)
    return {"meetings": items}


@app.get("/meetings/{meeting_id}/status", dependencies=[Depends(require_auth)])
def meeting_status(meeting_id: str) -> dict:
    meeting_dir = _meeting_dir(meeting_id)
    status_path = meeting_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Meeting not found")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    _attach_report_health(status, meeting_dir)
    status["artifacts"] = _available_artifacts(meeting_dir)
    return status


@app.get(
    "/meetings/{meeting_id}/artifacts/{kind}", dependencies=[Depends(require_auth)]
)
def meeting_artifact(meeting_id: str, kind: str):
    if kind not in ARTIFACTS:
        raise HTTPException(status_code=404, detail="Artifact not found")
    filename, media_type = ARTIFACTS[kind]
    path = _meeting_dir(meeting_id) / "artifacts" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type=media_type, filename=filename)


def _process_meeting(meeting_id: str) -> None:
    meeting_dir = _meeting_dir(meeting_id)
    with _PROCESS_LOCK:
        try:
            _write_status(
                meeting_dir,
                "processing",
                phase="queued",
                progress=0.03,
                message="Waiting for the local transcriber",
            )
            result = transcribe_meeting(
                meeting_dir=meeting_dir,
                model_name=MODEL_NAME,
                cache_dir=CACHE_DIR,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                status_callback=lambda status, **payload: _write_status(
                    meeting_dir, status, **payload
                ),
            )
            _write_status(
                meeting_dir,
                "completed",
                phase="completed",
                progress=1.0,
                message="Transcription complete",
                result=result,
            )
        except ReportQuotaError as exc:
            # Нет денег у AI-провайдера: НЕ выдаём сырой local-отчёт и НЕ помечаем
            # failed. Транскрипт уже сохранён; ставим на паузу до пополнения, потом
            # дообработаем через reprocess_blocked.py.
            _write_status(
                meeting_dir,
                "blocked_on_quota",
                phase="blocked_on_quota",
                progress=0.9,
                message=f"Paused: AI quota exhausted. Top up, then reprocess. ({exc})"[
                    :300
                ],
            )
        except Exception as exc:  # noqa: BLE001
            _write_status(
                meeting_dir,
                "failed",
                phase="failed",
                progress=1.0,
                message="Transcription failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )


def _write_status(meeting_dir: Path, status: str, **extra: object) -> None:
    meeting_dir.mkdir(parents=True, exist_ok=True)
    status_path = meeting_dir / "status.json"
    previous = _read_status(meeting_dir)
    now = datetime.now(UTC).isoformat()
    payload = {
        **previous,
        "id": meeting_dir.name,
        "status": status,
        "created_at": previous.get("created_at", now),
        "updated_at": now,
        **extra,
    }
    if status != "failed":
        payload.pop("error", None)
        payload.pop("traceback", None)
    if "memory_sync_error" not in extra:
        payload.pop("memory_sync_error", None)
    payload = _sanitize_status_payload(payload)
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if status in {"completed", "failed"}:
        _refresh_existing_manifest_and_sync_memory(meeting_dir, status_path, payload)


def _read_status(meeting_dir: Path) -> dict:
    status_path = meeting_dir / "status.json"
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _refresh_existing_manifest_and_sync_memory(
    meeting_dir: Path,
    status_path: Path,
    status_payload: dict,
) -> None:
    manifest_path = _refresh_existing_manifest(meeting_dir)
    if manifest_path is None:
        return
    try:
        memory_sync_error = sync_meeting_memory(meeting_dir, manifest_path)
    except Exception as exc:  # noqa: BLE001
        memory_sync_error = f"Memory sync failed: {exc}"
    if not memory_sync_error:
        return

    status_payload = _sanitize_status_payload(
        {**status_payload, "memory_sync_error": memory_sync_error}
    )
    status_path.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _refresh_existing_manifest(meeting_dir)


def _refresh_existing_manifest(meeting_dir: Path) -> Path | None:
    manifest_path = meeting_dir / "artifacts" / "manifest.json"
    if not manifest_path.exists():
        return None

    metadata = _read_json_file(meeting_dir / "input" / "metadata.json")
    manifest = _read_json_file(manifest_path)
    manifest_metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
    result = manifest.get("result") if isinstance(manifest, dict) else None
    if isinstance(manifest_metadata, dict):
        metadata = {**metadata, **manifest_metadata}
    return write_meeting_manifest(
        meeting_dir,
        metadata,
        result if isinstance(result, dict) else None,
    )


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _available_artifacts(meeting_dir: Path) -> list[str]:
    artifacts_dir = meeting_dir / "artifacts"
    return [
        kind
        for kind, (filename, _media_type) in ARTIFACTS.items()
        if (artifacts_dir / filename).exists()
    ]


def _attach_report_health(status: dict, meeting_dir: Path) -> None:
    health_path = meeting_dir / "artifacts" / "report_health.json"
    if not health_path.exists():
        return
    try:
        health = json.loads(health_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        status["report_health_status"] = "failed"
        status["report_alerts"] = ["report_health.json is not valid JSON."]
        return
    status["report_health_status"] = health.get("status")
    status["report_generator"] = health.get("generated_by")
    status["report_alerts"] = health.get("alerts") or []


def _sanitize_status_payload(value: object) -> object:
    if isinstance(value, dict):
        return {key: _sanitize_status_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_status_payload(item) for item in value]
    if isinstance(value, str) and API_KEY:
        return value.replace(API_KEY, "[redacted]")
    return value


async def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            file_obj.write(chunk)
    await upload.close()


def _meeting_dir(meeting_id: str) -> Path:
    if "/" in meeting_id or "\\" in meeting_id or meeting_id.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid meeting id")
    return ROOT / "meetings" / meeting_id


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tracks(value: str | None) -> list[dict]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
