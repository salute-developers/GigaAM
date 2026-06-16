#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import quote

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mac_transcriber.asr import transcribe_meeting
from mac_transcriber.zoom_import import sanitize_zoom_recording_metadata


DEFAULT_WORK_ROOT = ".local/mac_transcriber/meetings"
DEFAULT_MODEL = "v3_e2e_rnnt"
DEFAULT_CACHE_DIR = "/tmp/gigaam-cache"
ZOOM_API_BASE_URL = "https://api.zoom.us/v2"
ZOOM_OAUTH_BASE_URL = "https://zoom.us"
RUNTIME_ENV_KEYS = {
    "MAC_TRANSCRIBER_DATABASE_URL",
    "MAC_TRANSCRIBER_POSTGRES_DSN",
    "MAC_TRANSCRIBER_REPORT_MODE",
    "MAC_TRANSCRIBER_REPORT_MODEL",
    "MAC_TRANSCRIBER_REPORT_PDF",
    "OPENAI_API_KEY",
}


@dataclass(frozen=True)
class PreparedMeeting:
    meeting_id: str
    meeting_dir: Path
    title: str
    participant_tracks: int


class ZoomClient:
    def __init__(
        self,
        *,
        account_id: str,
        client_id: str,
        client_secret: str,
        api_base_url: str = ZOOM_API_BASE_URL,
        oauth_base_url: str = ZOOM_OAUTH_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.account_id = account_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_base_url = api_base_url.rstrip("/")
        self.oauth_base_url = oauth_base_url.rstrip("/")
        self.timeout = timeout
        self._access_token: str | None = None

    def get_access_token(self) -> str:
        if self._access_token:
            return self._access_token
        credentials = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        auth = base64.b64encode(credentials).decode("ascii")
        response = self._request_with_retries(
            "POST",
            f"{self.oauth_base_url}/oauth/token",
            params={"grant_type": "account_credentials", "account_id": self.account_id},
            headers={"Authorization": f"Basic {auth}"},
        )
        self._access_token = str(response.json()["access_token"])
        return self._access_token

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.get_access_token()}"}

    def list_recordings(
        self,
        *,
        user_id: str,
        start_date: str,
        end_date: str,
        page_size: int,
    ) -> list[dict[str, object]]:
        meetings: list[dict[str, object]] = []
        next_page_token = ""
        while True:
            params: dict[str, object] = {
                "from": start_date,
                "to": end_date,
                "page_size": page_size,
            }
            if next_page_token:
                params["next_page_token"] = next_page_token
            response = self._request_with_retries(
                "GET",
                f"{self.api_base_url}/users/{user_id}/recordings",
                params=params,
                headers=self.auth_headers(),
            )
            payload = response.json()
            meetings.extend(item for item in payload.get("meetings", []) if isinstance(item, dict))
            next_page_token = str(payload.get("next_page_token") or "")
            if not next_page_token:
                break
        return meetings

    def get_recordings(self, *, meeting_id: str) -> dict[str, object]:
        encoded = zoom_recording_api_id(meeting_id)
        response = self._request_with_retries(
            "GET",
            f"{self.api_base_url}/meetings/{encoded}/recordings",
            headers=self.auth_headers(),
        )
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def download_recording_file(self, url: str) -> bytes:
        response = self._request_with_retries("GET", url, headers=self.auth_headers(), follow_redirects=True)
        return response.content

    def _request_with_retries(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                follow_redirects = bool(kwargs.get("follow_redirects", False))
                request_kwargs = {key: value for key, value in kwargs.items() if key != "follow_redirects"}
                with httpx.Client(timeout=self.timeout, follow_redirects=follow_redirects) as client:
                    response = client.request(method, url, **request_kwargs)
                    response.raise_for_status()
                return response
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(attempt * 2)
        assert last_error is not None
        raise last_error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = load_env_files(args.env_file)
    results = backfill_zoom_recordings(
        env=env,
        start_date=args.start_date,
        end_date=args.end_date,
        work_root=args.work_root.expanduser().resolve(),
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
        download_only=args.download_only,
        page_size=args.page_size,
        model=args.model,
        cache_dir=str(args.cache_dir.expanduser()),
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Zoom cloud recordings locally.")
    parser.add_argument("--from", dest="start_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--env-file",
        type=Path,
        action="append",
        default=[Path(".env.local"), Path(".local/secrets/slack_zoom-vps.env")],
        help="Env file to load; may be repeated",
    )
    parser.add_argument("--work-root", type=Path, default=Path(DEFAULT_WORK_ROOT))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args(argv)


def load_env_files(paths: list[Path]) -> dict[str, str]:
    env = dict(os.environ)
    for path in paths:
        expanded = path.expanduser()
        if not expanded.exists():
            continue
        for line in expanded.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def backfill_zoom_recordings(
    *,
    env: Mapping[str, str],
    start_date: str,
    end_date: str,
    work_root: Path,
    dry_run: bool,
    limit: int | None,
    force: bool,
    download_only: bool,
    page_size: int,
    model: str = DEFAULT_MODEL,
    cache_dir: str = DEFAULT_CACHE_DIR,
    device: str = "cpu",
    batch_size: int = 4,
) -> list[dict[str, object]]:
    client = ZoomClient(
        account_id=required_env(env, "ZOOM_ACCOUNT_ID"),
        client_id=required_env(env, "ZOOM_CLIENT_ID"),
        client_secret=required_env(env, "ZOOM_CLIENT_SECRET"),
    )
    apply_runtime_env(env)
    recordings = client.list_recordings(
        user_id=env.get("ZOOM_HOST_USER_ID", "me"),
        start_date=start_date,
        end_date=end_date,
        page_size=page_size,
    )
    if limit is not None:
        recordings = recordings[: max(0, limit)]

    results: list[dict[str, object]] = []
    for recording in recordings:
        meeting_id = zoom_meeting_id(recording)
        title = recording_title(recording)
        if dry_run:
            results.append({"meeting_id": meeting_id, "status": "dry_run", "title": title})
            continue
        try:
            recording_detail = client.get_recordings(meeting_id=zoom_recording_lookup_id(recording))
            prepared = prepare_meeting_dir(
                recording={**recording, **recording_detail},
                client=client,
                work_root=work_root,
                force=force,
            )
            if download_only:
                results.append(
                    {
                        "meeting_id": prepared.meeting_id,
                        "status": "downloaded",
                        "title": prepared.title,
                        "meeting_dir": str(prepared.meeting_dir),
                        "participant_tracks": prepared.participant_tracks,
                    }
                )
                continue
            os.environ["MAC_TRANSCRIBER_DIARIZATION"] = "0"
            result = transcribe_meeting(
                meeting_dir=prepared.meeting_dir,
                model_name=model,
                cache_dir=cache_dir,
                device=device,
                batch_size=batch_size,
            )
            results.append(
                {
                    "meeting_id": prepared.meeting_id,
                    "status": "transcribed",
                    "title": prepared.title,
                    "meeting_dir": str(prepared.meeting_dir),
                    "participant_tracks": prepared.participant_tracks,
                    "segments": result.get("segments"),
                    "tracks": result.get("tracks"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"meeting_id": meeting_id, "status": "failed", "title": title, "error": str(exc)})
    return results


def prepare_meeting_dir(
    *,
    recording: Mapping[str, object],
    client: object,
    work_root: Path,
    force: bool,
) -> PreparedMeeting:
    meeting_id = zoom_meeting_id(recording)
    title = recording_title(recording)
    meeting_dir = work_root / meeting_id
    input_dir = meeting_dir / "input"
    participants_dir = input_dir / "participants"
    if meeting_dir.exists():
        if not force and (input_dir / "audio.m4a").exists():
            metadata = load_metadata(input_dir / "metadata.json")
            participant_files = sorted(participants_dir.glob("*.m4a"))
            if valid_existing_participant_tracks(metadata, participant_files):
                tracks = metadata["zoom_participant_tracks"]
                return PreparedMeeting(
                    meeting_id=meeting_id,
                    meeting_dir=meeting_dir,
                    title=str(metadata.get("title") or title) if isinstance(metadata, dict) else title,
                    participant_tracks=len(tracks),
                )
        shutil.rmtree(meeting_dir)
    participants_dir.mkdir(parents=True, exist_ok=True)

    participant_files = participant_audio_files(recording)
    if not participant_files:
        raise ValueError("recording does not contain participant_audio_files")
    downloadable_participant_files = [
        participant_file for participant_file in participant_files if participant_file.get("download_url")
    ]
    if len(downloadable_participant_files) != len(participant_files):
        raise ValueError("recording participant_audio_files are missing download_url")

    primary_file = select_primary_audio_file(recording)
    audio_bytes = client.download_recording_file(str(primary_file["download_url"]))
    (input_dir / "audio.m4a").write_bytes(audio_bytes)

    tracks = []
    participants = []
    for index, participant_file in enumerate(downloadable_participant_files, start=1):
        download_url = str(participant_file["download_url"])
        participant_name = participant_track_name(participant_file) or f"Zoom participant {index}"
        (participants_dir / f"{index:02d}.m4a").write_bytes(
            client.download_recording_file(download_url)
        )
        participants.append(participant_name)
        tracks.append(
            {
                "label": f"zoom_participant_{index:02d}",
                "speaker_name": participant_name,
                "source_file_id": str(participant_file.get("id") or participant_file.get("file_id") or ""),
            }
        )

    metadata = {
        **sanitize_zoom_recording_metadata(recording),
        "meeting_id": meeting_id,
        "zoom_uuid": str(recording.get("uuid") or ""),
        "zoom_meeting_id": str(recording.get("id") or ""),
        "title": title,
        "source": "zoom_cloud_backfill",
        "source_filename": "audio.m4a",
        "language_code": "ru",
        "processing_mode": "zoom_participant_tracks",
        "participants": participants,
        "zoom_participant_tracks": tracks,
        "primary_recording_file": sanitize_zoom_recording_metadata(primary_file),
    }
    (input_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PreparedMeeting(
        meeting_id=meeting_id,
        meeting_dir=meeting_dir,
        title=title,
        participant_tracks=len(tracks),
    )


def valid_existing_participant_tracks(metadata: dict[str, object], participant_files: list[Path]) -> bool:
    tracks = metadata.get("zoom_participant_tracks") if isinstance(metadata, dict) else []
    if not isinstance(tracks, list) or not tracks:
        return False
    return len(participant_files) == len(tracks)


def load_metadata(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def select_primary_audio_file(recording: Mapping[str, object]) -> dict[str, object]:
    files = [item for item in recording.get("recording_files", []) if isinstance(item, dict)]
    audio_files = [
        item
        for item in files
        if str(item.get("file_type", "")).upper() == "M4A"
        or str(item.get("recording_type", "")).lower() == "audio_only"
        or str(item.get("file_extension", "")).upper() == "M4A"
    ]
    if audio_files:
        return sorted(audio_files, key=recording_priority)[0]
    mp4_files = [
        item
        for item in files
        if str(item.get("file_type", "")).upper() == "MP4"
        or str(item.get("file_extension", "")).upper() == "MP4"
    ]
    if mp4_files:
        return sorted(mp4_files, key=recording_priority)[0]
    raise ValueError("recording does not contain M4A or MP4")


def participant_audio_files(recording: Mapping[str, object]) -> list[dict[str, object]]:
    return [item for item in recording.get("participant_audio_files", []) if isinstance(item, dict)]


def recording_priority(recording_file: Mapping[str, object]) -> tuple[int, int]:
    recording_type = str(recording_file.get("recording_type", "")).lower()
    file_type = str(recording_file.get("file_type", "")).upper()
    if recording_type == "audio_only" or file_type == "M4A":
        kind_rank = 0
    elif file_type == "MP4":
        kind_rank = 1
    else:
        kind_rank = 2
    return kind_rank, -int(recording_file.get("file_size") or 0)


def participant_track_name(recording_file: Mapping[str, object]) -> str | None:
    for key in ("participant_name", "user_name", "name", "display_name"):
        value = clean_participant_name(recording_file.get(key))
        if value:
            return value
    return clean_participant_name(name_from_filename(recording_file.get("file_name")))


def name_from_filename(value: object) -> str | None:
    if not value:
        return None
    stem = Path(str(value)).stem
    normalized = stem.casefold()
    for separator in (" - ", " — ", "_-_"):
        prefix = f"audio only{separator}"
        if normalized.startswith(prefix):
            return stem.split(separator, 1)[1]
    return None


def clean_participant_name(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.casefold()
    if lowered.replace("_", " ").replace("-", " ") in {"audio only", "unknown"}:
        return None
    return text[:120]


def zoom_meeting_id(recording: Mapping[str, object]) -> str:
    source = str(recording.get("uuid") or recording.get("meeting_id") or recording.get("id") or "zoom")
    return "zoom_" + safe_name(source)


def zoom_recording_lookup_id(recording: Mapping[str, object]) -> str:
    return str(recording.get("uuid") or recording.get("meeting_id") or recording.get("id") or "")


def zoom_recording_api_id(meeting_id: str) -> str:
    encoded = quote(meeting_id, safe="")
    if meeting_id.startswith("/") or "//" in meeting_id:
        encoded = quote(encoded, safe="")
    return encoded


def recording_title(recording: Mapping[str, object]) -> str:
    return str(recording.get("topic") or recording.get("title") or zoom_meeting_id(recording)).strip()


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip())
    return safe or "meeting"


def required_env(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise RuntimeError(f"{key} is not configured")
    return value


def apply_runtime_env(env: Mapping[str, str]) -> None:
    for key in RUNTIME_ENV_KEYS:
        value = env.get(key)
        if value:
            os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
