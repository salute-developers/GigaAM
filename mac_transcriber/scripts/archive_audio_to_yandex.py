#!/usr/bin/env python3
"""Архивирование ЗАПИСЕЙ встреч на Яндекс.Диск (REST API) с дедупликацией.

Грузит всю запись, как в Zoom: сведённый ``audio.m4a`` + раздельные дорожки спикеров
из ``input/participants/`` (переименованные по именам из metadata.zoom_participant_tracks,
напр. ``participants/Егор.m4a``). Раскладка: одна папка на запись с читаемым именем
``<ГГГГ-ММ-ДД> <имя>``. Транскрипты остаются локально; аудио уезжает на Диск,
локальная копия (опц.) удаляется только после подтверждённой заливки. Дедуп по sha256
(дубли zoom_*/UUID заливаются один раз). Производные ``*.diarization.*`` пропускаются.

WebDAV у Яндекса рвёт крупные файлы, поэтому REST API (cloud-api.yandex.net) — он даёт
upload-URL на storage-хост. Нужен OAuth-токен (cloud_api:disk.write) в .env.local:
  YANDEX_DISK_OAUTH_TOKEN, YANDEX_DISK_AUDIO_DIR, MAC_TRANSCRIBER_ROOT.

Безопасность: без флагов — DRY-RUN; --apply — заливка; --delete-local — удаление
локального только для подтверждённо залитого.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

API = "https://cloud-api.yandex.net/v1/disk"
AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".mp4", ".aac", ".flac", ".ogg", ".webm"}
HTTP_TIMEOUT = httpx.Timeout(1800.0, connect=30.0)


def _load_env_local(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _sanitize(label: str) -> str:
    return re.sub(r'[\\/:*?"<>|\n\r\t]+', "_", label).strip().strip(".")


def _is_zoom_legacy(meeting_id: str) -> bool:
    return meeting_id.startswith("zoom_")


TEST_KEYWORDS = ("test", "тест", "webhook", "harness", "example", "ping", "проверка")


def _speech_seconds(root: Path, meeting_id: str) -> float | None:
    """Суммарная длительность речи по транскрипту (None, если транскрипта нет)."""
    tj = root / "meetings" / meeting_id / "artifacts" / "transcript.json"
    try:
        data = json.loads(tj.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    segs = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(segs, list):
        return None
    return sum(
        float(s.get("end", 0)) - float(s.get("start", 0))
        for s in segs
        if isinstance(s, dict)
    )


def _is_test_name(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in TEST_KEYWORDS)


def _readable_folder(root: Path, meeting_id: str) -> str:
    d = root / "meetings" / meeting_id
    date = ""
    try:
        date = str(
            json.loads((d / "status.json").read_text("utf-8")).get("created_at") or ""
        )[:10]
    except (OSError, json.JSONDecodeError):
        pass
    name = ""
    try:
        meta = json.loads((d / "input" / "metadata.json").read_text("utf-8"))
        name = os.path.splitext(str(meta.get("source_filename") or ""))[0]
    except (OSError, json.JSONDecodeError):
        pass
    label = (
        f"{date} {name}".strip() if (date and name) else (name or date or meeting_id)
    )
    return _sanitize(label) or meeting_id


def _speaker_map(root: Path, meeting_id: str) -> dict[str, str]:
    """basename дорожки ('01.m4a') -> имя спикера ('Егор') из metadata."""
    out: dict[str, str] = {}
    try:
        meta = json.loads(
            (root / "meetings" / meeting_id / "input" / "metadata.json").read_text(
                "utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return out
    for t in meta.get("zoom_participant_tracks") or []:
        base = str(t.get("audio_url") or "").rsplit("/", 1)[-1]
        name = str(t.get("speaker_name") or "").strip()
        if base and name:
            out[base] = name
    return out


def _remote_rel(rel: Path, speakers: dict[str, str]) -> str:
    """Относительный путь файла на Диске; дорожки спикеров переименованы по именам."""
    if rel.parent.name == "participants":
        spk = speakers.get(rel.name)
        if spk:
            # Номер дорожки сохраняем как префикс: один спикер может иметь несколько
            # треков (переподключения) — иначе одинаковое имя перезатёрло бы файл.
            return f"participants/{rel.stem} {_sanitize(spk)}{rel.suffix}"
    return rel.as_posix()


def _discover(root: Path) -> list[dict]:
    items: list[dict] = []
    for mdir in sorted((root / "meetings").glob("*")):
        inp = mdir / "input"
        if not (mdir.is_dir() and inp.is_dir()):
            continue
        # Только завершённые встречи: нельзя удалять/трогать аудио ещё
        # обрабатывающейся встречи (status processing/uploaded/failed).
        try:
            status = json.loads((mdir / "status.json").read_text("utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = None
        if status != "completed":
            continue
        speakers = _speaker_map(root, mdir.name)
        for f in sorted(inp.rglob("*")):
            if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
                continue
            if ".diarization." in f.name:  # производный файл, не часть записи
                continue
            rel = f.relative_to(inp)
            items.append(
                {
                    "meeting_id": mdir.name,
                    "path": f,
                    "size": f.stat().st_size,
                    "remote_rel": _remote_rel(rel, speakers),
                }
            )
    return items


def _mkdir(client: httpx.Client, disk_path: str) -> None:
    client.put(f"{API}/resources", params={"path": disk_path})  # 201/409 — оба ок


def _remote_size(client: httpx.Client, disk_path: str) -> int | None:
    r = client.get(f"{API}/resources", params={"path": disk_path, "fields": "size"})
    return r.json().get("size") if r.status_code == 200 else None


def _upload(client: httpx.Client, disk_path: str, data: bytes) -> None:
    r = client.get(
        f"{API}/resources/upload", params={"path": disk_path, "overwrite": "true"}
    )
    if r.status_code != 200:
        raise RuntimeError(f"upload-url {r.status_code}: {r.text[:160]}")
    put = httpx.put(r.json()["href"], content=data, timeout=HTTP_TIMEOUT)
    if put.status_code not in (200, 201, 202):
        raise RuntimeError(f"PUT {put.status_code}: {put.text[:160]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--delete-local", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--include-all", action="store_true", help="не отсекать пустые/тестовые встречи"
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    _load_env_local(repo_root / ".env.local")
    token = os.environ.get("YANDEX_DISK_OAUTH_TOKEN", "").strip()
    audio_dir = os.environ.get("YANDEX_DISK_AUDIO_DIR", "").strip().strip('"')
    if not (token and audio_dir):
        print(
            "ERROR: нужен YANDEX_DISK_OAUTH_TOKEN и YANDEX_DISK_AUDIO_DIR в .env.local"
        )
        return 1
    root = Path(
        os.environ.get("MAC_TRANSCRIBER_ROOT")
        or (repo_root / ".local" / "mac_transcriber")
    )
    if not (root / "meetings").is_dir():
        print(f"ERROR: не найден каталог встреч: {root}/meetings")
        return 1

    items = _discover(root)

    # Отсекаем пустые/тихие/тестовые встречи (если не --include-all).
    if not args.include_all:
        min_speech = float(
            os.environ.get("MAC_TRANSCRIBER_ARCHIVE_MIN_SPEECH_SEC", "60")
        )
        skip: dict[str, str] = {}
        for mid in {it["meeting_id"] for it in items}:
            sec = _speech_seconds(root, mid)
            max_sz = max(
                (it["size"] for it in items if it["meeting_id"] == mid), default=0
            )
            name = _readable_folder(root, mid)
            if max_sz == 0:
                skip[mid] = "пустое аудио"
            elif sec is not None and sec < min_speech:
                skip[mid] = f"речь {sec:.0f}с < {min_speech:.0f}с"
            elif sec is None and max_sz < 60_000:
                skip[mid] = f"нет транскрипта, аудио {max_sz / 1000:.0f}KB"
            elif _is_test_name(name) and (sec is None or sec < min_speech * 5):
                skip[mid] = "тест-имя + мало речи"
        if skip:
            print(
                f"Пропускаю {len(skip)} встреч(и) (пусто/тихо/тест; --include-all = залить всё):"
            )
            for mid in sorted(skip):
                print(f"  - {_readable_folder(root, mid)}: {skip[mid]}")
            print()
        items = [it for it in items if it["meeting_id"] not in skip]

    if args.limit:
        items = items[: args.limit]
    if not items:
        print("Аудиофайлов не найдено (после фильтра).")
        return 0

    by_hash: dict[str, list[dict]] = {}
    print(f"Хеширую {len(items)} файл(ов) (сведённые + дорожки спикеров)…")
    for it in items:
        it["sha256"] = _sha256(it["path"])
        by_hash.setdefault(it["sha256"], []).append(it)

    total = sum(it["size"] for it in items)
    uniq = sum(g[0]["size"] for g in by_hash.values())
    print(
        f"\nФайлов: {len(items)} | уникальных: {len(by_hash)} | локально: {_human(total)} | "
        f"к заливке: {_human(uniq)} | дубли освободят: {_human(total - uniq)}"
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    if args.apply and args.delete_local:
        mode += " + DELETE-LOCAL"
    print(
        f"Режим: {mode} (REST)  →  disk:/{audio_dir}/<ГГГГ-ММ-ДД имя>/<файл|participants/Имя>\n"
    )

    # Имя папки записи закрепляем за каждой встречей (по её первому файлу).
    folder_by_meeting: dict[str, str] = {}
    used: dict[str, str] = {}
    for it in items:
        mid = it["meeting_id"]
        if mid in folder_by_meeting:
            continue
        name = _readable_folder(root, mid)
        if used.get(name, mid) != mid:
            name = f"{name} [{mid[:8]}]"
        used[name] = mid
        folder_by_meeting[mid] = name

    uploaded = skipped = deleted = errors = 0
    made_dirs: set[str] = set()
    with httpx.Client(
        headers={"Authorization": f"OAuth {token}"}, timeout=HTTP_TIMEOUT
    ) as client:
        if args.apply:
            _mkdir(client, f"/{audio_dir}")
            made_dirs.add(f"/{audio_dir}")
        for sha, group in by_hash.items():
            canonical = sorted(
                group, key=lambda x: (_is_zoom_legacy(x["meeting_id"]), x["meeting_id"])
            )[0]
            folder = folder_by_meeting[canonical["meeting_id"]]
            disk_path = f"/{audio_dir}/{folder}/{canonical['remote_rel']}"
            tag = f"{folder}/{canonical['remote_rel']} ({_human(canonical['size'])})"
            dup = f"  [+{len(group) - 1} дубл.]" if len(group) > 1 else ""
            if not args.apply:
                print(f"  PLAN  {tag}{dup}")
                continue
            try:
                # создаём недостающие родительские папки (вкл. participants/)
                parent = disk_path.rsplit("/", 1)[0]
                acc = ""
                for part in parent.strip("/").split("/"):
                    acc += "/" + part
                    if acc not in made_dirs:
                        _mkdir(client, acc)
                        made_dirs.add(acc)
                if _remote_size(client, disk_path) == canonical["size"]:
                    print(f"  SKIP  {tag} (уже на Диске){dup}")
                    skipped += 1
                else:
                    _upload(client, disk_path, canonical["path"].read_bytes())
                    if _remote_size(client, disk_path) != canonical["size"]:
                        raise RuntimeError("verify: размер на Диске не совпал")
                    uploaded += 1
                    print(f"  UP    {tag}{dup}")
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"  ERR   {tag}: {exc}")
                continue
            remote = _remote_size(client, disk_path)
            for m in group:
                _record_archive(root, m, disk_path, sha, remote)
                if args.delete_local and remote == m["size"]:
                    try:
                        m["path"].unlink()
                        deleted += 1
                    except OSError as exc:
                        print(f"        delete failed {m['meeting_id']}: {exc}")

    print(
        f"\nИтог: uploaded={uploaded} skipped={skipped} deleted={deleted} errors={errors}"
    )
    if not args.apply:
        print("DRY-RUN. Заливка: --apply. Удаление локального: + --delete-local.")
    return 0 if errors == 0 else 2


def _record_archive(
    root: Path, member: dict, disk_path: str, sha: str, size: int | None
) -> None:
    sp = root / "meetings" / member["meeting_id"] / "status.json"
    try:
        status = json.loads(sp.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    archive = status.get("archived_audio")
    if not isinstance(archive, dict) or "files" not in archive:
        archive = {"files": []}
    entry = {
        "disk_path": disk_path,
        "sha256": sha,
        "size": size,
        "local_name": member["path"].name,
    }
    archive["files"] = [
        f for f in archive["files"] if f.get("disk_path") != disk_path
    ] + [entry]
    archive["archived_at"] = datetime.now(UTC).isoformat()
    status["archived_audio"] = archive
    sp.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
