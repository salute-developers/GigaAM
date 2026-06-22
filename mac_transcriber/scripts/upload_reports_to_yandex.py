#!/usr/bin/env python3
"""Заливка ОТЧЁТОВ встреч на Яндекс.Диск (REST API), по одной подпапке на встречу.

Кладёт отчёты в отдельную папку Диска (по умолч. «Отчёты встреч Coherent [ZOOM]»),
по подпапке на УНИКАЛЬНУЮ встречу (дедуп по mapping.json). Имя подпапки совпадает с
аудио-папкой архива (берётся из status.json → archived_audio.disk_path), иначе строится
из created_at + source_filename. Грузит report.pdf / report.md / report.html.

Токен и базовая папка — из .env.local (YANDEX_DISK_OAUTH_TOKEN). Секреты не печатаем.
Без флагов — DRY-RUN; --apply — заливка.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import httpx

API = "https://cloud-api.yandex.net/v1/disk"
HTTP_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
DEFAULT_REPORTS_DIR = "Отчёты встреч Coherent [ZOOM]"
REPORT_FILES = ["report.pdf", "report.md", "report.html"]


def _load_env_local(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _sanitize(label: str) -> str:
    return re.sub(r'[\\/:*?"<>|\n\r\t]+', "_", label).strip().strip(".")


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _status(meeting_dir: Path) -> dict:
    for p in (meeting_dir / "status.json", meeting_dir / "artifacts" / "status.json"):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def _folder_label(group_dirs: list[Path], audio_dir: str) -> str:
    """Имя подпапки: как у аудио-архива (из archived_audio.disk_path), иначе из метаданных."""
    for d in group_dirs:
        st = _status(d)
        arch = st.get("archived_audio") if isinstance(st, dict) else None
        for f in (arch or {}).get("files", []) if isinstance(arch, dict) else []:
            dp = str(f.get("disk_path") or "")
            parts = dp.strip("/").split("/")
            if len(parts) >= 3 and parts[0] == audio_dir:
                return parts[1]
    # fallback
    for d in group_dirs:
        st = _status(d)
        meta = st.get("metadata", {}) if isinstance(st, dict) else {}
        date = str(st.get("created_at") or "")[:10]
        src = os.path.splitext(str(meta.get("source_filename") or ""))[0]
        label = f"{date} {src}".strip() if (date and src) else (src or date or d.name)
        if label:
            return _sanitize(label)
    return _sanitize(group_dirs[0].name)


def _report_source(group_dirs: list[Path]) -> Path | None:
    """Папка артефактов с готовым отчётом (приоритет — не-zoom UUID)."""
    ranked = sorted(group_dirs, key=lambda d: 1 if d.name.startswith("zoom_") else 0)
    for d in ranked:
        if (d / "artifacts" / "report.pdf").exists() or (
            d / "artifacts" / "report.md"
        ).exists():
            return d / "artifacts"
    return None


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


def _report_title(src: Path) -> str:
    """Заголовок отчёта из report.json (meeting.title)."""
    rj = src / "report.json"
    if rj.exists():
        try:
            return str(
                (
                    json.loads(rj.read_text(encoding="utf-8")).get("meeting", {}) or {}
                ).get("title", "")
            ).strip()
        except json.JSONDecodeError:
            return ""
    return ""


def _clean_title(t: str) -> str:
    """Заголовок -> имя папки: ':' -> ' —', срезаем запрещённые символы, ограничиваем длину."""
    t = re.sub(r"(?<!\d):(?!\d)", " — ", t)  # ':' как разделитель -> тире
    t = t.replace(":", "-")  # оставшиеся (напр. время 16:00) -> 16-00
    t = re.sub(r'[\\/*?"<>|\n\r\t]+', " ", t)
    t = re.sub(r"\s+", " ", t).strip().strip(".")
    if len(t) > 120:
        t = t[:120].rsplit(" ", 1)[0].strip()  # по границе слова
    return t


def _date_for(group_dirs: list[Path], old_label: str) -> str:
    """Дата встречи: из начала старого имени (ГГГГ-ММ-ДД) либо из status.created_at."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", old_label)
    if m:
        return m.group(1)
    for d in group_dirs:
        st = _status(d)
        date = str(st.get("created_at") or "")[:10]
        if re.match(r"\d{4}-\d{2}-\d{2}", date):
            return date
    return ""


def _move(client: httpx.Client, src: str, dst: str) -> None:
    r = client.post(
        f"{API}/resources/move",
        params={"from": src, "path": dst, "overwrite": "false"},
    )
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"move {r.status_code}: {r.text[:160]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", help="путь к mapping.json (массовый режим)")
    ap.add_argument(
        "--meeting-dirs",
        help="запятая-разделённый список папок встреч; имя на Диске = '<дата> <заголовок>'",
    )
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    ap.add_argument(
        "--rename",
        action="store_true",
        help="переименовать уже залитые папки (mapping-режим) в '<дата> <заголовок>' (move)",
    )
    args = ap.parse_args(argv)
    if not (args.mapping or args.meeting_dirs):
        print("ERROR: укажи --mapping ИЛИ --meeting-dirs")
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    _load_env_local(repo_root / ".env.local")
    token = os.environ.get("YANDEX_DISK_OAUTH_TOKEN", "").strip()
    audio_dir = os.environ.get("YANDEX_DISK_AUDIO_DIR", "").strip().strip('"')
    if not token:
        print("ERROR: нужен YANDEX_DISK_OAUTH_TOKEN в .env.local")
        return 1

    base = f"/{args.reports_dir}"
    plan = []
    if args.meeting_dirs:
        # Режим отдельных встреч: имя папки сразу титульное '<дата> <заголовок>'.
        for d in [Path(p.strip()) for p in args.meeting_dirs.split(",") if p.strip()]:
            src = d / "artifacts"
            title = _report_title(src)
            ct = _clean_title(title) if title else ""
            date = _date_for([d], "")
            label = f"{date} {ct}".strip() if (date and ct) else (ct or d.name)
            files = [(src / f) for f in REPORT_FILES if (src / f).exists()]
            if not files:
                plan.append(
                    {
                        "key": d.name,
                        "label": label,
                        "group": [d],
                        "error": "нет report.*",
                    }
                )
            else:
                plan.append(
                    {
                        "key": d.name,
                        "label": label,
                        "group": [d],
                        "src": src,
                        "files": files,
                    }
                )
    else:
        mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
        for key, v in mapping.items():
            group = [Path(p) for p in v["all_dirs"]]
            label = _folder_label(group, audio_dir)
            src = _report_source(group)
            if src is None:
                plan.append(
                    {
                        "key": key,
                        "label": label,
                        "group": group,
                        "error": "нет report.* в группе",
                    }
                )
                continue
            files = [(src / f) for f in REPORT_FILES if (src / f).exists()]
            plan.append(
                {
                    "key": key,
                    "label": label,
                    "group": group,
                    "src": src,
                    "files": files,
                    "words": v.get("words"),
                }
            )

    # Дизамбигуация: разные встречи с одинаковым именем (напр. два «бд.m4a»)
    # не должны делить папку и затирать друг друга — добавляем суффикс [key].
    from collections import Counter

    label_counts = Counter(p["label"] for p in plan)
    for p in plan:
        if label_counts[p["label"]] > 1:
            p["label"] = f"{p['label']} [{p['key']}]"

    if args.rename:
        for p in plan:
            if "error" in p:
                p["new_label"] = None
                continue
            title = _report_title(p["src"])
            date = _date_for(p["group"], p["label"])
            nl = _clean_title(title)
            p["new_label"] = (f"{date} {nl}".strip() if date else nl) or p["label"]
        new_counts = Counter(p["new_label"] for p in plan if p.get("new_label"))
        for p in plan:
            if p.get("new_label") and new_counts[p["new_label"]] > 1:
                p["new_label"] = f"{p['new_label']} [{p['key']}]"

        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"Режим RENAME: {mode}  →  переименование папок в disk:{base}\n")
        renamed = same = errors = 0
        with httpx.Client(
            headers={"Authorization": f"OAuth {token}"}, timeout=HTTP_TIMEOUT
        ) as client:
            for p in sorted(plan, key=lambda x: x.get("label", "")):
                if "error" in p or not p.get("new_label"):
                    continue
                if p["label"] == p["new_label"]:
                    print(f"  SAME  {p['label']}")
                    same += 1
                    continue
                print(
                    f"  {'MOVE' if args.apply else 'PLAN'}  {p['label']}  →  {p['new_label']}"
                )
                if not args.apply:
                    continue
                try:
                    _move(client, f"{base}/{p['label']}", f"{base}/{p['new_label']}")
                    renamed += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERR   {p['label']}: {exc}")
                    errors += 1
        print(f"\nИтог RENAME: renamed={renamed} same={same} errors={errors}")
        if not args.apply:
            print("DRY-RUN. Применить: --rename --apply.")
        return 0 if errors == 0 else 2

    ok_plan = [p for p in plan if "files" in p]
    total_files = sum(len(p["files"]) for p in ok_plan)
    total_bytes = sum(f.stat().st_size for p in ok_plan for f in p["files"])
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Режим: {mode}  →  disk:{base}/<встреча>/<report.pdf|md|html>")
    print(
        f"Встреч: {len(ok_plan)} | файлов: {total_files} | объём: {_human(total_bytes)}\n"
    )

    uploaded = skipped = errors = 0
    with httpx.Client(
        headers={"Authorization": f"OAuth {token}"}, timeout=HTTP_TIMEOUT
    ) as client:
        if args.apply:
            _mkdir(client, base)
        for p in sorted(plan, key=lambda x: x.get("label", "")):
            if "error" in p:
                print(f"  ERR   [{p['key']}] {p['label']}: {p['error']}")
                errors += 1
                continue
            folder = f"{base}/{p['label']}"
            if not args.apply:
                names = ", ".join(f.name for f in p["files"])
                print(f"  PLAN  {p['label']}/  ({names})")
                continue
            _mkdir(client, folder)
            for f in p["files"]:
                disk_path = f"{folder}/{f.name}"
                size = f.stat().st_size
                try:
                    if _remote_size(client, disk_path) == size:
                        print(f"  SKIP  {p['label']}/{f.name} (уже на Диске)")
                        skipped += 1
                        continue
                    _upload(client, disk_path, f.read_bytes())
                    if _remote_size(client, disk_path) != size:
                        raise RuntimeError("verify: размер не совпал")
                    print(f"  UP    {p['label']}/{f.name} ({_human(size)})")
                    uploaded += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERR   {p['label']}/{f.name}: {exc}")
                    errors += 1

    print(f"\nИтог: uploaded={uploaded} skipped={skipped} errors={errors}")
    if not args.apply:
        print("DRY-RUN. Заливка: --apply.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
