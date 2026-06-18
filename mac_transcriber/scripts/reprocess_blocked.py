#!/usr/bin/env python3
"""Дообработка встреч, поставленных на паузу из-за исчерпанной AI-квоты.

Сценарий: AI-провайдер вернул 429 "нет денег" -> сервис пометил встречу
`blocked_on_quota` и НЕ выдал сырой local-отчёт. После пополнения баланса
запусти этот скрипт: он берёт такие встречи ПО ОЧЕРЕДИ (старые первыми),
переобрабатывает их и ОСТАНАВЛИВАЕТСЯ на первой же встрече, которая снова
упёрлась в квоту (значит, денег опять не хватает — дальше не идём).

Использование:
    .venv/bin/python mac_transcriber/scripts/reprocess_blocked.py [--env-file .env.local] [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BLOCKED = "blocked_on_quota"


def _load_env_files(paths: list[Path]) -> None:
    for path in paths:
        expanded = path.expanduser()
        if not expanded.exists():
            continue
        for line in expanded.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _read_status(meeting_dir: Path) -> dict:
    path = meeting_dir / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _blocked_meetings(meetings_root: Path) -> list[Path]:
    found: list[tuple[str, Path]] = []
    for status_path in meetings_root.glob("*/status.json"):
        status = _read_status(status_path.parent)
        if status.get("status") == BLOCKED:
            # Сортируем по created_at (старые первыми), чтобы очередь была честной.
            found.append((str(status.get("created_at") or ""), status_path.parent))
    found.sort(key=lambda item: item[0])
    return [meeting_dir for _created, meeting_dir in found]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, action="append", default=None)
    parser.add_argument("--limit", type=int, help="Обработать не более N встреч.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать очередь, не обрабатывать.",
    )
    args = parser.parse_args(argv)

    _load_env_files(
        args.env_file if args.env_file is not None else [Path(".env.local")]
    )

    # service читает env на импорте — грузим env ДО импорта.
    from mac_transcriber import service
    from mac_transcriber.asr import regenerate_report_from_transcript
    from mac_transcriber.reporting import ReportQuotaError

    meetings_root = service.ROOT / "meetings"
    blocked = _blocked_meetings(meetings_root)
    if args.limit:
        blocked = blocked[: args.limit]

    if not blocked:
        print("No meetings are blocked_on_quota. Nothing to do.")
        return 0

    print(f"{len(blocked)} meeting(s) blocked_on_quota (oldest first):")
    for meeting_dir in blocked:
        print(f"  - {meeting_dir.name}")
    if args.dry_run:
        return 0

    done = 0
    for meeting_dir in blocked:
        meeting_id = meeting_dir.name
        print(f"\n== regenerating report for {meeting_id} (no re-ASR) ...", flush=True)
        try:
            # Только отчёт из готового транскрипта — ASR не перезапускаем.
            artifacts = regenerate_report_from_transcript(meeting_dir)
        except ReportQuotaError:
            remaining = len(blocked) - done
            print(
                f"   STILL out of quota. Stopping. {remaining} meeting(s) left paused; "
                "top up more and re-run.",
                flush=True,
            )
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED (left blocked): {exc}", flush=True)
            continue
        # Финализация (манифест + синк памяти) — через штатный сервисный путь.
        service._write_status(
            meeting_dir,
            "completed",
            phase="completed",
            progress=1.0,
            message="Reprocessed after top-up",
            result={
                "report_generator": artifacts.generated_by,
                "report_status": artifacts.status,
            },
        )
        print(f"   -> completed ({artifacts.generated_by})", flush=True)
        done += 1

    print(f"\nDone. Reprocessed {done}/{len(blocked)} meeting(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
