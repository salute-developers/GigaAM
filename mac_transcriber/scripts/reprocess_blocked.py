#!/usr/bin/env python3
"""Дообработка встреч, поставленных в очередь из-за недоступности AI.

Сценарии: AI-провайдер вернул "нет денег" -> `blocked_on_quota`; либо API был
недоступен (сеть/таймаут/5xx/нет ключа) -> `blocked_on_ai`. В обоих случаях сервис
сохранил транскрипт и НЕ выдал сырой local-отчёт. Когда AI снова доступен (пополнил
баланс / поднялся API), запусти этот скрипт: он берёт такие встречи ПО ОЧЕРЕДИ (старые
первыми), переобрабатывает их и ОСТАНАВЛИВАЕТСЯ на первой же, которая снова упёрлась в
недоступность (значит, AI ещё не готов — дальше не идём).

Можно повесить на расписание (cron/launchd) для автоматического дренажа очереди.

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

BLOCKED = ("blocked_on_quota", "blocked_on_ai")


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
        if status.get("status") in BLOCKED:
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
    from mac_transcriber.reporting import ReportQuotaError, ReportUnavailableError

    meetings_root = service.ROOT / "meetings"
    blocked = _blocked_meetings(meetings_root)
    if args.limit:
        blocked = blocked[: args.limit]

    if not blocked:
        print("No meetings are queued (blocked_on_quota/blocked_on_ai). Nothing to do.")
        return 0

    print(f"{len(blocked)} meeting(s) queued for AI (oldest first):")
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
        except (ReportQuotaError, ReportUnavailableError) as exc:
            remaining = len(blocked) - done
            reason = (
                "STILL out of quota"
                if isinstance(exc, ReportQuotaError)
                else "AI STILL unavailable"
            )
            print(
                f"   {reason}. Stopping. {remaining} meeting(s) left queued; "
                "retry when AI is back.",
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
