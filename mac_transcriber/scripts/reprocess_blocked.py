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


def _speech_seconds(meeting_dir: Path) -> float:
    """Суммарная длительность речи по транскрипту (0, если транскрипта нет)."""
    tj = meeting_dir / "artifacts" / "transcript.json"
    try:
        data = json.loads(tj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    segs = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(segs, list):
        return 0.0
    return sum(
        float(s.get("end", 0)) - float(s.get("start", 0))
        for s in segs
        if isinstance(s, dict)
    )


def _blocked_meetings(meetings_root: Path) -> list[Path]:
    found: list[tuple[str, Path]] = []
    for status_path in meetings_root.glob("*/status.json"):
        status = _read_status(status_path.parent)
        if status.get("status") in BLOCKED:
            # Сортируем по created_at (старые первыми), чтобы очередь была честной.
            found.append((str(status.get("created_at") or ""), status_path.parent))
    found.sort(key=lambda item: item[0])
    return [meeting_dir for _created, meeting_dir in found]


# Макс. попыток авто-перегенерации завершённой встречи (защита от зацикливания на
# встрече, которая упорно срывается в local-фоллбэк).
MAX_REGEN_ATTEMPTS = 2


def _needs_regen_meetings(meetings_root: Path, exclude: list[Path]) -> list[Path]:
    """Завершённые встречи без нормального AI-отчёта: заглушка или local-фоллбэк.

    Старые встречи, обработанные до включения AI-отчётов (report.md — заглушка, нет
    report_health.json), и сорвавшиеся в сырой local-«отчёт». У них есть транскрипт —
    регенерим отчёт из него (без повторного ASR) текущей моделью (gpt-5.5 + критик).
    """
    excluded = {m.resolve() for m in exclude}
    found: list[tuple[str, Path]] = []
    for status_path in meetings_root.glob("*/status.json"):
        meeting_dir = status_path.parent
        if meeting_dir.resolve() in excluded:
            continue
        status = _read_status(meeting_dir)
        if status.get("status") != "completed":
            continue
        if not (meeting_dir / "artifacts" / "transcript.json").exists():
            continue  # без транскрипта регенерить нечего
        if _speech_seconds(meeting_dir) < 60:
            continue  # пустые/тестовые встречи (мало речи) не регенерим
        if int(status.get("report_regen_count") or 0) >= MAX_REGEN_ATTEMPTS:
            continue
        health = meeting_dir / "artifacts" / "report_health.json"
        generated_by = ""
        if health.exists():
            try:
                generated_by = str(
                    json.loads(health.read_text(encoding="utf-8")).get("generated_by")
                    or ""
                )
            except json.JSONDecodeError:
                generated_by = ""
        # Заглушка (health не сгенерился), regex-фоллбэк или странный facts-regen-путь
        # -> на перегенерацию штатным gpt-5.5 + критик.
        if (
            (not health.exists())
            or generated_by == "local"
            or "facts-regen" in generated_by
        ):
            found.append((str(status.get("created_at") or ""), meeting_dir))
    found.sort(key=lambda item: item[0])
    return [meeting_dir for _created, meeting_dir in found]


def _bump_regen_count(meeting_dir: Path) -> None:
    """Счётчик попыток авто-перегенерации (до регенерации, чтобы не зациклиться)."""
    status = _read_status(meeting_dir)
    status["report_regen_count"] = int(status.get("report_regen_count") or 0) + 1
    (meeting_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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
    regen = _needs_regen_meetings(meetings_root, blocked)
    regen_set = {m.resolve() for m in regen}
    queue = blocked + regen
    if args.limit:
        queue = queue[: args.limit]

    if not queue:
        print("Nothing to do: no blocked meetings and no stale reports to regenerate.")
        return 0

    print(
        f"{len(queue)} meeting(s) to process "
        f"({len(blocked)} blocked, {len(regen)} stale report(s); oldest first):"
    )
    for meeting_dir in queue:
        tag = "regen" if meeting_dir.resolve() in regen_set else "blocked"
        print(f"  - [{tag}] {meeting_dir.name}")
    if args.dry_run:
        return 0

    done = 0
    for meeting_dir in queue:
        meeting_id = meeting_dir.name
        is_regen = meeting_dir.resolve() in regen_set
        print(f"\n== regenerating report for {meeting_id} (no re-ASR) ...", flush=True)
        if is_regen:
            # Счётчик попыток до регенерации: упорно сбойную встречу не крутим вечно.
            _bump_regen_count(meeting_dir)
        try:
            # Только отчёт из готового транскрипта — ASR не перезапускаем.
            artifacts = regenerate_report_from_transcript(meeting_dir)
        except (ReportQuotaError, ReportUnavailableError) as exc:
            remaining = len(queue) - done
            reason = (
                "STILL out of quota"
                if isinstance(exc, ReportQuotaError)
                else "AI STILL unavailable"
            )
            print(
                f"   {reason}. Stopping. {remaining} meeting(s) left; "
                "retry when AI is back.",
                flush=True,
            )
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED (left as-is): {exc}", flush=True)
            continue
        # Финализация (манифест + синк памяти) — через штатный сервисный путь.
        service._write_status(
            meeting_dir,
            "completed",
            phase="completed",
            progress=1.0,
            message="Report regenerated",
            result={
                "report_generator": artifacts.generated_by,
                "report_status": artifacts.status,
            },
        )
        print(f"   -> completed ({artifacts.generated_by})", flush=True)
        done += 1

    print(f"\nDone. Reprocessed {done}/{len(queue)} meeting(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
