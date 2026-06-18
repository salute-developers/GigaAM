#!/usr/bin/env python3
"""Заливает в память факты, сгенерированные Claude-субагентами (без OpenAI).

Для встреч с деградировавшим (local) отчётом субагент читает транскрипт и пишет чистые
факты в ``.local/regen/<meeting_id>.json`` (decisions/tasks/questions/risks). Этот скрипт
встраивает их: обновляет ``report.json``, ставит ``report_health.json`` с
``generated_by=claude-...`` (чтобы гейт памяти доверял), и синкает память БЕЗ эмбеддингов
(факты ищутся токенным поиском → платный API не нужен).

Запуск из рабочего .venv::

    .venv/bin/python mac_transcriber/scripts/regen_facts.py --env-file .env.local --all
    .venv/bin/python mac_transcriber/scripts/regen_facts.py --env-file .env.local --meeting <id>
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

from mac_transcriber import memory_db  # noqa: E402

REGEN_DIR = REPO_ROOT / ".local" / "regen"
GENERATED_BY = "claude-opus-4-8/facts-regen"
FACT_KEYS = ("decisions", "tasks", "questions", "risks")


def _meeting_root() -> Path:
    """Корень с записями встреч (как в service.py), из MAC_TRANSCRIBER_ROOT."""
    return Path(
        os.environ.get("MAC_TRANSCRIBER_ROOT", ".local/mac_transcriber")
    ).expanduser()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.env_file:
        _load_env(args.env_file)
    database_url = memory_db.database_url_from_env()
    if not database_url:
        raise SystemExit("MAC_TRANSCRIBER_DATABASE_URL is not configured")

    if args.meeting:
        targets = [args.meeting]
    else:
        targets = sorted(p.stem for p in REGEN_DIR.glob("*.json"))
    if not targets:
        raise SystemExit(f"No fact files in {REGEN_DIR}")

    total_facts = 0
    for meeting_id in targets:
        facts_path = REGEN_DIR / f"{meeting_id}.json"
        if not facts_path.exists():
            print(f"  SKIP {meeting_id}: нет {facts_path.name}", flush=True)
            continue
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        n = integrate(database_url, meeting_id, facts, dry_run=args.dry_run)
        total_facts += n
        mode = "DRY" if args.dry_run else "OK "
        print(f"  {mode} {meeting_id[:34]:34} facts={n}", flush=True)
    print(f"\nИтого фактов {'было бы' if args.dry_run else 'залито'}: {total_facts}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--meeting", help="Один meeting_id; иначе все из .local/regen/."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def integrate(database_url: str, meeting_id: str, facts: dict, *, dry_run: bool) -> int:
    meeting_dir = _meeting_root() / "meetings" / meeting_id
    artifacts = meeting_dir / "artifacts"
    report_path = artifacts / "report.json"
    if not artifacts.exists():
        print(f"  SKIP {meeting_id}: нет artifacts", flush=True)
        return 0

    counts = {key: facts.get(key) or [] for key in FACT_KEYS}
    n_facts = sum(len(v) for v in counts.values())
    if dry_run:
        return n_facts

    # Обновляем report.json: сохраняем прочие поля, заменяем фактовые списки.
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    if not isinstance(report, dict):
        report = {}
    for key in FACT_KEYS:
        report[key] = counts[key]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (artifacts / "report_health.json").write_text(
        json.dumps(
            {"status": "ok", "generated_by": GENERATED_BY, "alerts": []},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Синк памяти БЕЗ эмбеддингов: upsert_meeting_memory не трогает embedding_chunks.
    manifest_path = artifacts / "manifest.json"
    memory_db.upsert_meeting_memory(database_url, meeting_dir, manifest_path)
    return n_facts


def _load_env(path: Path) -> None:

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


if __name__ == "__main__":
    raise SystemExit(main())
