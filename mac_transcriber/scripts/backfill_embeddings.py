#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mac_transcriber import memory_db


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = load_env_files(args.env_file)
    database_url = memory_db.database_url_from_env(env)
    api_key = memory_db.openai_api_key_from_env(env)
    if not database_url:
        raise SystemExit("MAC_TRANSCRIBER_DATABASE_URL is not configured")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not configured")

    stats = memory_db.upsert_meeting_embeddings(
        database_url,
        api_key=api_key,
        meeting_id=args.meeting_id,
        model=args.model,
        batch_size=args.batch_size,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill OpenAI embeddings into Postgres meeting memory."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        action="append",
        default=[Path(".env.local")],
        help="Env file to load; may be repeated.",
    )
    parser.add_argument(
        "--meeting-id",
        help="Only rebuild embeddings for one meeting_id. Defaults to all meetings.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "MAC_TRANSCRIBER_EMBEDDING_MODEL",
            memory_db.DEFAULT_EMBEDDING_MODEL,
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(
            os.environ.get(
                "MAC_TRANSCRIBER_EMBEDDING_BATCH_SIZE",
                str(memory_db.DEFAULT_EMBEDDING_BATCH_SIZE),
            )
        ),
    )
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


if __name__ == "__main__":
    raise SystemExit(main())
