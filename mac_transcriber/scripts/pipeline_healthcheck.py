#!/usr/bin/env python3
"""Health-check пайплайна mac_transcriber для внешнего монитора (dead-man's-switch).

Запускается launchd'ом каждые ~10 мин. Проверяет живость пайплайна и шлёт
heartbeat на внешний монитор (Uptime Kuma / Healthchecks / ntfy — что угодно).
Главная идея: если Mac уснул / джоб умер / claude разлогинен — пинг НЕ уходит,
и монитор бьёт тревогу по grace-периоду. При явных проблемах шлём down с причиной.

Проверки:
  1. транскрайбер /healthz отвечает {"status":"ok"};
  2. нет встреч, застрявших в blocked_* дольше N часов (claude сломан/лимит/разлогин/кап);
  3. claude CLI отвечает на --version (нужен дренажу).

ENV (в .env.local):
  MAC_TRANSCRIBER_HEALTHCHECK_URL       push-URL монитора (без него — no-op).
  MAC_TRANSCRIBER_HEALTHCHECK_FAIL_URL  опц. URL для down (Healthchecks: <URL>/fail).
                                        Если не задан — шлём на основной URL ?status=down.
  MAC_TRANSCRIBER_HEALTHCHECK_STUCK_HOURS  порог «застряло» (по умолч. 3).
  MAC_TRANSCRIBER_PORT                  порт транскрайбера (по умолч. 18003).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKED = ("blocked_on_ai", "blocked_on_quota")


def _load_env_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _meetings_root() -> Path:
    root = os.environ.get("MAC_TRANSCRIBER_ROOT") or str(
        REPO_ROOT / ".local" / "mac_transcriber"
    )
    return Path(root) / "meetings"


def _parse_dt(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def blocked_too_long(meetings_root: Path, max_hours: float, now: datetime) -> list[str]:
    """ID встреч, застрявших в blocked_* дольше max_hours (по updated_at|created_at)."""
    stuck: list[str] = []
    for status_path in meetings_root.glob("*/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") not in BLOCKED:
            continue
        stamp = _parse_dt(
            str(status.get("updated_at") or status.get("created_at") or "")
        )
        if stamp is None:
            stuck.append(status_path.parent.name)  # нет метки — считаем подозрительной
            continue
        if (now - stamp).total_seconds() > max_hours * 3600:
            stuck.append(status_path.parent.name)
    return stuck


def check_healthz(port: int) -> tuple[bool, str]:
    url = f"http://127.0.0.1:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            body = resp.read().decode("utf-8", "replace")
        ok = resp.status == 200 and '"ok"' in body
        return ok, "healthz ok" if ok else f"healthz bad: {resp.status} {body[:60]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"healthz unreachable: {exc}"[:80]


def check_claude() -> tuple[bool, str]:
    claude = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not claude:
        return False, "claude not on PATH"
    try:
        res = subprocess.run(
            [claude, "--version"], capture_output=True, text=True, timeout=20
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"claude --version error: {exc}"[:80]
    if res.returncode != 0:
        return False, f"claude --version rc={res.returncode}"
    return True, f"claude {res.stdout.strip()[:30]}"


def run_checks(now: datetime) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    h_ok, h_msg = check_healthz(int(os.environ.get("MAC_TRANSCRIBER_PORT", "18003")))
    lines.append(h_msg)
    ok = ok and h_ok

    max_hours = float(os.environ.get("MAC_TRANSCRIBER_HEALTHCHECK_STUCK_HOURS", "3"))
    stuck = blocked_too_long(_meetings_root(), max_hours, now)
    if stuck:
        ok = False
        lines.append(f"stuck blocked >{max_hours:g}h: {len(stuck)} ({stuck[0]})")
    else:
        lines.append("queue ok (no long-blocked)")

    c_ok, c_msg = check_claude()
    lines.append(c_msg)
    ok = ok and c_ok

    return ok, lines


def _ping(url: str, params: dict[str, str]) -> None:
    full = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(full, timeout=10) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"healthcheck ping failed: {exc}", file=sys.stderr)


def _telegram(token: str, chat_id: str, text: str) -> None:
    """Алерт в Telegram (Bot API). Молчит, если токен/чат не заданы."""
    if not (token and chat_id):
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"telegram alert failed: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _load_env_files([REPO_ROOT / ".env.local"])
    url = os.environ.get("MAC_TRANSCRIBER_HEALTHCHECK_URL", "").strip()
    tg_token = os.environ.get("MAC_TRANSCRIBER_TELEGRAM_TOKEN", "").strip()
    tg_chat = os.environ.get("MAC_TRANSCRIBER_TELEGRAM_CHAT_ID", "").strip()

    now = datetime.now(UTC)
    ok, lines = run_checks(now)
    summary = "; ".join(lines)
    print(f"[{'OK' if ok else 'FAIL'}] {summary}")

    # Telegram — только алерт о проблеме (то, что VPS-watchdog не видит:
    # застрявшая очередь / сломанный claude). Не heartbeat.
    if not ok and tg_token and tg_chat:
        _telegram(tg_token, tg_chat, f"🔴 Mac-пайплайн отчётов: {summary[:350]}")

    # Heartbeat-URL (Uptime Kuma / Healthchecks): up при здоровье, down при сбое.
    if url:
        if ok:
            _ping(url, {"status": "up", "msg": "OK"})
        else:
            fail_url = os.environ.get(
                "MAC_TRANSCRIBER_HEALTHCHECK_FAIL_URL", ""
            ).strip()
            if fail_url:
                _ping(fail_url, {"msg": summary[:300]})
            else:
                _ping(url, {"status": "down", "msg": summary[:300]})

    if not url and not (tg_token and tg_chat):
        print("монитор не настроен (нет URL и Telegram) — no-op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
