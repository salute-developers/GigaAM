"""Тесты на чистую логику health-check (blocked_too_long, run_checks).

scripts/ не пакет — грузим модуль по пути (как в test_drain_reports.py).
Сетевые проверки (healthz/claude) монкипатчим, время инжектим — без сети/субпроцессов.
"""

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

_HC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pipeline_healthcheck.py"
_spec = importlib.util.spec_from_file_location("pipeline_healthcheck", _HC_PATH)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)

NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)


def _meeting(root: Path, name: str, status: str, updated_at: str | None) -> None:
    d = root / name
    d.mkdir(parents=True)
    payload: dict = {"status": status}
    if updated_at is not None:
        payload["updated_at"] = updated_at
    (d / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_blocked_too_long_flags_only_old_blocked(tmp_path: Path):
    root = tmp_path / "meetings"
    root.mkdir()
    # свежий blocked (10 мин назад) — не застрял
    _meeting(root, "fresh", "blocked_on_ai", (NOW - timedelta(minutes=10)).isoformat())
    # старый blocked (5 ч назад) — застрял
    _meeting(root, "stale", "blocked_on_ai", (NOW - timedelta(hours=5)).isoformat())
    # completed старый — не считается
    _meeting(root, "done", "completed", (NOW - timedelta(hours=9)).isoformat())

    stuck = hc.blocked_too_long(root, max_hours=3, now=NOW)
    assert stuck == ["stale"]


def test_blocked_without_timestamp_is_flagged(tmp_path: Path):
    root = tmp_path / "meetings"
    root.mkdir()
    _meeting(root, "no_ts", "blocked_on_quota", None)
    assert hc.blocked_too_long(root, max_hours=3, now=NOW) == ["no_ts"]


def test_run_checks_fails_when_stuck(tmp_path, monkeypatch):
    root = tmp_path / "meetings"
    root.mkdir()
    _meeting(root, "stale", "blocked_on_ai", (NOW - timedelta(hours=5)).isoformat())
    monkeypatch.setattr(hc, "_meetings_root", lambda: root)
    monkeypatch.setattr(hc, "check_healthz", lambda port: (True, "healthz ok"))
    monkeypatch.setattr(hc, "check_claude", lambda: (True, "claude 2.1.169"))
    monkeypatch.delenv("MAC_TRANSCRIBER_HEALTHCHECK_STUCK_HOURS", raising=False)

    ok, lines = hc.run_checks(NOW)
    assert ok is False
    assert any("stuck blocked" in ln for ln in lines)


def test_run_checks_ok_when_healthy(tmp_path, monkeypatch):
    root = tmp_path / "meetings"
    root.mkdir()
    _meeting(root, "done", "completed", NOW.isoformat())
    monkeypatch.setattr(hc, "_meetings_root", lambda: root)
    monkeypatch.setattr(hc, "check_healthz", lambda port: (True, "healthz ok"))
    monkeypatch.setattr(hc, "check_claude", lambda: (True, "claude 2.1.169"))

    ok, lines = hc.run_checks(NOW)
    assert ok is True
    assert any("queue ok" in ln for ln in lines)
