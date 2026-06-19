"""Тесты на чистые helper-функции захардённого дрейнера отчётов.

Тестируемый модуль: scripts/drain_reports_via_claude.py. Покрываются только
deterministic helper'ы, не требующие сети/launchd/claude/субпроцессов:
_needs_disk_upload, _mark_disk_uploaded, _health_status, _attempt_count,
_bump_attempt и файловый лок _acquire_lock (fcntl.flock).

Все записи — только во tmp_path.
"""

import importlib.util
import json
from pathlib import Path

# scripts/ не является пакетом (нет __init__.py), поэтому грузим модуль по пути.
# Импорт сам выполняет sys.path.insert(repo_root/scripts) из тела модуля и
# подтягивает reprocess_blocked + fcntl без сети/ключей.
_DRAIN_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "drain_reports_via_claude.py"
)
_spec = importlib.util.spec_from_file_location("drain_reports_via_claude", _DRAIN_PATH)
drain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drain)


def _make_meeting(tmp_path: Path, status: dict | None = None) -> Path:
    """Создаёт каталог встречи с artifacts/ и (опционально) status.json."""
    meeting_dir = tmp_path / "meeting-001"
    (meeting_dir / "artifacts").mkdir(parents=True)
    if status is not None:
        (meeting_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return meeting_dir


def _read_status_file(meeting_dir: Path) -> dict:
    return json.loads((meeting_dir / "status.json").read_text(encoding="utf-8"))


# --- классификация исхода генерации: ok | limit | fail -----------------------
# Ключевое для «кончились лимиты»: НЕТ файла -> "limit" (транзиентно, без штрафа),
# файл есть но кривой -> "fail" (детерминированно, под кап), валидный -> "ok".


def test_classify_valid_payload_is_ok():
    assert drain._classify_generation(payload_valid=True, payload_exists=True) == "ok"


def test_classify_missing_payload_is_limit():
    # claude не отработал (лимит/авторизация/сеть) — файла нет вообще.
    assert (
        drain._classify_generation(payload_valid=False, payload_exists=False) == "limit"
    )


def test_classify_present_but_invalid_is_fail():
    # claude отработал, но выдал брак — это детерминированно, идёт под кап.
    assert (
        drain._classify_generation(payload_valid=False, payload_exists=True) == "fail"
    )


def test_payload_is_valid_variants(tmp_path: Path):
    missing = tmp_path / "nope.json"
    assert drain._payload_is_valid(missing) is False

    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert drain._payload_is_valid(broken) is False

    empty_sections = tmp_path / "empty.json"
    empty_sections.write_text(json.dumps({"adaptive_sections": []}), encoding="utf-8")
    assert drain._payload_is_valid(empty_sections) is False

    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"adaptive_sections": [{"title": "x", "items": []}]}),
        encoding="utf-8",
    )
    assert drain._payload_is_valid(good) is True


# --- _needs_disk_upload ---------------------------------------------------


def test_needs_disk_upload_true_when_report_present_and_not_uploaded(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})
    (meeting_dir / "artifacts" / "report.md").write_text("# report\n", encoding="utf-8")

    assert drain._needs_disk_upload(meeting_dir) is True


def test_needs_disk_upload_false_after_uploaded_flag(tmp_path):
    meeting_dir = _make_meeting(
        tmp_path, status={"meeting_id": "m-1", "report_disk_uploaded": True}
    )
    (meeting_dir / "artifacts" / "report.md").write_text("# report\n", encoding="utf-8")

    assert drain._needs_disk_upload(meeting_dir) is False


def test_needs_disk_upload_false_without_report_md(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})
    # report.md отсутствует — заливать нечего.

    assert drain._needs_disk_upload(meeting_dir) is False


# --- _mark_disk_uploaded --------------------------------------------------


def test_mark_disk_uploaded_sets_flag_and_preserves_fields(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "keep-me"})
    (meeting_dir / "artifacts" / "report.md").write_text("# report\n", encoding="utf-8")
    assert drain._needs_disk_upload(meeting_dir) is True

    drain._mark_disk_uploaded(meeting_dir)

    status = _read_status_file(meeting_dir)
    assert status["report_disk_uploaded"] is True
    # Остальные поля не потёрты.
    assert status["meeting_id"] == "keep-me"
    # Идемпотентно: теперь заливка больше не нужна.
    assert drain._needs_disk_upload(meeting_dir) is False


# --- _health_status -------------------------------------------------------


def test_health_status_reads_failed(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    (meeting_dir / "artifacts" / "report_health.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )

    assert drain._health_status(meeting_dir) == "failed"


def test_health_status_reads_degraded(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    (meeting_dir / "artifacts" / "report_health.json").write_text(
        json.dumps({"status": "degraded"}), encoding="utf-8"
    )

    assert drain._health_status(meeting_dir) == "degraded"


def test_health_status_empty_when_missing(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    # report_health.json отсутствует.

    assert drain._health_status(meeting_dir) == ""


def test_health_status_empty_on_broken_json(tmp_path):
    meeting_dir = _make_meeting(tmp_path)
    (meeting_dir / "artifacts" / "report_health.json").write_text(
        "{ not valid json", encoding="utf-8"
    )

    # Не падает, возвращает "".
    assert drain._health_status(meeting_dir) == ""


# --- _attempt_count / _bump_attempt --------------------------------------


def test_attempt_count_starts_at_zero(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})

    assert drain._attempt_count(meeting_dir) == 0


def test_bump_attempt_increments_and_persists(tmp_path):
    meeting_dir = _make_meeting(tmp_path, status={"meeting_id": "m-1"})

    assert drain._bump_attempt(meeting_dir) == 1
    assert drain._attempt_count(meeting_dir) == 1

    assert drain._bump_attempt(meeting_dir) == 2
    assert drain._attempt_count(meeting_dir) == 2

    # Значение персистится в status.json.
    status = _read_status_file(meeting_dir)
    assert status["report_attempt_count"] == 2
    # Прочие поля не потёрты.
    assert status["meeting_id"] == "m-1"


# --- _acquire_lock (fcntl.flock) -----------------------------------------


def test_acquire_lock_is_exclusive_and_releases(tmp_path):
    lock_path = tmp_path / "drain.lock"

    fd1 = drain._acquire_lock(lock_path)
    fd2 = None
    fd3 = None
    try:
        # Первый вызов берёт лок.
        assert fd1 is not None

        # Второй вызов на тот же путь (пока первый держится) — None.
        fd2 = drain._acquire_lock(lock_path)
        assert fd2 is None

        # Освобождаем первый лок.
        fd1.close()
        fd1 = None

        # Теперь лок снова берётся.
        fd3 = drain._acquire_lock(lock_path)
        assert fd3 is not None
    finally:
        for fd in (fd1, fd2, fd3):
            if fd is not None:
                fd.close()
