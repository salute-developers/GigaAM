"""Round-trip тест финализатора суб-агентного отчёта (scripts/agent_report.py).

prepare: transcript.json -> agent_input.json с каноническими segment_id.
finalize: ai_payload.json (ответ агента) -> отрендеренные артефакты, при этом
несуществующие segment_id в citations вырезаются фильтром, а title из payload
применяется к отчёту.

Без сети: finalize не зовёт LLM-API (это и есть смысл суб-агентного пути).
Все записи — только во tmp_path.
"""

import argparse
import importlib.util
import json
from pathlib import Path

# scripts/ не является пакетом (нет __init__.py), поэтому грузим модуль по пути.
# Импорт сам выполняет sys.path.insert(repo_root) из тела agent_report.py.
_AGENT_REPORT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "agent_report.py"
_spec = importlib.util.spec_from_file_location("agent_report", _AGENT_REPORT_PATH)
agent_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_report)


def _write_meeting(tmp_path: Path) -> Path:
    """Создаёт структуру встречи: artifacts/transcript.json + status.json."""
    meeting_dir = tmp_path / "meeting-001"
    artifacts = meeting_dir / "artifacts"
    artifacts.mkdir(parents=True)

    # Сегменты без segment_id: id присваиваются в build_local_report (S0001...).
    transcript = [
        {
            "start": 0.0,
            "end": 12.0,
            "speaker": "Илья",
            "text": "Договорились сделать PDF отчеты с полным транскриптом.",
        },
        {
            "start": 12.5,
            "end": 25.0,
            "speaker": "Анна",
            "text": "Нужно сохранить ссылки на таймкоды и сегменты.",
        },
        {
            "start": 26.0,
            "end": 39.0,
            "speaker": "Илья",
            "text": "Как будем проверять что детали не потерялись?",
        },
        {
            "start": 40.0,
            "end": 54.0,
            "speaker": "Анна",
            "text": "Сделаю прототип шаблона и проверю на длинной встрече.",
        },
    ]
    (artifacts / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (meeting_dir / "status.json").write_text(
        json.dumps(
            {"metadata": {"source_filename": "planning-call.m4a"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return meeting_dir


def test_prepare_writes_agent_input_with_canonical_segment_ids(tmp_path):
    meeting_dir = _write_meeting(tmp_path)
    work_dir = tmp_path / "work"

    rc = agent_report.cmd_prepare(
        argparse.Namespace(meeting_dir=str(meeting_dir), work_dir=str(work_dir))
    )

    assert rc == 0
    agent_input_path = work_dir / "agent_input.json"
    assert agent_input_path.exists()

    payload = json.loads(agent_input_path.read_text(encoding="utf-8"))
    segment_ids = [seg["segment_id"] for seg in payload["segments"]]
    assert segment_ids == ["S0001", "S0002", "S0003", "S0004"]
    assert payload["segment_count"] == 4
    assert payload["source_filename"] == "planning-call.m4a"


def test_finalize_round_trip_filters_unknown_citations(tmp_path):
    meeting_dir = _write_meeting(tmp_path)
    work_dir = tmp_path / "work"
    out_dir = meeting_dir / "artifacts"

    # prepare даёт нам реальные segment_id для построения ai_payload.
    agent_report.cmd_prepare(
        argparse.Namespace(meeting_dir=str(meeting_dir), work_dir=str(work_dir))
    )
    agent_input = json.loads(
        (work_dir / "agent_input.json").read_text(encoding="utf-8")
    )
    real_ids = [seg["segment_id"] for seg in agent_input["segments"]]
    real_id = real_ids[0]
    fake_id = "S9999"
    assert fake_id not in real_ids

    ai_payload = {
        "title": "Планирование PDF-отчётов",
        "profile": {
            "kind": "project_sync",
            "label": "Проектная встреча",
            "confidence": 0.9,
            "rationale": "Тестовый ответ агента.",
        },
        "overview": "Команда согласовала формат PDF-отчётов и проверку полноты.",
        "adaptive_sections": [
            {
                "kind": "decisions",
                "title": "Решения",
                "purpose": "Зафиксировать договорённости.",
                "summary": "Договорились о формате отчётов.",
                "items": [
                    {
                        "title": "Формат отчётов",
                        "text": "Делаем PDF-отчёты с полным транскриптом.",
                        # Реальный id + несуществующий S9999 (должен быть вырезан).
                        "citations": [real_id, fake_id],
                    }
                ],
                "citations": [real_id, fake_id],
                "accent": "blue",
            }
        ],
        "timeline": [],
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "risks": [],
        "notable_quotes": [],
    }
    (work_dir / "ai_payload.json").write_text(
        json.dumps(ai_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rc = agent_report.cmd_finalize(
        argparse.Namespace(
            meeting_dir=str(meeting_dir),
            work_dir=str(work_dir),
            out_dir=str(out_dir),
            pdf=False,
        )
    )

    assert rc == 0

    # Артефакты на месте.
    for name in ("report.md", "report.json", "coverage.json", "report_health.json"):
        assert (out_dir / name).exists(), f"{name} was not written"

    report_json_text = (out_dir / "report.json").read_text(encoding="utf-8")
    coverage_text = (out_dir / "coverage.json").read_text(encoding="utf-8")

    # Фильтр цитат сработал: несуществующий id не утёк в артефакты.
    assert fake_id not in report_json_text
    assert fake_id not in coverage_text
    # Реальный id при этом сохранён.
    assert real_id in report_json_text

    # Health собрался и не "failed".
    health = json.loads((out_dir / "report_health.json").read_text(encoding="utf-8"))
    assert health["status"] in ("ok", "degraded")

    # Title из payload применился к отчёту.
    report_json = json.loads(report_json_text)
    rendered = json.dumps(report_json, ensure_ascii=False)
    assert "Планирование PDF-отчётов" in rendered
    # И в Markdown заголовок тоже подхватился.
    assert "Планирование PDF-отчётов" in (out_dir / "report.md").read_text(
        encoding="utf-8"
    )
