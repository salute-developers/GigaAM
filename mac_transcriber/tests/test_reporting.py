import json
import importlib.util
from pathlib import Path

import pytest

from mac_transcriber.asr import Segment
from mac_transcriber import asr, reporting
from mac_transcriber.reporting import (
    build_report,
    build_local_report,
    build_protocol_data,
    render_report_html,
    render_report_markdown,
    render_report_typst,
    validate_report_citations,
    write_report_artifacts,
)


def _segment(speaker: str, start: float, end: float, text: str) -> Segment:
    return Segment(
        speaker=speaker,
        track="audio.m4a",
        start=start,
        end=end,
        text=text,
    )


def test_local_report_keeps_citations_and_protocol_refs():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты с полным транскриптом."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды и сегменты."),
            _segment("Илья", 26.0, 39.0, "Как будем проверять что детали не потерялись?"),
            _segment("Анна", 40.0, 54.0, "Сделаю прототип шаблона и проверю на длинной встрече."),
        ],
    )

    validate_report_citations(report)

    assert report.segment_count == 4
    assert [record.segment_id for record in report.transcript] == [
        "S0001",
        "S0002",
        "S0003",
        "S0004",
    ]
    assert report.decisions[0].citations == ["S0001"]
    assert report.action_items[0].owner == "Анна"
    assert report.open_questions[0].citations == ["S0003"]

    markdown = render_report_markdown(report)

    assert "# Planning Call" in markdown
    assert "Transcript: `transcript.md`" in markdown
    assert "Coverage: `coverage.json`" in markdown
    assert "`S0001 · 00:00`" in markdown
    assert "## Полный транскрипт" not in markdown


def test_local_report_accepts_saved_transcript_dicts_without_segment_ids():
    report = build_local_report(
        meeting_id="legacy-transcript",
        title="Legacy Transcript",
        source_filename="legacy.json",
        model_name="v3_e2e_rnnt",
        segments=[
            {
                "start": 12.5,
                "end": 15.0,
                "speaker": "Анна",
                "text": "Нужно сохранить совместимость с прежними transcript.json.",
            },
            {
                "start": 1.0,
                "end": 4.0,
                "speaker": "Илья",
                "text": "Договорились пересобрать отчет из сохраненного файла.",
            },
        ],
    )

    assert [record.segment_id for record in report.transcript] == ["S0001", "S0002"]
    assert [record.speaker for record in report.transcript] == ["Илья", "Анна"]
    assert report.decisions[0].citations == ["S0001"]
    assert report.action_items[0].citations == ["S0002"]


def test_local_report_adapts_structure_for_lecture_content():
    report = build_local_report(
        meeting_id="lecture",
        title="Лекция про продуктовую аналитику",
        source_filename="lecture.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Лектор", 0.0, 20.0, "Сегодня разберем продуктовую аналитику и ключевые метрики."),
            _segment("Лектор", 21.0, 45.0, "Определение retention: это доля пользователей которые вернулись в продукт."),
            _segment("Лектор", 46.0, 70.0, "Например, если клиент вернулся через неделю, мы считаем недельное удержание."),
            _segment("Студент", 71.0, 82.0, "А чем retention отличается от engagement?"),
        ],
    )

    assert report.profile.kind == "lecture"
    assert report.profile.label == "Лекция / обучение"
    assert {section.kind for section in report.adaptive_sections} >= {
        "lecture_notes",
        "terms",
        "examples",
    }
    assert {entry.segment_id for entry in report.coverage} == {
        record.segment_id for record in report.transcript
    }
    assert all(entry.status in {"covered", "supporting", "low_signal"} for entry in report.coverage)

    markdown = render_report_markdown(report)

    assert "## TL;DR" in markdown
    assert "## KPI" in markdown
    assert "retention" in markdown
    assert "coverage.json" in markdown
    assert "## Coverage audit" not in markdown


def test_local_report_adapts_structure_for_project_sync_content():
    report = build_local_report(
        meeting_id="sync",
        title="Проектный синк",
        source_filename="sync.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("PM", 0.0, 12.0, "Договорились запустить PDF отчеты в MVP на этой неделе."),
            _segment("Dev", 13.0, 25.0, "Нужно добавить coverage audit и проверить длинные записи."),
            _segment("PM", 26.0, 36.0, "Риск в том что отчеты будут слишком общими для разных тем."),
            _segment("Dev", 37.0, 50.0, "Следующий шаг: сделать адаптивные секции по типу транскрипта."),
        ],
    )

    assert report.profile.kind == "project_sync"
    assert report.profile.label == "Проектная встреча"
    assert {section.kind for section in report.adaptive_sections} >= {
        "decisions",
        "actions",
        "risks",
    }
    assert report.coverage[0].section_titles


def test_local_report_caps_and_deduplicates_fallback_items():
    segments = [
        _segment("Илья", float(index), float(index) + 0.5, "Открытый вопрос: нужно обновить доску задач?")
        for index in range(30)
    ]

    report = build_local_report(
        meeting_id="fallback",
        title="Fallback",
        source_filename="fallback.m4a",
        model_name="v3_e2e_rnnt",
        segments=segments,
    )

    assert len(report.open_questions) == 1
    assert len(report.action_items) == 1
    assert report.open_questions[0].citations == [f"S{index:04d}" for index in range(1, 31)]
    assert report.action_items[0].citations == [f"S{index:04d}" for index in range(1, 31)]


def test_local_report_filters_conversational_task_and_question_noise():
    report = build_local_report(
        meeting_id="fallback-noise",
        title="Fallback Noise",
        source_filename="fallback-noise.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 1.0, "А?"),
            _segment("Илья", 2.0, 3.0, "Неправильно. Надо говорит Вадим «да»."),
            _segment("Вадим", 4.0, 5.0, "Не, это даже не надо сдвигать."),
            _segment("Илья", 6.0, 7.0, "Нужно отметить дату."),
            _segment("Вадим", 8.0, 9.0, "Я добавлю задачу на прототип и реализацию."),
            _segment("Илья", 10.0, 11.0, "Как будем проверять результат?"),
        ],
    )

    assert [item.text for item in report.action_items] == [
        "Нужно отметить дату.",
        "Я добавлю задачу на прототип и реализацию.",
    ]
    assert [item.text for item in report.open_questions] == ["Как будем проверять результат?"]


def test_report_validation_rejects_uncited_segment_ids():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
        ],
    )
    report.decisions[0].citations = ["S9999"]

    with pytest.raises(ValueError, match="unknown segment id"):
        validate_report_citations(report)


def test_report_validation_rejects_uncited_adaptive_sections():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
        ],
    )
    report.adaptive_sections[0].items[0].citations = ["S9999"]

    with pytest.raises(ValueError, match="adaptive_sections"):
        validate_report_citations(report)


def test_report_validation_rejects_missing_coverage_entries():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
    )
    report.coverage = report.coverage[:1]

    with pytest.raises(ValueError, match="coverage is missing segment ids"):
        validate_report_citations(report)


def test_ai_report_falls_back_when_coverage_is_incomplete(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тестовый ответ.",
                    },
                    "overview": "AI summary",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Решения"],
                            "rationale": "Тест.",
                        }
                    ],
                    "timeline": [],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    assert report.generated_by == "local"
    assert any("AI report fallback" in warning for warning in report.warnings)
    assert {entry.segment_id for entry in report.coverage} == {"S0001", "S0002"}


def test_direct_ai_report_receives_prior_context_pack(monkeypatch):
    request_payloads = []

    def fake_openai_response(*, payload, api_key):
        request_payloads.append(json.loads(payload["input"][1]["content"]))
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тестовый ответ.",
                    },
                    "overview": "AI summary with memory.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Контекст"],
                            "rationale": "Тест.",
                        }
                    ],
                    "timeline": [],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    context_pack = {
        "facts": [
            {
                "meeting_id": "old-call",
                "fact_type": "decision",
                "text": "Ранее решили хранить транскрипты локально.",
            }
        ],
        "segments": [],
    }
    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
        context_pack=context_pack,
    )

    assert report.generated_by == "gpt-5.5"
    assert request_payloads[0]["prior_context"] == context_pack


def test_ai_report_uses_chunked_synthesis_for_long_transcripts(monkeypatch):
    calls = []
    synthesis_payloads = []

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        calls.append(schema_name)
        request_payload = json.loads(payload["input"][1]["content"])
        if schema_name == "meeting_report_chunk":
            first = request_payload["segments"][0]
            return {
                "output_text": json.dumps(
                    {
                        "summary": f"Фрагмент про {first['text']}",
                        "key_points": [
                            {
                                "title": "Деталь",
                                "text": first["text"],
                                "citations": [first["segment_id"]],
                            }
                        ],
                        "decisions": [],
                        "action_items": [],
                        "open_questions": [],
                        "risks": [],
                        "notable_quotes": [],
                    },
                    ensure_ascii=False,
                )
            }
        synthesis_payloads.append(request_payload)
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "technical_discussion",
                        "label": "Техническое обсуждение",
                        "confidence": 0.91,
                        "rationale": "Обсуждаются технические детали.",
                    },
                    "overview": "AI собрал общий отчёт из фрагментов.",
                    "adaptive_sections": [
                        {
                            "kind": "architecture",
                            "title": "Архитектура",
                            "purpose": "Сохранить детали.",
                            "summary": "Сводка по архитектуре.",
                            "items": [
                                {
                                    "title": "База данных",
                                    "text": "Обсуждалась структура базы данных.",
                                    "citations": ["S0001"],
                                }
                            ],
                            "citations": ["S0001"],
                            "accent": "blue",
                        }
                    ],
                    "timeline": [
                        {
                            "title": "00:00-00:40",
                            "summary": "Разбор деталей.",
                            "citations": ["S0001", "S0002"],
                        }
                    ],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 2)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 2)
    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)
    context_pack = {
        "facts": [{"meeting_id": "old-call", "text": "Старый контекст."}],
        "segments": [],
    }

    report = build_report(
        meeting_id="long-call",
        title="Long Call",
        source_filename="long.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 8.0, "Обсуждаем структуру базы данных."),
            _segment("Анна", 9.0, 16.0, "Нужно сохранить связи между таблицами."),
            _segment("Илья", 17.0, 24.0, "Проверяем стратегии обслуживания."),
            _segment("Анна", 25.0, 32.0, "Фиксируем версионирование техкарт."),
            _segment("Илья", 33.0, 40.0, "Открытый вопрос по подрядчикам."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
        context_pack=context_pack,
    )

    assert calls == [
        "meeting_report_chunk",
        "meeting_report_chunk",
        "meeting_report_chunk",
        "meeting_report_synthesis",
    ]
    assert report.generated_by == "gpt-5.5"
    assert report.profile.kind == "technical_discussion"
    assert synthesis_payloads[0]["prior_context"] == context_pack
    assert {entry.segment_id for entry in report.coverage} == {
        "S0001",
        "S0002",
        "S0003",
        "S0004",
        "S0005",
    }


def test_write_report_artifacts_marks_ai_fallback_as_degraded(tmp_path, monkeypatch):
    def fake_openai_response(*, payload, api_key):
        raise reporting.ReportGenerationError("OpenAI API request failed: test disconnect")

    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)

    result = write_report_artifacts(
        output_dir=tmp_path,
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
        use_ai=True,
        make_pdf=False,
        api_key="sk-test",
    )

    health = json.loads((tmp_path / "report_health.json").read_text(encoding="utf-8"))

    assert result.status == "degraded"
    assert health["status"] == "degraded"
    assert any("AI report requested" in alert for alert in health["alerts"])
    assert any("AI report fallback" in warning for warning in health["warnings"])


def test_report_health_alerts_include_report_warning_details(tmp_path):
    report = build_local_report(
        meeting_id="long-call",
        title="Long Call",
        source_filename="long.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
    )
    report.warnings.append(
        "AI synthesis skipped: 19 chunks exceed "
        "MAC_TRANSCRIBER_AI_SYNTHESIS_CHUNK_LIMIT=6; "
        "assembled report from chunk notes."
    )
    (tmp_path / "report.json").write_text(
        json.dumps(build_protocol_data(report), ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "report.md").write_text(render_report_markdown(report), encoding="utf-8")
    (tmp_path / "report.html").write_text(render_report_html(report), encoding="utf-8")
    (tmp_path / "coverage.json").write_text("{}", encoding="utf-8")

    health = reporting.build_report_health(
        report=report,
        output_dir=tmp_path,
        requested_ai=True,
        requested_pdf=False,
        pdf_path=None,
        pdf_error=None,
    )

    assert health["status"] == "degraded"
    assert any("AI synthesis skipped" in alert for alert in health["alerts"])


def test_report_health_marks_ai_synthesis_fallback_check_failed(tmp_path):
    report = build_local_report(
        meeting_id="long-call",
        title="Long Call",
        source_filename="long.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
        ],
    )
    report.generated_by = "gpt-5.5/chunked"
    report.warnings.append(
        "AI synthesis fallback: AI synthesis failed after 3 attempt(s): "
        "OpenAI API request failed"
    )
    (tmp_path / "report.json").write_text(
        json.dumps(build_protocol_data(report), ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "report.md").write_text(render_report_markdown(report), encoding="utf-8")
    (tmp_path / "report.html").write_text(render_report_html(report), encoding="utf-8")
    (tmp_path / "coverage.json").write_text("{}", encoding="utf-8")

    health = reporting.build_report_health(
        report=report,
        output_dir=tmp_path,
        requested_ai=True,
        requested_pdf=False,
        pdf_path=None,
        pdf_error=None,
    )

    checks = {check["name"]: check for check in health["checks"]}
    assert checks["ai_synthesis_succeeded"]["ok"] is False
    assert "AI synthesis fallback" in checks["ai_synthesis_succeeded"]["details"]
    assert len(health["alerts"]) == len(set(health["alerts"]))


def test_chunked_ai_uses_chunk_notes_when_synthesis_fails(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_synthesis":
            raise reporting.ReportGenerationError("AI synthesis failed after 1 attempt(s): test disconnect")
        request_payload = json.loads(payload["input"][1]["content"])
        first = request_payload["segments"][0]
        return {
            "output_text": json.dumps(
                {
                    "summary": f"Сводка блока: {first['text']}",
                    "key_points": [
                        {
                            "title": "Техническая деталь",
                            "text": first["text"],
                            "citations": [first["segment_id"]],
                        }
                    ],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 1)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 1)
    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="long-call",
        title="Long Call",
        source_filename="long.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 8.0, "Обсуждаем структуру базы данных."),
            _segment("Анна", 9.0, 16.0, "Нужно сохранить связи между таблицами."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    assert report.generated_by == "gpt-5.5/chunked"
    assert any("AI synthesis fallback" in warning for warning in report.warnings)
    assert report.adaptive_sections[0].items[0].text == "Обсуждаем структуру базы данных."
    assert {entry.segment_id for entry in report.coverage} == {"S0001", "S0002"}


def test_chunked_ai_batches_synthesis_for_many_chunk_notes(monkeypatch):
    synthesis_batch_sizes = []

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        request_payload = json.loads(payload["input"][1]["content"])
        if schema_name == "meeting_report_chunk":
            first = request_payload["segments"][0]
            return {
                "output_text": json.dumps(
                    {
                        "summary": f"Сводка блока: {first['text']}",
                        "key_points": [
                            {
                                "title": "Тема",
                                "text": first["text"],
                                "citations": [first["segment_id"]],
                            }
                        ],
                        "decisions": [
                            {
                                "title": "Решение",
                                "text": first["text"],
                                "citations": [first["segment_id"]],
                            }
                        ],
                        "action_items": [],
                        "open_questions": [],
                        "risks": [],
                        "notable_quotes": [],
                    },
                    ensure_ascii=False,
                )
            }

        synthesis_batch_sizes.append(len(request_payload["chunk_notes"]))
        first_note = request_payload["chunk_notes"][0]
        first_item = first_note["decisions"][0]
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тестовый batch synthesis.",
                    },
                    "overview": first_note["summary"],
                    "adaptive_sections": [
                        {
                            "kind": "decisions",
                            "title": "Решения",
                            "purpose": "Сохранить решения.",
                            "summary": first_note["summary"],
                            "items": [first_item],
                            "citations": first_item["citations"],
                            "accent": "green",
                        }
                    ],
                    "timeline": [],
                    "decisions": [first_item],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 1)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 1)
    monkeypatch.setattr(reporting, "AI_SYNTHESIS_CHUNK_LIMIT", 10)
    monkeypatch.setattr(reporting, "AI_SYNTHESIS_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="long-call",
        title="Long Call",
        source_filename="long.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 8.0, "Обсуждаем структуру базы данных."),
            _segment("Анна", 9.0, 16.0, "Нужно сохранить связи между таблицами."),
            _segment("Илья", 17.0, 24.0, "Проверяем стратегии обслуживания."),
            _segment("Анна", 25.0, 32.0, "Фиксируем версионирование техкарт."),
            _segment("Илья", 33.0, 40.0, "Открытый вопрос по подрядчикам."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    assert synthesis_batch_sizes == [2, 2, 1]
    assert report.generated_by == "gpt-5.5"
    assert not report.warnings
    assert len(report.decisions) == 3


def test_chunked_ai_fallback_deduplicates_repeated_items(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_synthesis":
            raise reporting.ReportGenerationError("AI synthesis failed after 1 attempt(s): test disconnect")
        request_payload = json.loads(payload["input"][1]["content"])
        first = request_payload["segments"][0]
        return {
            "output_text": json.dumps(
                {
                    "summary": f"Сводка блока: {first['text']}",
                    "key_points": [],
                    "decisions": [
                        {
                            "title": "Единое решение",
                            "text": "Договорились обновить доску задач и оценки.",
                            "citations": [first["segment_id"]],
                        }
                    ],
                    "action_items": [
                        {
                            "title": "Единая задача",
                            "text": "Обновить доску задач и оценки.",
                            "owner": "Илья",
                            "due": "",
                            "citations": [first["segment_id"]],
                        }
                    ],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 1)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 1)
    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="long-call",
        title="Long Call",
        source_filename="long.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 8.0, "Нужно обновить доску."),
            _segment("Анна", 9.0, 16.0, "Да, обновляем доску."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    assert len(report.decisions) == 1
    assert len(report.action_items) == 1
    assert report.decisions[0].citations == ["S0001", "S0002"]
    assert report.action_items[0].citations == ["S0001", "S0002"]


def test_chunked_ai_skips_synthesis_when_chunk_limit_is_exceeded(monkeypatch):
    calls = []

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        calls.append(schema_name)
        if schema_name == "meeting_report_synthesis":
            raise AssertionError("synthesis should be skipped")
        request_payload = json.loads(payload["input"][1]["content"])
        first = request_payload["segments"][0]
        return {
            "output_text": json.dumps(
                {
                    "summary": f"Сводка блока: {first['text']}",
                    "key_points": [
                        {
                            "title": "Техническая деталь",
                            "text": first["text"],
                            "citations": [first["segment_id"]],
                        }
                    ],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 1)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 1)
    monkeypatch.setattr(reporting, "AI_SYNTHESIS_CHUNK_LIMIT", 1)
    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="long-call",
        title="Long Call",
        source_filename="long.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 8.0, "Обсуждаем структуру базы данных."),
            _segment("Анна", 9.0, 16.0, "Нужно сохранить связи между таблицами."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    assert calls == ["meeting_report_chunk", "meeting_report_chunk"]
    assert report.generated_by == "gpt-5.5/chunked"
    assert any("AI synthesis skipped" in warning for warning in report.warnings)


def test_report_typst_uses_basic_report_template_and_omits_full_transcript():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
    )

    typst = render_report_typst(report)

    assert '#import "@preview/basic-report:0.5.0": *' in typst
    assert "#show: it => basic-report(" in typst
    assert 'doc-title: "Planning Call"' in typst
    assert "compact-mode: true" in typst
    assert "show-outline: false" in typst
    assert "#let memo_block" not in typst
    assert "#let status_pill" not in typst
    assert "#let metric_tile" not in typst
    assert "#let transcript_row" not in typst
    assert "Planning Call" in typst
    assert "Проектная встреча" in typst
    assert "= Сводка отчета" in typst
    assert "= Структура отчета" in typst
    assert "== Решения" in typst
    assert typst.count("== Решения") == 1
    assert "== Задачи\n" not in typst
    assert "= Coverage audit" in typst
    assert "Low signal" in typst
    assert "== Полный транскрипт" not in typst
    assert "Полный транскрипт хранится в Markdown-артефакте" in typst


def test_report_typst_wraps_long_russian_title_for_basic_report():
    report = build_local_report(
        meeting_id="lecture",
        title="Лекция про продуктовую аналитику",
        source_filename="lecture.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Лектор", 0.0, 20.0, "Сегодня разберем продуктовую аналитику и ключевые метрики."),
        ],
    )

    typst = render_report_typst(report)

    assert 'doc-title: "Лекция\\nпро продуктовую\\nаналитику"' in typst


def test_report_html_renders_protocol_without_full_transcript_or_coverage():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
    )

    html = render_report_html(report)

    assert "<!DOCTYPE html>" in html
    assert "Протокол встречи" in html
    assert "Главное" in html
    assert "Planning Call" in html
    assert "S0001 · 00:00" in html
    assert "Coverage audit" not in html
    assert "coverage.json" in html
    assert "## Полный транскрипт" not in html
    assert "@font-face" in html


def test_report_html_omits_empty_time_separator():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning-2026-06-15.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
        ],
    )

    html = render_report_html(report)

    assert "·  ·" not in html
    assert "Понедельник, 15 июня 2026</b> · длительность" in html


def test_protocol_data_extracts_human_date_and_time_from_source_filename():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="test pipeline- бд - 2026-06-12 05-57 UTC.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
        ],
    )

    protocol = build_protocol_data(report)

    assert protocol["meeting"]["date_short"] == "12.06.2026"
    assert protocol["meeting"]["date_human"] == "Пятница, 12 июня 2026"
    assert protocol["meeting"]["time_human"] == "08:57 МСК (05:57 UTC)"


def test_transcript_markdown_carries_raw_segment_id_ranges(tmp_path):
    markdown = asr.render_transcript_markdown(
        meeting_id="planning-call",
        title="Planning Call",
        model_name="v3_e2e_rnnt",
        tracks=[],
        segments=[
            _segment("Илья", 0.0, 3.0, "Первая часть."),
            _segment("Илья", 3.4, 5.0, "Вторая часть."),
            _segment("Анна", 6.0, 8.0, "Ответ."),
        ],
    )
    transcript_path = tmp_path / "transcript.md"
    transcript_path.write_text(markdown, encoding="utf-8")

    assert "[S0001-S0002] [00:00:00 - 00:00:05] **Илья:** Первая часть. Вторая часть." in markdown
    assert "[S0003] [00:00:06 - 00:00:08] **Анна:** Ответ." in markdown

    pipeline_path = Path("mac_transcriber/protocol/scripts/protocol_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("protocol_pipeline", pipeline_path)
    assert spec and spec.loader
    protocol_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol_pipeline)

    inventory = protocol_pipeline.parse_transcript(str(transcript_path))

    assert len(inventory) == 2
    assert inventory[0]["covers"] == ["S0001", "S0002"]
    assert inventory[1]["covers"] == ["S0003"]
    assert sum(len(row["covers"]) for row in inventory) == 3


def test_transcript_markdown_uses_meeting_title_in_heading():
    markdown = asr.render_transcript_markdown(
        meeting_id="f003-title-health-new",
        title="Быстрый созвон - tech-leads",
        model_name="v3_e2e_rnnt",
        tracks=[],
        segments=[],
    )

    assert markdown.startswith("# Быстрый созвон - tech-leads\n")
    assert "# Zoom transcript: f003-title-health-new" not in markdown


def test_write_report_artifacts_writes_json_markdown_html_and_typst(tmp_path):
    result = write_report_artifacts(
        output_dir=tmp_path,
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
        use_ai=False,
        make_pdf=False,
    )

    assert result.markdown_path == tmp_path / "report.md"
    assert result.html_path == tmp_path / "report.html"
    assert result.typst_path == tmp_path / "report.typ"
    assert result.health_path == tmp_path / "report_health.json"
    assert result.coverage_path == tmp_path / "coverage.json"
    assert result.slack_summary_path == tmp_path / "slack_summary.md"
    assert "*Planning Call*" in result.slack_text
    assert {item["kind"] for item in result.slack_files} == {"report_md"}
    assert result.status == "ok"
    assert result.pdf_path is None
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.typ").exists()
    assert (tmp_path / "report_health.json").exists()
    assert (tmp_path / "coverage.json").exists()
    assert (tmp_path / "slack_summary.md").exists()
    assert not (tmp_path / "slack_payload.json").exists()
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    health = json.loads((tmp_path / "report_health.json").read_text(encoding="utf-8"))
    coverage = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert payload["meeting"]["title"] == "Planning Call"
    assert payload["decisions"][0]["id"] == "D-01"
    assert payload["tasks"][0]["id"] == "T-01"
    assert coverage["coverage"][0]["segment_id"] == "S0001"
    assert health["status"] == "ok"
    assert health["checks"]


def test_slack_summary_humanizes_openai_quota_alert(tmp_path):
    summary = reporting.render_slack_summary(
        protocol={
            "meeting": {
                "title": "Quota Call",
                "date_human": "Понедельник, 15 июня 2026",
                "duration": "35:44",
            },
            "participants": ["Ilya", "Vadim"],
            "kpis": [
                {"cap": "Решений", "num": 1},
                {"cap": "Задач", "num": 2},
                {"cap": "Вопросов", "num": 3},
                {"cap": "Рисков", "num": 4},
            ],
            "tldr": ["Короткая сводка."],
        },
        health={
            "status": "degraded",
            "alerts": [
                (
                    "Report was generated with warnings: AI report fallback: "
                    "OpenAI API returned HTTP 429: {\"error\":{\"code\":\"insufficient_quota\","
                    "\"message\":\"You exceeded your current quota\"}}"
                ),
                "AI report requested but generation fell back to the local report.",
            ],
        },
        output_dir=tmp_path,
        pdf_path=None,
    )

    assert "*Статус отчёта:* требует проверки" in summary
    assert "OpenAI: превышена квота или не настроен биллинг" in summary
    assert "локальным резервным режимом" in summary
    assert "degraded" not in summary
    assert "fallback" not in summary
    assert "insufficient_quota" not in summary
    assert "You exceeded your current quota" not in summary
    assert "AI report requested but generation fell back" not in summary


def test_asr_write_artifacts_includes_rich_report_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_MODE", "local")
    monkeypatch.delenv("MAC_TRANSCRIBER_REPORT_PDF", raising=False)

    asr.write_artifacts(
        output_dir=tmp_path,
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        tracks=[],
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
        elapsed_seconds=1.25,
    )

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    report_html = (tmp_path / "report.html").read_text(encoding="utf-8")
    report_json = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    summary_json = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.typ").exists()
    assert (tmp_path / "report_health.json").exists()
    assert (tmp_path / "coverage.json").exists()
    assert (tmp_path / "slack_summary.md").exists()
    assert not (tmp_path / "slack_payload.json").exists()
    assert "Протокол встречи" in report_html
    assert "hero" in report_html
    assert "# Planning Call" in report_md
    assert "## Полный транскрипт" not in report_md
    assert report_json["meeting"]["title"] == "Planning Call"
    assert summary_json["report_status"] == "ok"
    assert summary_json["report_health"].endswith("report_health.json")
    assert "*Planning Call*" in summary_json["slack"]["text"]
    assert {item["kind"] for item in summary_json["slack"]["files"]} == {"report_md", "transcript_md"}
    assert summary_json["coverage_json"].endswith("coverage.json")


def test_asr_write_artifacts_defaults_to_ai_and_pdf_for_service_jobs(tmp_path, monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_REPORT_MODE", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_REPORT_PDF", raising=False)

    asr.write_artifacts(
        output_dir=tmp_path,
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        tracks=[],
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
        elapsed_seconds=1.25,
    )

    health = json.loads((tmp_path / "report_health.json").read_text(encoding="utf-8"))
    summary_json = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert health["requested_ai"] is True
    assert health["requested_pdf"] is True
    assert {item["kind"] for item in summary_json["slack"]["files"]}.issubset(
        {"report_pdf", "report_md", "transcript_md"}
    )
    assert all("path" not in item for item in summary_json["slack"]["files"])
