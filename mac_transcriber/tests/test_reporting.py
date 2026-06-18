import json
import importlib.util
import types
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


@pytest.fixture(autouse=True)
def _disable_baseline_upgrade_by_default(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL", " ")
    # Критик выключен по умолчанию: тесты, которые его не ждут, не должны падать,
    # даже если у разработчика в окружении выставлен флаг.
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_CRITIC_MODEL", " ")


def test_local_report_keeps_citations_and_protocol_refs():
    report = build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment(
                "Илья",
                0.0,
                12.0,
                "Договорились сделать PDF отчеты с полным транскриптом.",
            ),
            _segment(
                "Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды и сегменты."
            ),
            _segment(
                "Илья", 26.0, 39.0, "Как будем проверять что детали не потерялись?"
            ),
            _segment(
                "Анна",
                40.0,
                54.0,
                "Сделаю прототип шаблона и проверю на длинной встрече.",
            ),
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
            _segment(
                "Лектор",
                0.0,
                20.0,
                "Сегодня разберем продуктовую аналитику и ключевые метрики.",
            ),
            _segment(
                "Лектор",
                21.0,
                45.0,
                "Определение retention: это доля пользователей которые вернулись в продукт.",
            ),
            _segment(
                "Лектор",
                46.0,
                70.0,
                "Например, если клиент вернулся через неделю, мы считаем недельное удержание.",
            ),
            _segment(
                "Студент", 71.0, 82.0, "А чем retention отличается от engagement?"
            ),
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
    assert all(
        entry.status in {"covered", "supporting", "low_signal"}
        for entry in report.coverage
    )

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
            _segment(
                "PM",
                0.0,
                12.0,
                "Договорились запустить PDF отчеты в MVP на этой неделе.",
            ),
            _segment(
                "Dev",
                13.0,
                25.0,
                "Нужно добавить coverage audit и проверить длинные записи.",
            ),
            _segment(
                "PM",
                26.0,
                36.0,
                "Риск в том что отчеты будут слишком общими для разных тем.",
            ),
            _segment(
                "Dev",
                37.0,
                50.0,
                "Следующий шаг: сделать адаптивные секции по типу транскрипта.",
            ),
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
        _segment(
            "Илья",
            float(index),
            float(index) + 0.5,
            "Открытый вопрос: нужно обновить доску задач?",
        )
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
    assert report.open_questions[0].citations == [
        f"S{index:04d}" for index in range(1, 31)
    ]
    assert report.action_items[0].citations == [
        f"S{index:04d}" for index in range(1, 31)
    ]


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
    assert [item.text for item in report.open_questions] == [
        "Как будем проверять результат?"
    ]


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


def test_ai_report_recomputes_incomplete_ai_coverage(monkeypatch):
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

    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )

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

    assert report.generated_by == "gpt-5.5"
    assert not report.warnings
    assert {entry.segment_id for entry in report.coverage} == {"S0001", "S0002"}


def test_direct_ai_report_receives_prior_context_in_memory_enrichment(monkeypatch):
    request_payloads = []
    schema_names = []

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        schema_names.append(schema_name)
        request_payloads.append(json.loads(payload["input"][1]["content"]))
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {
                        "overview_addendum": "Связано с прошлым решением.",
                        "memory_sections": [],
                    },
                    ensure_ascii=False,
                )
            }
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
                        },
                        {
                            "segment_id": "S0002",
                            "status": "covered",
                            "section_titles": ["Контекст"],
                            "rationale": "Тест.",
                        },
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
    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )

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
        context_pack=context_pack,
    )

    assert report.generated_by == "gpt-5.5"
    assert schema_names == ["meeting_report", "meeting_report_memory_enrichment"]
    assert "prior_context" not in request_payloads[0]
    assert request_payloads[1]["prior_context"] == context_pack
    assert report.decisions
    assert report.action_items


def test_ai_report_uses_openrouter_fallback_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_ENV_FILE", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL", raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-primary-test",
                "OPENROUTER_API_KEY=sk-or-test",
                "MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL=openai/gpt-4o-mini",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_openai_response(*, payload, api_key):
        raise reporting.ReportGenerationError("OpenAI API request failed: quota")

    def fake_openrouter_response(*, payload, api_key, model):
        calls.append(
            {"api_key": api_key, "model": model, "payload_model": payload["model"]}
        )
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тестовый fallback.",
                    },
                    "overview": "OpenRouter fallback summary.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Решения"],
                            "rationale": "Тест.",
                        },
                        {
                            "segment_id": "S0002",
                            "status": "covered",
                            "section_titles": ["Задачи"],
                            "rationale": "Тест.",
                        },
                    ],
                    "timeline": [],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            ),
            "_provider_model": "openrouter:openai/gpt-4o-mini",
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)
    monkeypatch.setattr(
        reporting, "_post_openrouter_response", fake_openrouter_response
    )

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
    )

    assert calls == [
        {
            "api_key": "sk-or-test",
            "model": "openai/gpt-4o-mini",
            "payload_model": "gpt-5.5",
        }
    ]
    assert report.generated_by == "openrouter:openai/gpt-4o-mini"
    assert report.overview == "OpenRouter fallback summary."

    calls.clear()
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=sk-or-test",
                "MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL=openai/gpt-4o-mini",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

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
    )

    assert calls == [
        {
            "api_key": "sk-or-test",
            "model": "openai/gpt-4o-mini",
            "payload_model": "gpt-5.5",
        }
    ]
    assert report.generated_by == "openrouter:openai/gpt-4o-mini"


def test_ai_report_uses_openrouter_model_spec_as_primary(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_ENV_FILE", raising=False)
    (tmp_path / ".env.local").write_text(
        "OPENROUTER_API_KEY=sk-or-primary-test\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_openai_response(*, payload, api_key):
        raise AssertionError("OpenAI should not be called for openrouter: model specs")

    def fake_openrouter_response(*, payload, api_key, model):
        calls.append(
            {"api_key": api_key, "model": model, "payload_model": payload["model"]}
        )
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тестовый primary OpenRouter.",
                    },
                    "overview": "OpenRouter primary summary.",
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
            ),
            "_provider_model": "openrouter:google/gemini-3.1-pro-preview",
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)
    monkeypatch.setattr(
        reporting, "_post_openrouter_response", fake_openrouter_response
    )

    report = build_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
        ],
        use_ai=True,
        report_model="openrouter:google/gemini-3.1-pro-preview",
    )

    assert calls == [
        {
            "api_key": "sk-or-primary-test",
            "model": "google/gemini-3.1-pro-preview",
            "payload_model": "openrouter:google/gemini-3.1-pro-preview",
        }
    ]
    assert report.generated_by == "openrouter:google/gemini-3.1-pro-preview"
    assert report.overview == "OpenRouter primary summary."


def test_ai_json_parser_accepts_unescaped_control_characters():
    payload = reporting._loads_ai_json(
        '{"overview": "Первая строка\nвторая строка", "items": []}',
        context="OpenRouter",
    )

    assert payload == {"overview": "Первая строка\nвторая строка", "items": []}


def test_direct_ai_report_retries_invalid_json_once(monkeypatch):
    calls = 0

    def fake_openai_response(*, payload, api_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"output_text": '{"overview": "обрезанный'}
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Повторный ответ принят.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Итог"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="retry-call",
        title="Retry Call",
        source_filename="retry.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 5.0, "Договорились проверить retry.")],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
    )

    assert calls == 2
    assert report.overview == "Повторный ответ принят."


def test_ai_report_keeps_current_transcript_formal_items_when_ai_returns_fewer(
    monkeypatch,
):
    def fake_openai_response(*, payload, api_key):
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "AI сжал текущую встречу.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Решения"],
                            "rationale": "Тест.",
                        },
                        {
                            "segment_id": "S0002",
                            "status": "covered",
                            "section_titles": ["Решения"],
                            "rationale": "Тест.",
                        },
                        {
                            "segment_id": "S0003",
                            "status": "covered",
                            "section_titles": ["Задачи"],
                            "rationale": "Тест.",
                        },
                    ],
                    "timeline": [],
                    "decisions": [
                        {
                            "title": "PDF отчеты",
                            "text": "Договорились сделать PDF отчеты.",
                            "citations": ["S0001"],
                        }
                    ],
                    "action_items": [
                        {
                            "title": "Добавить coverage",
                            "text": "Нужно добавить coverage audit.",
                            "owner": "",
                            "due": "",
                            "citations": ["S0003"],
                        }
                    ],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="context-regression",
        title="Context Regression",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 10.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 11.0, 20.0, "Решили сохранить ссылки на таймкоды."),
            _segment("Илья", 21.0, 30.0, "Нужно добавить coverage audit."),
        ],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Фоновый контекст прошлых встреч."}]},
    )

    decision_text = "\n".join(item.text for item in report.decisions)

    assert len(report.decisions) >= 2
    assert "ссылки на таймкоды" in decision_text
    assert report.action_items


def test_context_ai_report_filters_methodology_example_formal_items(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {"overview_addendum": "", "memory_sections": []},
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "consultation",
                        "label": "Консультация",
                        "confidence": 0.9,
                        "rationale": "Методологический фрагмент.",
                    },
                    "overview": "Обсудили подход к стратегии обслуживания.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Методология"],
                            "rationale": "Тест.",
                        }
                    ],
                    "timeline": [],
                    "decisions": [],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [
                        {
                            "title": "Принятие риска",
                            "text": "Просто принимаем риск. О'кей, пусть он сломается, мы его отремонтируем, так будет в пять раз дешевле. Технологический процесс от этого не встанет.",
                            "citations": ["S0001"],
                        }
                    ],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="methodology-example-filter",
        title="Methodology Example Filter",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment(
                "Илья",
                0.0,
                10.0,
                "Обсуждаем экономическую проверку стратегии обслуживания оборудования.",
            )
        ],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Раньше обсуждали ТОиР."}]},
    )

    assert report.risks == []


def test_context_ai_report_filters_synthesized_methodology_decision(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {"overview_addendum": "", "memory_sections": []},
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "consultation",
                        "label": "Консультация",
                        "confidence": 0.9,
                        "rationale": "Методологический фрагмент.",
                    },
                    "overview": "Обсудили подход к стратегии обслуживания.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Методология"],
                            "rationale": "Тест.",
                        }
                    ],
                    "timeline": [],
                    "decisions": [
                        {
                            "title": "Принятие риска",
                            "text": "Для оборудования, где ремонт дешевле предупредительного обслуживания, принимается риск и используется подход ремонтировать по факту отказа.",
                            "citations": ["S0001"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="methodology-synth-decision-filter",
        title="Methodology Synth Decision Filter",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment(
                "Илья",
                0.0,
                10.0,
                "Обсуждаем экономическую проверку стратегии обслуживания оборудования.",
            )
        ],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Раньше обсуждали ТОиР."}]},
    )

    assert report.decisions == []


def test_context_ai_report_filters_methodology_example_from_local_recovery(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {"overview_addendum": "", "memory_sections": []},
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "consultation",
                        "label": "Консультация",
                        "confidence": 0.9,
                        "rationale": "Методологический фрагмент.",
                    },
                    "overview": "Обсудили подход к стратегии обслуживания.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Методология"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="methodology-local-recovery-filter",
        title="Methodology Local Recovery Filter",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment(
                "Илья",
                0.0,
                10.0,
                "Просто принимаем риск. О'кей, пусть он сломается, мы его отремонтируем, так будет в пять раз дешевле.",
            )
        ],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Раньше обсуждали ТОиР."}]},
    )

    assert report.risks == []


def test_context_ai_report_recovers_base_action_after_filtering_bad_ai_action(
    monkeypatch,
):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {"overview_addendum": "", "memory_sections": []},
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Обсудили текущую задачу.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Задачи"],
                            "rationale": "Тест.",
                        }
                    ],
                    "timeline": [],
                    "decisions": [],
                    "action_items": [
                        {
                            "title": "Методологическая пометка",
                            "text": "Просто нужно понимать, что в отчете надо учитывать контекст, окей.",
                            "owner": "",
                            "due": "",
                            "citations": ["S0001"],
                        }
                    ],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="bad-ai-action-recovery",
        title="Bad AI Action Recovery",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 10.0, "Нужно добавить coverage audit.")],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Фоновый контекст."}]},
    )

    assert [item.text for item in report.action_items] == [
        "Нужно добавить coverage audit."
    ]


def test_context_ai_report_upgrades_thin_baseline_before_memory(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL", "gpt-5.5")
    calls: list[tuple[str, str]] = []

    def report_payload(*, model: str, formal: str) -> dict[str, object]:
        if formal == "thin":
            decisions = [
                {
                    "title": "Тонкий пункт",
                    "text": "Зафиксирован один общий пункт без достаточной структуры.",
                    "citations": ["S0001"],
                }
            ]
            sections = [
                {
                    "kind": "summary",
                    "title": "Общее обсуждение",
                    "purpose": "Кратко.",
                    "summary": "Обсудили данные.",
                    "items": [],
                    "citations": ["S0001"],
                    "accent": "blue",
                }
            ]
        else:
            decisions = [
                {
                    "title": "Слой данных",
                    "text": "Зафиксировали текущую структуру слоя данных.",
                    "citations": ["S0001"],
                },
                {
                    "title": "Проверка отчетов",
                    "text": "Согласовали проверку качества отчетов.",
                    "citations": ["S0002"],
                },
            ]
            sections = [
                {
                    "kind": "architecture",
                    "title": "Текущая структура",
                    "purpose": "Зафиксировать текущие сущности.",
                    "summary": "Обсудили структуру данных и отчеты.",
                    "items": [],
                    "citations": ["S0001"],
                    "accent": "blue",
                },
                {
                    "kind": "quality",
                    "title": "Проверка качества",
                    "purpose": "Зафиксировать контроль качества.",
                    "summary": "Отдельно обсудили проверку отчетов.",
                    "items": [],
                    "citations": ["S0002"],
                    "accent": "green",
                },
            ]
        return {
            "profile": {
                "kind": "project_sync",
                "label": "Проектная встреча",
                "confidence": 0.9,
                "rationale": model,
            },
            "overview": f"{model} {formal}",
            "adaptive_sections": sections,
            "coverage": [
                {
                    "segment_id": "S0001",
                    "status": "covered",
                    "section_titles": ["Текущее"],
                    "rationale": "Тест.",
                },
                {
                    "segment_id": "S0002",
                    "status": "covered",
                    "section_titles": ["Качество"],
                    "rationale": "Тест.",
                },
            ],
            "timeline": [],
            "decisions": decisions,
            "action_items": [
                {
                    "title": "Проверить отчет",
                    "text": "Проверить отчет на полноту.",
                    "owner": "",
                    "due": "",
                    "citations": ["S0002"],
                }
            ]
            if formal == "rich"
            else [],
            "open_questions": [],
            "risks": [],
            "notable_quotes": [],
        }

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        model = payload["model"]
        calls.append((schema_name, model))
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {
                        "overview_addendum": "",
                        "memory_sections": [
                            {
                                "kind": "memory_context",
                                "title": "Фон",
                                "purpose": "Добавить память.",
                                "summary": "Память добавлена после upgrade.",
                                "items": [],
                                "citations": [],
                                "accent": "violet",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }
        formal = "rich" if model == "gpt-5.5" else "thin"
        return {
            "output_text": json.dumps(
                report_payload(model=model, formal=formal), ensure_ascii=False
            )
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="baseline-upgrade",
        title="Baseline Upgrade",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 10.0, "Обсудили текущую структуру слоя данных."),
            _segment("Анна", 11.0, 20.0, "Обсудили проверку качества отчетов."),
        ],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Фоновая память."}]},
    )

    assert calls == [
        ("meeting_report", "gpt-4.1-mini"),
        ("meeting_report", "gpt-5.5"),
        ("meeting_report_memory_enrichment", "gpt-4.1-mini"),
    ]
    assert report.generated_by == "gpt-5.5"
    assert len(report.decisions) == 2
    assert report.action_items
    assert any("AI baseline upgraded" in warning for warning in report.warnings)
    assert any(section.title == "Память: Фон" for section in report.adaptive_sections)


def test_ai_report_uses_memory_as_enrichment_after_current_extraction(monkeypatch):
    request_payloads: list[dict[str, object]] = []
    schema_names: list[str] = []

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        schema_names.append(schema_name)
        request_payload = json.loads(payload["input"][1]["content"])
        request_payloads.append(request_payload)
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {
                        "overview_addendum": "Память связывает это с прошлой схемой БД.",
                        "memory_sections": [
                            {
                                "kind": "memory_context",
                                "title": "Связь с памятью",
                                "purpose": "Показать фон прошлых встреч отдельно от фактов текущей.",
                                "summary": "Похожая схема БД обсуждалась раньше.",
                                "items": [
                                    {
                                        "title": "Предыдущий слой данных",
                                        "text": "В прошлой встрече уже обсуждали bronze/silver/gold.",
                                        "citations": ["S0001"],
                                    }
                                ],
                                "citations": ["S0001"],
                                "accent": "violet",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "technical_discussion",
                        "label": "Техническое обсуждение",
                        "confidence": 0.9,
                        "rationale": "Текущий транскрипт.",
                    },
                    "overview": "Обсудили текущую схему.",
                    "adaptive_sections": [
                        {
                            "kind": "architecture",
                            "title": "Текущая схема",
                            "purpose": "Зафиксировать текущий фрагмент.",
                            "summary": "Только факты текущей встречи.",
                            "items": [
                                {
                                    "title": "PDF отчеты",
                                    "text": "Договорились сделать PDF отчеты.",
                                    "citations": ["S0001"],
                                }
                            ],
                            "citations": ["S0001"],
                            "accent": "blue",
                        }
                    ],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Текущая схема"],
                            "rationale": "Тест.",
                        },
                        {
                            "segment_id": "S0002",
                            "status": "covered",
                            "section_titles": ["Текущая схема"],
                            "rationale": "Тест.",
                        },
                    ],
                    "timeline": [],
                    "decisions": [
                        {
                            "title": "PDF отчеты",
                            "text": "Договорились сделать PDF отчеты.",
                            "citations": ["S0001"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="memory-enrichment",
        title="Memory Enrichment",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 10.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 11.0, 20.0, "Это связано с прошлой схемой БД."),
        ],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Раньше обсуждали bronze/silver/gold."}]},
    )

    assert schema_names == ["meeting_report", "meeting_report_memory_enrichment"]
    assert "prior_context" not in request_payloads[0]
    assert request_payloads[1]["prior_context"] == {
        "facts": [{"summary": "Раньше обсуждали bronze/silver/gold."}]
    }
    assert len(report.decisions) == 1
    assert report.decisions[0].citations == ["S0001"]
    assert any(section.kind == "memory_context" for section in report.adaptive_sections)
    assert "Память связывает" not in report.overview
    assert any(
        section.title == "Память: контекст прошлых встреч"
        and "Память связывает" in section.summary
        for section in report.adaptive_sections
    )


def test_memory_enrichment_strips_prior_context_segment_refs_from_text(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {
                        "overview_addendum": "В прошлой встрече это называли стратегией (S0232).",
                        "memory_sections": [
                            {
                                "kind": "memory_context",
                                "title": "Связь с прошлым",
                                "purpose": "Показать фон без чужих ссылок.",
                                "summary": "Открытый вопрос был в S0232.",
                                "items": [
                                    {
                                        "title": "Прошлая стратегия",
                                        "text": "Стратегия обсуждалась раньше (S0232).",
                                        "citations": ["S0232"],
                                    }
                                ],
                                "citations": ["S0232"],
                                "accent": "violet",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "technical_discussion",
                        "label": "Техническое обсуждение",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Текущая встреча.",
                    "adaptive_sections": [],
                    "coverage": [
                        {
                            "segment_id": "S0001",
                            "status": "covered",
                            "section_titles": ["Текущее"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="memory-ref-filter",
        title="Memory Ref Filter",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 10.0, "Обсуждаем текущую стратегию.")],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Старый вопрос.", "citations": ["S0232"]}]},
    )
    rendered = json.dumps(build_protocol_data(report), ensure_ascii=False)

    assert "S0232" not in rendered
    assert report.adaptive_sections[-1].items[0].citations == []


def test_memory_enrichment_keeps_memory_sections_separate_and_uncited(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            return {
                "output_text": json.dumps(
                    {
                        "overview_addendum": "",
                        "memory_sections": [
                            {
                                "kind": "actions",
                                "title": "Задачи и next steps",
                                "purpose": "Не должно слиться с текущими задачами.",
                                "summary": "В прошлом обсуждали похожий шаг S0001.",
                                "items": [
                                    {
                                        "title": "Прошлый шаг",
                                        "text": "Старый контекст с совпадающим S0001.",
                                        "citations": ["S0001"],
                                    }
                                ],
                                "citations": ["S0001"],
                                "accent": "red",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "technical_discussion",
                        "label": "Техническое обсуждение",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Текущая встреча.",
                    "adaptive_sections": [
                        {
                            "kind": "actions",
                            "title": "Задачи и next steps",
                            "purpose": "Текущие задачи.",
                            "summary": "Текущий раздел не должен быть перезаписан памятью.",
                            "items": [],
                            "citations": ["S0001"],
                            "accent": "blue",
                        }
                    ],
                    "coverage": [],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="memory-section-isolation",
        title="Memory Section Isolation",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 10.0, "Нужно добавить текущую задачу.")],
        use_ai=True,
        report_model="gpt-4.1-mini",
        api_key="sk-test",
        context_pack={
            "facts": [{"summary": "Старый похожий шаг.", "citations": ["S0001"]}]
        },
    )

    assert [section.title for section in report.adaptive_sections] == [
        "Задачи и next steps",
        "Память: Задачи и next steps",
    ]
    memory_section = report.adaptive_sections[-1]
    assert memory_section.kind == "memory_context"
    assert memory_section.citations == []
    assert memory_section.items[0].citations == []
    assert "S0001" not in memory_section.summary
    assert "S0001" not in memory_section.items[0].text


def test_local_action_items_ignore_methodology_noise():
    report = reporting.build_local_report(
        meeting_id="methodology-noise",
        title="Methodology Noise",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment(
                "Илья",
                0.0,
                10.0,
                "Методологически нужно учитывать риск дешевого обслуживания.",
            ),
            _segment(
                "Илья",
                11.0,
                20.0,
                "В отчете нужно показывать это как оговорку, а не как задачу.",
            ),
            _segment("Анна", 21.0, 30.0, "Я добавлю отдельный раздел про ограничения."),
        ],
    )

    assert [item.text for item in report.action_items] == [
        "Я добавлю отдельный раздел про ограничения."
    ]


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
        if schema_name == "meeting_report_memory_enrichment":
            synthesis_payloads.append(request_payload)
            return {
                "output_text": json.dumps(
                    {
                        "overview_addendum": "Старый контекст учтен отдельно.",
                        "memory_sections": [],
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
    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )
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
        "meeting_report_memory_enrichment",
    ]
    assert report.generated_by == "gpt-5.5"
    assert report.profile.kind == "technical_discussion"
    assert "prior_context" not in synthesis_payloads[0]
    assert synthesis_payloads[1]["prior_context"] == context_pack
    assert {entry.segment_id for entry in report.coverage} == {
        "S0001",
        "S0002",
        "S0003",
        "S0004",
        "S0005",
    }


def test_write_report_artifacts_marks_ai_fallback_as_degraded(tmp_path, monkeypatch):
    def fake_openai_response(*, payload, api_key):
        raise reporting.ReportGenerationError(
            "OpenAI API request failed: test disconnect"
        )

    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )

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
    (tmp_path / "report.md").write_text(
        render_report_markdown(report), encoding="utf-8"
    )
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


def test_report_health_treats_baseline_upgrade_as_non_blocking_warning(tmp_path):
    report = build_local_report(
        meeting_id="upgrade-call",
        title="Upgrade Call",
        source_filename="upgrade.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
    )
    report.generated_by = "gpt-5.5"
    report.warnings.append(
        "AI baseline upgraded: gpt-4.1-mini -> gpt-5.5; "
        "review: thin formal coverage (7/10) -> pass (10/10)."
    )
    (tmp_path / "report.json").write_text(
        json.dumps(build_protocol_data(report), ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "report.md").write_text(
        render_report_markdown(report), encoding="utf-8"
    )
    (tmp_path / "report.html").write_text(render_report_html(report), encoding="utf-8")
    (tmp_path / "coverage.json").write_text(
        json.dumps(reporting.build_coverage_payload(report), ensure_ascii=False),
        encoding="utf-8",
    )

    health = reporting.build_report_health(
        report=report,
        output_dir=tmp_path,
        requested_ai=True,
        requested_pdf=False,
        pdf_path=None,
        pdf_error=None,
    )

    assert health["status"] == "ok"
    assert health["warnings"] == report.warnings
    assert health["alerts"] == []
    checks = {check["name"]: check for check in health["checks"]}
    assert checks["report_has_no_blocking_warnings"]["ok"] is True


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
    (tmp_path / "report.md").write_text(
        render_report_markdown(report), encoding="utf-8"
    )
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
            raise reporting.ReportGenerationError(
                "AI synthesis failed after 1 attempt(s): test disconnect"
            )
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
    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )

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
    assert (
        report.adaptive_sections[0].items[0].text == "Обсуждаем структуру базы данных."
    )
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
    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )

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
            raise reporting.ReportGenerationError(
                "AI synthesis failed after 1 attempt(s): test disconnect"
            )
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
    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )

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
    monkeypatch.setattr(
        "mac_transcriber.reporting._post_openai_response", fake_openai_response
    )

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
            _segment(
                "Лектор",
                0.0,
                20.0,
                "Сегодня разберем продуктовую аналитику и ключевые метрики.",
            ),
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

    assert (
        "[S0001-S0002] [00:00:00 - 00:00:05] **Илья:** Первая часть. Вторая часть."
        in markdown
    )
    assert "[S0003] [00:00:06 - 00:00:08] **Анна:** Ответ." in markdown

    pipeline_path = Path(
        "mac_transcriber/protocol/scripts/protocol_pipeline.py"
    ).resolve()
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
    assert payload["sections"]
    assert payload["decisions"][0]["id"] == "D-01"
    assert payload["tasks"][0]["id"] == "T-01"
    assert "## Смысловые разделы" in (tmp_path / "report.md").read_text(
        encoding="utf-8"
    )
    assert "Смысловые разделы" in (tmp_path / "report.html").read_text(encoding="utf-8")
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
                    'OpenAI API returned HTTP 429: {"error":{"code":"insufficient_quota",'
                    '"message":"You exceeded your current quota"}}'
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
    assert {item["kind"] for item in summary_json["slack"]["files"]} == {
        "report_md",
        "transcript_md",
    }
    assert summary_json["coverage_json"].endswith("coverage.json")


def test_asr_write_artifacts_defaults_to_ai_and_pdf_for_service_jobs(
    tmp_path, monkeypatch
):
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


# --- OpenRouter HTTP layer / parser coverage (audit-driven) ---


import httpx as _httpx
import openai as _openai


def _api_status_error(status, body, *, cls=_openai.APIStatusError):
    response = _httpx.Response(
        status_code=status, request=_httpx.Request("POST", "https://api.test")
    )
    return cls("provider error", response=response, body=body)


class _FakeResponsesStream:
    def __init__(self, *, output_text=None, error=None):
        self._output_text = output_text
        self._error = error

    def __enter__(self):
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *_a):
        return False

    def __iter__(self):
        return iter(())

    def get_final_response(self):
        return types.SimpleNamespace(output_text=self._output_text)


class _FakeResponsesAPI:
    def __init__(self, *, output_text=None, error=None):
        self._output_text = output_text
        self._error = error

    def stream(self, **_kwargs):
        return _FakeResponsesStream(output_text=self._output_text, error=self._error)


def _chat_chunk(text):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))]
    )


class _FakeChatCompletions:
    def __init__(self, behaviors):
        # behaviors: на каждый .create() — Exception (бросить) или str (стримить как контент).
        self._behaviors = list(behaviors)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        text = behavior
        mid = len(text) // 2

        def gen():
            for piece in (text[:mid], text[mid:]):
                if piece:
                    yield _chat_chunk(piece)

        return gen()


def _fake_openai_client(monkeypatch, *, responses=None, chat_completions=None):
    client = types.SimpleNamespace()
    if responses is not None:
        client.responses = responses
    if chat_completions is not None:
        client.chat = types.SimpleNamespace(completions=chat_completions)
    monkeypatch.setattr(reporting, "_openai_client", lambda *a, **k: client)
    return client


def _openrouter_payload():
    return {
        "input": [{"role": "user", "content": "hi"}],
        "text": {"format": {"name": "meeting_report", "schema": {}, "strict": True}},
    }


def test_post_openrouter_response_retries_json_object_on_grammar_error(monkeypatch):
    grammar_err = _api_status_error(
        400,
        {"error": {"message": "Compiled grammar is too large"}},
        cls=_openai.BadRequestError,
    )
    chat = _FakeChatCompletions([grammar_err, '{"ok": true}'])
    _fake_openai_client(monkeypatch, chat_completions=chat)

    result = reporting._post_openrouter_response(
        payload=_openrouter_payload(), api_key="k", model="vendor/model"
    )

    assert len(chat.calls) == 2
    assert chat.calls[0]["response_format"]["type"] == "json_schema"
    assert chat.calls[0].get("extra_body") == {"structured_outputs": True}
    assert chat.calls[1]["response_format"] == {"type": "json_object"}
    assert chat.calls[1].get("extra_body") is None
    assert chat.calls[1]["messages"][0]["role"] == "system"
    assert chat.calls[0].get("stream") is True
    assert result["output_text"] == '{"ok": true}'
    assert result["_provider_model"] == "openrouter:vendor/model"


def test_post_openrouter_response_raises_on_non_grammar_http_error(monkeypatch):
    err = _api_status_error(
        429, {"error": "rate limit exceeded"}, cls=_openai.RateLimitError
    )
    _fake_openai_client(monkeypatch, chat_completions=_FakeChatCompletions([err]))

    with pytest.raises(reporting.ReportGenerationError, match="returned HTTP 429"):
        reporting._post_openrouter_response(
            payload=_openrouter_payload(), api_key="k", model="vendor/model"
        )


def test_post_openrouter_response_wraps_transport_error(monkeypatch):
    err = _openai.APIConnectionError(request=_httpx.Request("POST", "https://api.test"))
    _fake_openai_client(monkeypatch, chat_completions=_FakeChatCompletions([err]))

    with pytest.raises(reporting.ReportGenerationError, match="request failed"):
        reporting._post_openrouter_response(
            payload=_openrouter_payload(), api_key="k", model="vendor/model"
        )


def test_post_openrouter_response_raises_on_missing_content(monkeypatch):
    _fake_openai_client(monkeypatch, chat_completions=_FakeChatCompletions([""]))

    with pytest.raises(
        reporting.ReportGenerationError, match="did not contain message content"
    ):
        reporting._post_openrouter_response(
            payload=_openrouter_payload(), api_key="k", model="vendor/model"
        )


def test_post_openrouter_response_accumulates_streamed_chunks(monkeypatch):
    _fake_openai_client(monkeypatch, chat_completions=_FakeChatCompletions(['{"a":1}']))

    result = reporting._post_openrouter_response(
        payload=_openrouter_payload(), api_key="k", model="vendor/model"
    )

    assert result["output_text"] == '{"a":1}'


def test_post_openai_response_surfaces_http_error(monkeypatch):
    err = _api_status_error(500, {"error": {"message": "internal"}})
    _fake_openai_client(monkeypatch, responses=_FakeResponsesAPI(error=err))

    with pytest.raises(
        reporting.ReportGenerationError, match="OpenAI API returned HTTP 500"
    ):
        reporting._post_openai_response(payload={"model": "gpt-5.5"}, api_key="k")


def test_post_openai_response_wraps_network_error(monkeypatch):
    err = _openai.APIConnectionError(request=_httpx.Request("POST", "https://api.test"))
    _fake_openai_client(monkeypatch, responses=_FakeResponsesAPI(error=err))

    with pytest.raises(reporting.ReportGenerationError, match="request failed"):
        reporting._post_openai_response(payload={"model": "gpt-5.5"}, api_key="k")


def test_post_openai_response_streams_output_text(monkeypatch):
    _fake_openai_client(
        monkeypatch, responses=_FakeResponsesAPI(output_text='{"overview": "ok"}')
    )

    result = reporting._post_openai_response(payload={"model": "gpt-5.5"}, api_key="k")

    assert result["output_text"] == '{"overview": "ok"}'


@pytest.mark.parametrize(
    "body",
    [
        "Compiled grammar is too large",
        "please SIMPLIFY YOUR TOOL SCHEMAS and retry",
        "...compiled grammar is too large for this model...",
    ],
)
def test_is_openrouter_grammar_error_true(body):
    assert reporting._is_openrouter_grammar_error(body) is True


@pytest.mark.parametrize("body", ["rate limit exceeded", "internal server error", ""])
def test_is_openrouter_grammar_error_false(body):
    assert reporting._is_openrouter_grammar_error(body) is False


def test_openrouter_max_tokens_default_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_ENV_FILE", raising=False)
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    assert reporting.openrouter_max_tokens(env_path=env) == 12000


def test_openrouter_max_tokens_reads_value(tmp_path, monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_ENV_FILE", raising=False)
    env = tmp_path / ".env"
    env.write_text("MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS=8000\n", encoding="utf-8")
    assert reporting.openrouter_max_tokens(env_path=env) == 8000


def test_openrouter_max_tokens_clamps_to_one(tmp_path, monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_ENV_FILE", raising=False)
    env = tmp_path / ".env"
    env.write_text("MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS=0\n", encoding="utf-8")
    assert reporting.openrouter_max_tokens(env_path=env) == 1


def test_openrouter_max_tokens_falls_back_on_bad_int(tmp_path, monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_OPENROUTER_ENV_FILE", raising=False)
    env = tmp_path / ".env"
    env.write_text("MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS=abc\n", encoding="utf-8")
    assert reporting.openrouter_max_tokens(env_path=env) == 12000


def test_ai_json_parser_rejects_non_object_payload():
    with pytest.raises(
        reporting.ReportGenerationError, match="response JSON must be an object"
    ):
        reporting._loads_ai_json("[]", context="OpenRouter")


def test_ai_json_parser_reraises_when_strict_false_also_fails():
    # Контрол-символ запускает strict=False ретрай, но строка всё равно битая.
    with pytest.raises(
        reporting.ReportGenerationError, match="response was not valid JSON"
    ):
        reporting._loads_ai_json('{"a": "line1\nline2', context="OpenAI")


def test_enrich_memory_degrades_when_enrichment_call_fails(monkeypatch):
    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            raise reporting.ReportGenerationError("enrichment boom")
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Базовый отчёт без памяти.",
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
                    "decisions": [
                        {
                            "title": "PDF",
                            "text": "Договорились сделать PDF отчёты.",
                            "citations": ["S0001"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="enrich-fail",
        title="Enrich Fail",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 10.0, "Договорились сделать PDF отчёты.")],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Контекст прошлых встреч."}]},
    )

    # Базовый отчёт сохранён, память не стёрла текущие пункты, ошибка не уронила пайплайн.
    assert report.overview == "Базовый отчёт без памяти."
    assert report.decisions
    assert any(
        warning.startswith("AI memory enrichment skipped:")
        for warning in report.warnings
    )


def test_merge_ai_payload_survives_malformed_openrouter_payload():
    # OpenRouter json_object fallback не валидирует схему: строковый confidence и
    # не-dict элементы списков не должны ронять пайплайн (иначе нет локального фоллбэка).
    base = build_local_report(
        meeting_id="m",
        title="T",
        source_filename="a.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 5.0, "Договорились сделать отчёт.")],
    )
    payload = {
        "profile": {
            "kind": "general",
            "label": "L",
            "confidence": "high",
            "rationale": "r",
        },
        "overview": "ov",
        "adaptive_sections": [],
        "timeline": ["not-a-dict"],
        "decisions": ["also-bad", 7],
        "action_items": [42],
        "open_questions": [],
        "risks": [],
        "notable_quotes": [],
    }

    report = reporting._merge_ai_payload(base, payload, report_model="gpt-x")

    assert report.profile.confidence == base.profile.confidence
    assert report.timeline == []


def test_memory_enrichment_tracks_continuity_and_preserves_formal_items(monkeypatch):
    enrichment_payloads = []

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_memory_enrichment":
            enrichment_payloads.append(payload)
            return {
                "output_text": json.dumps(
                    {
                        "overview_addendum": "Связь с прошлой встречей.",
                        "memory_sections": [
                            {
                                "kind": "open_threads_from_memory",
                                "title": "Память: открытые хвосты",
                                "purpose": "Преемственность.",
                                "summary": "Статусы прошлых пунктов.",
                                "items": [
                                    {
                                        "title": "Аренда сервера",
                                        "text": "Задача с прошлой встречи — всё ещё открыто.",
                                    }
                                ],
                                "citations": [],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Текущая встреча.",
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
                    "decisions": [
                        {
                            "title": "Решение",
                            "text": "Договорились арендовать сервер.",
                            "citations": ["S0001"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="continuity",
        title="Continuity",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 10.0, "Договорились арендовать сервер.")],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
        context_pack={"facts": [{"summary": "Прошлая задача: аренда сервера."}]},
    )

    # Память добавлена отдельной секцией, текущее решение НЕ потеряно.
    assert report.decisions
    memory_titles = [
        s.title for s in report.adaptive_sections if s.title.startswith("Память")
    ]
    assert any("открытые хвосты" in t.lower() for t in memory_titles)
    # Промпт enrichment явно требует классификацию преемственности.
    prompt = enrichment_payloads[0]["input"][0]["content"]
    assert "всё ещё открыто" in prompt
    assert "закрыто на этой встрече" in prompt


def test_reasoning_effort_env_override(monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_REASONING_EFFORT", raising=False)
    assert reporting._reasoning_effort("medium") == "medium"
    assert reporting._reasoning_effort("low") == "low"
    monkeypatch.setenv("MAC_TRANSCRIBER_REASONING_EFFORT", "high")
    assert reporting._reasoning_effort("medium") == "high"
    assert reporting._reasoning_effort("low") == "high"
    monkeypatch.setenv("MAC_TRANSCRIBER_REASONING_EFFORT", "bogus")
    assert reporting._reasoning_effort("low") == "low"


def test_chunked_report_skips_failed_chunk_instead_of_local_fallback(monkeypatch):
    # Регресс на реальный баг: один пустой чанк ронял весь отчёт длинной встречи
    # в сырой local-дамп ("бред"). Теперь сбойный чанк пропускается.
    chunk_calls = {"n": 0}

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_chunk":
            chunk_calls["n"] += 1
            if chunk_calls["n"] == 2:
                return {"output_text": ""}  # второй чанк вернул пусто
            seg = json.loads(payload["input"][1]["content"])["segments"][0]
            return {
                "output_text": json.dumps(
                    {
                        "summary": f"Фрагмент {seg['text']}",
                        "key_points": [
                            {
                                "title": "K",
                                "text": seg["text"],
                                "citations": [seg["segment_id"]],
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
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "technical_discussion",
                        "label": "Тех",
                        "confidence": 0.9,
                        "rationale": "t",
                    },
                    "overview": "AI синтез из уцелевших чанков.",
                    "adaptive_sections": [],
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

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 2)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 2)
    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="long",
        title="Long",
        source_filename="l.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("A", 0, 8, "один"),
            _segment("B", 9, 16, "два"),
            _segment("A", 17, 24, "три"),
            _segment("B", 25, 32, "четыре"),
            _segment("A", 33, 40, "пять"),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    # НЕ откат в local: остаётся AI-синтез из уцелевших чанков, плюс честный warning.
    assert report.generated_by != "local"
    assert report.overview == "AI синтез из уцелевших чанков."
    assert any("chunks skipped" in w for w in report.warnings)


def test_quota_error_is_not_swallowed_into_local_report(monkeypatch):
    # "Нет денег" (429 quota) НЕ должно превращаться в сырой local-отчёт —
    # ошибка обязана всплыть, чтобы сервис поставил встречу на паузу.
    def fake(*, payload, api_key):
        raise reporting.ReportQuotaError(
            "OpenAI API returned HTTP 429: insufficient_quota"
        )

    monkeypatch.setattr(reporting, "_post_openai_response", fake)

    with pytest.raises(reporting.ReportQuotaError):
        build_report(
            meeting_id="m",
            title="T",
            source_filename="a.m4a",
            model_name="v3_e2e_rnnt",
            segments=[_segment("A", 0, 5, "Договорились что-то сделать.")],
            use_ai=True,
            report_model="gpt-5.5",
            api_key="sk-test",
        )


def test_quota_error_in_chunk_propagates_not_skipped(monkeypatch):
    # Quota на чанке не должна «пропускаться» как сбойный чанк — она терминальна.
    def fake(*, payload, api_key):
        raise reporting.ReportQuotaError(
            "OpenAI API returned HTTP 429: exceeded your current quota"
        )

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 1)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 1)
    monkeypatch.setattr(reporting, "_post_openai_response", fake)

    with pytest.raises(reporting.ReportQuotaError):
        build_report(
            meeting_id="m",
            title="T",
            source_filename="a.m4a",
            model_name="v3_e2e_rnnt",
            segments=[
                _segment("A", 0, 5, "один"),
                _segment("B", 6, 10, "два"),
            ],
            use_ai=True,
            report_model="gpt-5.5",
            api_key="sk-test",
        )


def test_post_openai_response_classifies_quota_as_quota_error(monkeypatch):
    err = _api_status_error(
        429, {"error": {"code": "insufficient_quota"}}, cls=_openai.RateLimitError
    )
    _fake_openai_client(monkeypatch, responses=_FakeResponsesAPI(error=err))

    with pytest.raises(reporting.ReportQuotaError):
        reporting._post_openai_response(payload={"model": "gpt-5.5"}, api_key="k")


def test_synthesis_skips_failed_batch_instead_of_noisy_notes_merge(monkeypatch):
    # Регресс: падение одного батча синтеза роняло весь синтез в шумную сборку
    # из чанк-заметок (generated_by .../chunked). Теперь сбойный батч пропускается,
    # синтез собирается из выживших батчей.
    def synthesis_payload(overview):
        return {
            "profile": {
                "kind": "project_sync",
                "label": "X",
                "confidence": 0.9,
                "rationale": "t",
            },
            "overview": overview,
            "adaptive_sections": [],
            "timeline": [],
            "decisions": [],
            "action_items": [],
            "open_questions": [],
            "risks": [],
            "notable_quotes": [],
        }

    def fake(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        content = payload["input"][1]["content"]
        if schema_name == "meeting_report_chunk":
            seg = json.loads(content)["segments"][0]
            return {
                "output_text": json.dumps(
                    {
                        "summary": f"note {seg['text']}",
                        "key_points": [
                            {
                                "title": "K",
                                "text": seg["text"],
                                "citations": [seg["segment_id"]],
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
        # synthesis batch: первый батч (содержит note alpha) — пустой ответ (сбой),
        # остальные — валидны.
        if "alpha" in content:
            return {"output_text": ""}
        return {
            "output_text": json.dumps(
                synthesis_payload("Синтез из выживших батчей."), ensure_ascii=False
            )
        }

    monkeypatch.setattr(reporting, "AI_DIRECT_SEGMENT_LIMIT", 1)
    monkeypatch.setattr(reporting, "AI_CHUNK_SIZE", 1)
    monkeypatch.setattr(reporting, "AI_SYNTHESIS_BATCH_SIZE", 1)
    monkeypatch.setattr(reporting, "AI_SYNTHESIS_CHUNK_LIMIT", 100)
    monkeypatch.setattr(reporting, "_post_openai_response", fake)

    report = build_report(
        meeting_id="long",
        title="Long",
        source_filename="l.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("A", 0, 5, "alpha"),
            _segment("B", 6, 10, "beta"),
            _segment("A", 11, 15, "gamma"),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    # Синтез (не шумная сборка /chunked), собран из выживших батчей + честный warning.
    assert report.generated_by == "gpt-5.5"
    assert report.overview.startswith("Синтез из выживших батчей")
    assert any("batches skipped" in w for w in report.warnings)


@pytest.mark.parametrize(
    "text",
    [
        # Сырой шум, который аудит транскрипт↔отчёт нашёл в "Решениях":
        "Да, я просто приложил. Договорились. Вадим, Вадим. Ага. Так, да, сейчас.",  # филлер+повтор
        "Но мы не решили, то есть если ты решил, что мы идём в сторону Тулов",  # отрицание решения
        "Тогда ты говоришь: «Ну всё, мы решили снова зло в нём».",  # гипотетическая речь
    ],
)
def test_raw_transcript_noise_is_filtered_from_formal_items(text):
    assert reporting._looks_like_raw_transcript_formal_item(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Решили сохранить ссылки на таймкоды.",
        "Нужно добавить coverage audit.",
        "Команда договорилась работать через регулярную выдачу спецификаций, "
        "постановку задач и оценку сроков в YouTrack.",
    ],
)
def test_substantive_formal_items_are_kept(text):
    assert reporting._looks_like_raw_transcript_formal_item(text) is False


@pytest.mark.parametrize(
    "text",
    [
        # Сырые реплики, которые recover тащил в Решения/Вопросы/Риски (tech-leads_2026-06-17):
        "Ну, с ребятами согласовали, все поняли свои задачи.",  # зачин «ну» + филлер
        "Коллеги, только от себя один комментарий: если что-то решили, надо это "
        "зафиксировать в Аде.",  # вокатив-зачин
        "Хорошо. Тогда что мы там решим? Илья? По поводу того, кто берёт задачу?",  # зачин «хорошо»
        "Позиции, да, свои по интервью, да. То есть я не знаю, как это, можно ли "
        "будет применить для PP.",  # филлер «то есть»
        "Да, ну тут, Вячеслав, смотрите, говорю, помимо самой там технологии.",  # зачин «да» + «ну»
    ],
)
def test_raw_recovered_statement_is_rejected(text):
    assert reporting._is_clean_recovered_statement(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Решили сохранить ссылки на таймкоды.",
        "Нужно добавить coverage audit.",
        "Команда договорилась работать через регулярную выдачу спецификаций, "
        "постановку задач и оценку сроков в YouTrack.",
    ],
)
def test_clean_recovered_statement_is_kept(text):
    assert reporting._is_clean_recovered_statement(text) is True


def test_recover_drops_raw_base_item_but_keeps_clean_dropped_decision():
    # AI выдал один чистый пункт; в базе — сырая реплика и реальное решение, которое
    # модель пропустила. recover должен вернуть только чистое решение.
    ai_items = [
        reporting.ReportItem(
            title="D-AI",
            text="Команда зафиксировала табличный список оборудования.",
            citations=["S0003"],
        )
    ]
    base_items = [
        reporting.ReportItem(
            title="Решение 1",
            text="Ну, с ребятами согласовали, все поняли свои задачи.",
            citations=["S0001"],
        ),
        reporting.ReportItem(
            title="Решение 2",
            text="Решили сохранить ссылки на таймкоды.",
            citations=["S0002"],
        ),
    ]
    merged = reporting._merge_current_report_items(
        ai_items, base_items, recover_base=True
    )
    texts = [item.text for item in merged]
    assert "Команда зафиксировала табличный список оборудования." in texts
    assert "Решили сохранить ссылки на таймкоды." in texts
    assert all("все поняли свои задачи" not in text for text in texts)


def _report_with_formal(
    *,
    decisions=None,
    action_items=None,
    transcript=None,
):
    transcript = transcript or [
        reporting.TranscriptRecord("S0001", 0.0, 5.0, "Илья", "Решили A."),
        reporting.TranscriptRecord("S0002", 5.0, 10.0, "Анна", "Решили B."),
    ]
    return reporting.MeetingReport(
        meeting_id="m",
        title="t",
        source_filename="f.m4a",
        model_name="x",
        generated_by="gpt-5.5",
        segment_count=len(transcript),
        duration=transcript[-1].end if transcript else 0.0,
        overview="",
        timeline=[],
        decisions=decisions or [],
        action_items=action_items or [],
        open_questions=[],
        risks=[],
        notable_quotes=[],
        transcript=transcript,
        profile=reporting.ReportProfile(
            kind="general", label="Общий разговор", confidence=0.5, rationale="r"
        ),
        adaptive_sections=[],
        coverage=[],
        warnings=[],
    )


def test_apply_critic_ops_merges_and_drops_with_citation_union():
    report = _report_with_formal(
        decisions=[
            reporting.ReportItem("D-1", "Решили A.", ["S0001"]),
            reporting.ReportItem("D-2", "Решили A иначе.", ["S0002"]),
            reporting.ReportItem("D-3", "Сырой шум.", ["S0001"]),
        ]
    )
    ops = {
        "decisions": [
            {
                "op": "merge",
                "id": "",
                "ids": ["D1", "D2"],
                "title": "Решение A",
                "text": "Решили выпускать A.",
                "reason": "дубль",
            },
            {
                "op": "drop",
                "id": "D3",
                "ids": [],
                "title": "",
                "text": "",
                "reason": "шум",
            },
        ],
        "action_items": [],
        "open_questions": [],
        "risks": [],
        "notes": "",
    }
    result = reporting._apply_critic_ops(report, ops)
    assert [item.text for item in result.decisions] == ["Решили выпускать A."]
    assert set(result.decisions[0].citations) == {"S0001", "S0002"}
    # coverage пересчитан и полон — итоговый отчёт проходит валидацию.
    validate_report_citations(result)


def test_apply_critic_ops_rewrite_keeps_citations_and_default_keeps_untouched():
    report = _report_with_formal(
        decisions=[
            reporting.ReportItem("D-1", "Зафиксировать в Аде.", ["S0002"]),
            reporting.ReportItem("D-2", "Решили B.", ["S0001"]),
        ]
    )
    ops = {
        "decisions": [
            {
                "op": "rewrite",
                "id": "D1",
                "ids": [],
                "title": "Фиксация в ADR",
                "text": "Решения фиксируются в ADR.",
                "reason": "ASR",
            }
            # D2 не упомянут — должен сохраниться по умолчанию.
        ],
        "action_items": [],
        "open_questions": [],
        "risks": [],
        "notes": "",
    }
    result = reporting._apply_critic_ops(report, ops)
    texts = [item.text for item in result.decisions]
    assert "Решения фиксируются в ADR." in texts
    assert "Решили B." in texts  # default keep
    assert all("в Аде" not in text for text in texts)
    rewritten = next(d for d in result.decisions if "ADR" in d.text)
    assert rewritten.citations == ["S0002"]  # citations исходного пункта сохранены


def test_apply_action_critic_ops_merge_keeps_owner_due():
    report = _report_with_formal(
        action_items=[
            reporting.ActionItem("T-1", "Сделать X.", "Илья", "", ["S0001"]),
            reporting.ActionItem("T-2", "Сделать X скорее.", "", "пятница", ["S0002"]),
        ]
    )
    ops = {
        "decisions": [],
        "action_items": [
            {
                "op": "merge",
                "id": "",
                "ids": ["T1", "T2"],
                "title": "X",
                "text": "Сделать X.",
                "reason": "дубль",
            }
        ],
        "open_questions": [],
        "risks": [],
        "notes": "",
    }
    result = reporting._apply_critic_ops(report, ops)
    assert len(result.action_items) == 1
    merged = result.action_items[0]
    assert merged.owner == "Илья"
    assert merged.due == "пятница"
    assert set(merged.citations) == {"S0001", "S0002"}


def test_build_report_applies_critic_pass_when_enabled(monkeypatch):
    monkeypatch.setattr(
        reporting, "report_critic_model", lambda env_path=None: "gpt-5.5"
    )

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_critic":
            return {
                "output_text": json.dumps(
                    {
                        "decisions": [
                            {
                                "op": "merge",
                                "id": "",
                                "ids": ["D1", "D2"],
                                "title": "Отчёты",
                                "text": "Решили выпускать PDF-отчёты.",
                                "reason": "дубль",
                            },
                            {
                                "op": "rewrite",
                                "id": "D3",
                                "ids": [],
                                "title": "Фиксация в ADR",
                                "text": "Решения фиксируются в ADR.",
                                "reason": "ASR",
                            },
                        ],
                        "action_items": [],
                        "open_questions": [],
                        "risks": [],
                        "notes": "merged PDF dupes; fixed ADR",
                    },
                    ensure_ascii=False,
                )
            }
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Обсудили отчёты.",
                    "adaptive_sections": [],
                    "coverage": [],
                    "timeline": [],
                    "decisions": [
                        {
                            "title": "D-1",
                            "text": "Решили выпускать отчёты в формате PDF.",
                            "citations": ["S0001"],
                        },
                        {
                            "title": "D-2",
                            "text": "Договорились готовить версии протоколов для печати.",
                            "citations": ["S0001"],
                        },
                        {
                            "title": "D-3",
                            "text": "Зафиксировать в Аде.",
                            "citations": ["S0002"],
                        },
                    ],
                    "action_items": [],
                    "open_questions": [],
                    "risks": [],
                    "notable_quotes": [],
                },
                ensure_ascii=False,
            )
        }

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="critic-call",
        title="Critic Call",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 10.0, "Решили сделать PDF отчёты."),
            _segment("Анна", 10.0, 20.0, "Зафиксировать в ADR."),
        ],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    texts = [item.text for item in report.decisions]
    # merge схлопнул два дубля PDF в один, rewrite починил "в Аде" -> ADR.
    assert "Решили выпускать PDF-отчёты." in texts
    assert any("ADR" in text for text in texts)
    assert all("в Аде" not in text for text in texts)
    assert len(report.decisions) == 2
    assert any(w.startswith("AI critic (gpt-5.5):") for w in report.warnings)
    validate_report_citations(report)


def test_build_report_critic_rollback_on_bad_output(monkeypatch):
    monkeypatch.setattr(
        reporting, "report_critic_model", lambda env_path=None: "gpt-5.5"
    )

    def fake_openai_response(*, payload, api_key):
        schema_name = payload["text"]["format"]["name"]
        if schema_name == "meeting_report_critic":
            return {"output_text": "это не json{"}
        return {
            "output_text": json.dumps(
                {
                    "profile": {
                        "kind": "project_sync",
                        "label": "Проектная встреча",
                        "confidence": 0.9,
                        "rationale": "Тест.",
                    },
                    "overview": "Обсудили отчёты.",
                    "adaptive_sections": [],
                    "coverage": [],
                    "timeline": [],
                    "decisions": [
                        {
                            "title": "D-1",
                            "text": "Решили сделать PDF отчёты.",
                            "citations": ["S0001"],
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

    monkeypatch.setattr(reporting, "_post_openai_response", fake_openai_response)

    report = build_report(
        meeting_id="critic-rollback",
        title="Critic Rollback",
        source_filename="call.m4a",
        model_name="v3_e2e_rnnt",
        segments=[_segment("Илья", 0.0, 10.0, "Решили сделать PDF отчёты.")],
        use_ai=True,
        report_model="gpt-5.5",
        api_key="sk-test",
    )

    # Критик упал — отдан доcritic-отчёт без потери пунктов + предупреждение.
    assert [item.text for item in report.decisions] == ["Решили сделать PDF отчёты."]
    assert any("AI critic skipped" in w for w in report.warnings)
    validate_report_citations(report)
