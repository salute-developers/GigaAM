import importlib.util
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ab_report_eval.py"
SPEC = importlib.util.spec_from_file_location("ab_report_eval", SCRIPT_PATH)
ab_report_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["ab_report_eval"] = ab_report_eval
SPEC.loader.exec_module(ab_report_eval)


def test_parse_model_spec_defaults_to_openai():
    assert ab_report_eval.parse_model_spec("gpt-5-mini") == ("openai", "gpt-5-mini")
    assert ab_report_eval.parse_model_spec("openrouter:qwen/qwen3-30b-a3b") == (
        "openrouter",
        "qwen/qwen3-30b-a3b",
    )


def test_parse_args_accepts_paired_memory():
    args = ab_report_eval.parse_args(
        [
            "--model",
            "openrouter:google/gemini-3.1-pro-preview",
            "--paired-memory",
        ]
    )

    assert args.paired_memory is True


def test_safe_name_removes_path_separators():
    assert ab_report_eval.safe_name("openrouter:qwen/qwen3-30b-a3b") == "openrouter_qwen_qwen3-30b-a3b"


def test_load_env_files_prefers_explicit_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("OPENROUTER_API_KEY=from-file\nCUSTOM_ONLY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-process")

    env = ab_report_eval.load_env_files([env_file])

    assert env["OPENROUTER_API_KEY"] == "from-file"
    assert env["CUSTOM_ONLY"] == "from-file"


def test_compare_variants_flags_formal_regression():
    no_context = {
        "status": "ok",
        "formal_total": 6,
        "sections": 2,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 4,
        "sections": 2,
        "markdown_bytes": 1200,
        "elapsed_seconds": 12.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(no_context, with_context)

    assert comparison["formal_delta"] == -2
    assert comparison["verdict"] == "regression: formal sections dropped"


def test_compare_variants_flags_lost_formal_evidence():
    no_context = {
        "status": "ok",
        "formal_total": 3,
        "formal_refs": ["S0101", "S0102", "S0103"],
        "formal_texts": [
            "Add Bronze database for uploaded files",
            "Preserve Silver layer for RAG",
            "Gold layer contains processed artifacts",
        ],
        "section_terms": ["bronze"],
        "sections": 2,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 3,
        "formal_refs": ["S0103"],
        "formal_texts": [
            "Gold layer contains processed artifacts",
            "Only unrelated memory context",
        ],
        "section_terms": ["memory"],
        "sections": 4,
        "markdown_bytes": 1200,
        "elapsed_seconds": 12.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(no_context, with_context)

    assert comparison["formal_refs_lost"] == ["S0101", "S0102"]
    assert comparison["formal_ref_recall"] == 0.333
    assert comparison["verdict"] == "review: lost formal evidence"


def test_compare_variants_does_not_flag_lost_refs_when_semantics_preserved():
    no_context = {
        "status": "ok",
        "formal_total": 3,
        "formal_refs": ["S0101", "S0102"],
        "formal_texts": [
            "Add Bronze database for uploaded files",
            "Report other managers tasks to Aziz",
            "Prepare server account",
        ],
        "section_terms": ["bronze"],
        "sections": 2,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 3,
        "formal_refs": ["S0103"],
        "formal_texts": [
            "Add a Bronze level database for information about uploaded files",
            "Tell Aziz about tasks assigned by other managers",
            "Prepare server account",
        ],
        "section_terms": ["bronze"],
        "sections": 3,
        "markdown_bytes": 1200,
        "elapsed_seconds": 12.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(no_context, with_context)

    assert comparison["formal_refs_lost"] == ["S0101", "S0102"]
    assert comparison["semantic_formal_recall"] == 1.0
    assert comparison["verdict"] == "pass"


def test_compare_variants_marks_thin_formal_coverage_as_review():
    no_context = {
        "status": "ok",
        "formal_total": 1,
        "sections": 2,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 2,
        "sections": 5,
        "markdown_bytes": 1200,
        "elapsed_seconds": 12.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(no_context, with_context)

    assert comparison["with_context_thin_formal"] is True
    assert comparison["verdict"] == "review: thin formal coverage"


def test_compare_variants_reports_paired_memory_impact_separately():
    no_context = {
        "status": "ok",
        "formal_total": 2,
        "formal_refs": ["S0101", "S0102"],
        "formal_texts": ["Decision one", "Task two"],
        "sections": 2,
        "memory_sections": 0,
        "methodology_sections": 0,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
        "paired_memory": True,
    }
    with_context = {
        "status": "ok",
        "formal_total": 2,
        "formal_refs": ["S0101", "S0102"],
        "formal_texts": ["Decision one", "Task two"],
        "sections": 5,
        "memory_sections": 3,
        "methodology_sections": 0,
        "quality_issues": [],
        "markdown_bytes": 1500,
        "elapsed_seconds": 12.0,
        "invalid_refs": [],
        "paired_memory": True,
    }

    comparison = ab_report_eval.compare_variants(no_context, with_context)

    assert comparison["verdict"] == "review: thin formal coverage"
    assert comparison["baseline_quality_verdict"].startswith(
        "review: thin formal coverage"
    )
    assert comparison["memory_impact_verdict"] == "pass: sidecar added"


def test_baseline_quality_allows_methodology_sections_without_formal_items():
    result = ab_report_eval.baseline_quality_verdict(
        {
            "status": "ok",
            "formal_total": 0,
            "formal_refs": [],
            "sections": 3,
            "methodology_sections": 2,
            "invalid_refs": [],
            "quality_issues": [],
        },
        genre="methodology",
    )

    assert result == {"verdict": "pass", "score": 10, "issues": []}


def test_baseline_quality_flags_thin_project_extraction():
    result = ab_report_eval.baseline_quality_verdict(
        {
            "status": "ok",
            "formal_total": 1,
            "formal_refs": ["S0101"],
            "sections": 1,
            "invalid_refs": [],
            "quality_issues": [],
        },
        genre="project",
    )

    assert result["score"] == 6
    assert result["verdict"] == "review: thin formal coverage, thin semantic sections"


def test_degraded_status_is_usable_for_quality_comparison():
    no_context = {
        "status": "degraded",
        "formal_total": 3,
        "formal_refs": ["S0101", "S0102", "S0103"],
        "formal_texts": ["Decision", "Task", "Risk"],
        "sections": 2,
        "memory_sections": 0,
        "methodology_sections": 0,
        "quality_issues": [],
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
        "paired_memory": True,
    }
    with_context = {
        **no_context,
        "sections": 5,
        "memory_sections": 3,
        "markdown_bytes": 1500,
        "elapsed_seconds": 12.0,
    }

    comparison = ab_report_eval.compare_variants(no_context, with_context)

    assert comparison["baseline_quality_verdict"] == "pass"
    assert comparison["memory_impact_verdict"] == "pass: sidecar added"
    assert comparison["verdict"] == "pass"


def test_memory_impact_flags_formal_ref_loss_in_paired_mode():
    comparison = {
        "paired_memory": True,
        "no_context_status": "ok",
        "with_context_status": "ok",
        "with_context_has_invalid_refs": False,
        "quality_issues_added": [],
        "formal_delta": 0,
        "formal_ref_recall": 0.5,
        "semantic_formal_recall": 1.0,
        "memory_sections_delta": 2,
        "sections_delta": 2,
    }

    assert (
        ab_report_eval.memory_impact_verdict(comparison)
        == "review: memory lost formal refs"
    )


def test_compare_variants_allows_thin_methodology_when_sections_improve():
    no_context = {
        "status": "ok",
        "formal_total": 1,
        "sections": 2,
        "methodology_sections": 2,
        "memory_sections": 0,
        "formal_overreach": False,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 2,
        "sections": 5,
        "methodology_sections": 3,
        "memory_sections": 2,
        "formal_overreach": False,
        "markdown_bytes": 1400,
        "elapsed_seconds": 13.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(
        no_context, with_context, genre="methodology"
    )

    assert comparison["with_context_thin_formal"] is True
    assert comparison["methodology_sections_delta"] == 1
    assert comparison["verdict"] == "pass"


def test_compare_variants_flags_methodology_over_formalization():
    no_context = {
        "status": "ok",
        "formal_total": 2,
        "sections": 3,
        "methodology_sections": 3,
        "memory_sections": 0,
        "formal_overreach": False,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 6,
        "sections": 6,
        "methodology_sections": 4,
        "memory_sections": 2,
        "formal_overreach": True,
        "markdown_bytes": 1600,
        "elapsed_seconds": 14.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(
        no_context, with_context, genre="methodology"
    )

    assert comparison["with_context_formal_overreach"] is True
    assert comparison["verdict"] == "review: methodology over-formalized"


def test_compare_variants_flags_methodology_formal_evidence_loss():
    no_context = {
        "status": "ok",
        "formal_total": 3,
        "formal_refs": ["S0278", "S0281", "S0288"],
        "formal_texts": [
            "Допустима реактивная стратегия при отсутствии угрозы производству.",
            "Мероприятия после экономической проверки переносятся в техкарты.",
            "Есть риск выхода за границы анализа надежности.",
        ],
        "sections": 2,
        "methodology_sections": 2,
        "memory_sections": 0,
        "formal_overreach": False,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 1,
        "formal_refs": ["S0281"],
        "formal_texts": [
            "Мероприятия после экономической проверки включаются в стратегию ТОиР."
        ],
        "sections": 6,
        "methodology_sections": 5,
        "memory_sections": 3,
        "formal_overreach": False,
        "markdown_bytes": 1800,
        "elapsed_seconds": 15.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(
        no_context, with_context, genre="methodology"
    )

    assert comparison["semantic_formal_recall"] < 0.75
    assert comparison["verdict"] == "review: methodology formal evidence lost"


def test_compare_variants_flags_raw_formal_item_regression():
    no_context = {
        "status": "ok",
        "formal_total": 2,
        "sections": 2,
        "methodology_sections": 2,
        "memory_sections": 0,
        "formal_overreach": False,
        "raw_formal_items": [],
        "methodology_example_formal_items": [],
        "quality_issues": [],
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 3,
        "sections": 5,
        "methodology_sections": 3,
        "memory_sections": 2,
        "formal_overreach": True,
        "raw_formal_items": [
            "Просто принимаем риск. О'кей, пусть он сломается, мы его отремонтируем."
        ],
        "methodology_example_formal_items": [
            "Просто принимаем риск. О'кей, пусть он сломается, мы его отремонтируем."
        ],
        "quality_issues": [
            "raw transcript-like formal item",
            "methodology example treated as formal item",
        ],
        "markdown_bytes": 1600,
        "elapsed_seconds": 14.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(
        no_context, with_context, genre="methodology"
    )

    assert comparison["quality_issues_added"] == [
        "methodology example treated as formal item",
        "raw transcript-like formal item",
    ]
    assert comparison["raw_formal_items_added"] == 1
    assert comparison["verdict"] == "review: methodology example formalized"


def test_compare_variants_distinguishes_consolidation_from_regression():
    no_context = {
        "status": "ok",
        "formal_total": 9,
        "sections": 4,
        "markdown_bytes": 1000,
        "elapsed_seconds": 10.0,
        "invalid_refs": [],
    }
    with_context = {
        "status": "ok",
        "formal_total": 7,
        "sections": 7,
        "markdown_bytes": 1200,
        "elapsed_seconds": 12.0,
        "invalid_refs": [],
    }

    comparison = ab_report_eval.compare_variants(no_context, with_context)

    assert comparison["formal_delta"] == -2
    assert comparison["sections_delta"] == 3
    assert comparison["verdict"] == "review: possible consolidation"


def test_normalize_payload_removes_reasoning_for_non_reasoning_models():
    payload = {"model": "gpt-4.1-mini", "reasoning": {"effort": "low"}, "input": []}

    normalized = ab_report_eval.normalize_payload_for_provider(
        payload,
        provider="openai",
        model="gpt-4.1-mini",
    )
    reasoning_payload = ab_report_eval.normalize_payload_for_provider(
        payload,
        provider="openai",
        model="gpt-5-mini",
    )

    assert "reasoning" not in normalized
    assert reasoning_payload["reasoning"] == {"effort": "low"}
    assert payload["reasoning"] == {"effort": "low"}


def test_provider_patch_uses_payload_model_for_openrouter(monkeypatch):
    calls = []

    def fake_post_openrouter_response(*, payload, api_key, model):
        calls.append({"model": model, "payload_model": payload["model"]})
        return {"output_text": "{}"}

    monkeypatch.setattr(
        ab_report_eval, "post_openrouter_response", fake_post_openrouter_response
    )

    with ab_report_eval.provider_patch("openrouter", "cheap/model"):
        ab_report_eval.reporting._post_openai_response(
            payload={"model": "premium/model", "input": []},
            api_key="key",
        )

    assert calls == [{"model": "premium/model", "payload_model": "premium/model"}]


def test_strip_result_order_removes_internal_sort_key():
    results = [
        {"_order": 2, "report_model": "b"},
        {"_order": 1, "report_model": "a"},
    ]

    stripped = ab_report_eval.strip_result_order(results)

    assert stripped == [{"report_model": "b"}, {"report_model": "a"}]
    assert results[0]["_order"] == 2


def test_analyze_variant_reports_missing_artifacts(tmp_path):
    variant_dir = tmp_path / "missing"
    variant_dir.mkdir()
    ab_report_eval.write_json(
        variant_dir / "ab_runtime.json",
        {"status": "timeout", "alerts": ["slow"], "elapsed_seconds": 3.0},
    )

    result = ab_report_eval.analyze_variant(
        variant_dir=variant_dir,
        selected_segment_ids=["S0001"],
    )

    assert result["status"] == "timeout"
    assert result["formal_total"] == 0
    assert result["alerts"] == ["slow"]


def test_generate_paired_variants_child_reuses_current_report(monkeypatch, tmp_path):
    events = []
    current_report = SimpleNamespace(generated_by="current")
    enriched_report = SimpleNamespace(generated_by="current+memory")

    monkeypatch.setattr(ab_report_eval, "provider_patch", lambda provider, model: nullcontext())
    monkeypatch.setattr(
        ab_report_eval.reporting,
        "build_local_report",
        lambda **kwargs: SimpleNamespace(generated_by="local"),
    )

    def fake_build_ai_report(**kwargs):
        events.append(("build_ai_report", kwargs["context_pack"], kwargs["recover_base_items"]))
        return current_report

    def fake_enrich(report, **kwargs):
        events.append(("enrich", report is current_report, kwargs["context_pack"]))
        return enriched_report

    def fake_write_artifacts(*, output_dir, report, requested_ai, make_pdf):
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = "current" if report is current_report else "enriched"
        ab_report_eval.write_json(
            output_dir / "report.json",
            {
                "decisions": [{"text": marker, "citations": ["S0001"]}],
                "action_items": [],
                "open_questions": [],
                "risks": [],
                "sections": [],
                "timeline": [],
            },
        )
        ab_report_eval.write_json(output_dir / "report_health.json", {"status": "ok"})
        return SimpleNamespace(
            generated_by=report.generated_by,
            status="ok",
            alerts=[],
        )

    monkeypatch.setattr(ab_report_eval.reporting, "build_ai_report", fake_build_ai_report)
    monkeypatch.setattr(
        ab_report_eval.reporting, "_enrich_ai_report_with_memory", fake_enrich
    )
    monkeypatch.setattr(
        ab_report_eval.reporting,
        "write_report_artifacts_from_report",
        fake_write_artifacts,
    )

    ab_report_eval.generate_paired_variants_child(
        case_dir=tmp_path,
        case_payload={
            "meeting_id": "m1",
            "title": "Meeting",
            "source_filename": "audio.m4a",
        },
        segments=[
            {
                "segment_id": "S0001",
                "start": 0.0,
                "end": 1.0,
                "speaker": "Ilya",
                "text": "Договорились проверить paired memory.",
            }
        ],
        context_pack={"facts": [{"summary": "memory"}]},
        provider="openrouter",
        report_model="test/model",
        api_key="key",
    )

    assert events == [
        ("build_ai_report", None, True),
        ("enrich", True, {"facts": [{"summary": "memory"}]}),
    ]
    assert (tmp_path / "no_context" / "report.json").exists()
    assert (tmp_path / "with_context" / "report.json").exists()
    assert (tmp_path / "no_context" / "ab_runtime.json").exists()
    assert (tmp_path / "with_context" / "ab_runtime.json").exists()


def test_paired_timeout_does_not_overwrite_completed_variant(monkeypatch, tmp_path):
    class FakeProcess:
        exitcode = None

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            no_context = tmp_path / "no_context"
            no_context.mkdir(parents=True, exist_ok=True)
            ab_report_eval.write_json(no_context / "report.json", {})
            ab_report_eval.write_json(no_context / "report_health.json", {"status": "ok"})
            ab_report_eval.write_json(
                no_context / "ab_runtime.json",
                {"status": "ok", "elapsed_seconds": 12.0},
            )

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return True

        def terminate(self):
            return None

    monkeypatch.setattr(ab_report_eval.multiprocessing, "Process", FakeProcess)

    ab_report_eval.generate_paired_variants_with_timeout(
        case_dir=tmp_path,
        case=ab_report_eval.DEFAULT_CASES[0],
        segments=[],
        context_pack={},
        provider="openrouter",
        report_model="test/model",
        api_key="key",
        timeout_seconds=1,
    )

    no_context_runtime = json.loads(
        (tmp_path / "no_context" / "ab_runtime.json").read_text(encoding="utf-8")
    )
    with_context_runtime = json.loads(
        (tmp_path / "with_context" / "ab_runtime.json").read_text(encoding="utf-8")
    )

    assert no_context_runtime == {"status": "ok", "elapsed_seconds": 12.0}
    assert with_context_runtime["status"] == "timeout"


def test_analyze_variant_reports_quality_issues(tmp_path):
    variant_dir = tmp_path / "variant"
    variant_dir.mkdir()
    ab_report_eval.write_json(
        variant_dir / "report.json",
        {
            "decisions": [],
            "action_items": [],
            "open_questions": [],
            "risks": [
                {
                    "text": "Просто принимаем риск. О'кей, пусть он сломается, мы его отремонтируем.",
                    "citations": ["S0001"],
                }
            ],
            "sections": [
                {
                    "title": "Память: контекст прошлых встреч",
                    "summary": "Фоновая память без конкретных пунктов.",
                    "items": [],
                }
            ],
            "timeline": [],
        },
    )
    ab_report_eval.write_json(variant_dir / "report_health.json", {"status": "ok"})

    result = ab_report_eval.analyze_variant(
        variant_dir=variant_dir,
        selected_segment_ids=["S0001"],
    )

    assert result["raw_formal_items"] == [
        "Просто принимаем риск. О'кей, пусть он сломается, мы его отремонтируем."
    ]
    assert result["methodology_example_formal_items"] == [
        "Просто принимаем риск. О'кей, пусть он сломается, мы его отремонтируем."
    ]
    assert result["empty_memory_sections"] == 1
    assert result["quality_issues"] == [
        "raw transcript-like formal item",
        "methodology example treated as formal item",
        "memory sections without concrete items",
    ]


def test_analyze_variant_flags_synthesized_methodology_example(tmp_path):
    variant_dir = tmp_path / "variant"
    variant_dir.mkdir()
    ab_report_eval.write_json(
        variant_dir / "report.json",
        {
            "decisions": [
                {
                    "text": "Для оборудования, где ремонт дешевле, принимается риск и используется подход ремонтировать по факту отказа.",
                    "citations": ["S0001"],
                }
            ],
            "action_items": [],
            "open_questions": [],
            "risks": [],
            "sections": [],
            "timeline": [],
        },
    )
    ab_report_eval.write_json(variant_dir / "report_health.json", {"status": "ok"})

    result = ab_report_eval.analyze_variant(
        variant_dir=variant_dir,
        selected_segment_ids=["S0001"],
    )

    assert result["methodology_example_formal_items"] == [
        "Для оборудования, где ремонт дешевле, принимается риск и используется подход ремонтировать по факту отказа."
    ]
    assert result["quality_issues"] == [
        "methodology example treated as formal item"
    ]


def test_openrouter_response_cost_sums_usage(tmp_path):
    variant_dir = tmp_path / "variant"
    variant_dir.mkdir()
    ab_report_eval.write_json(
        variant_dir / "openrouter_response_01.json",
        {"usage": {"cost": 0.1}},
    )
    ab_report_eval.write_json(
        variant_dir / "openrouter_response_02.json",
        {"usage": {"cost": 0.2345678}},
    )

    assert ab_report_eval.count_openrouter_responses(variant_dir) == 2
    assert ab_report_eval.openrouter_response_cost(variant_dir) == 0.334568


def test_report_outline_includes_formal_items_and_sections():
    report = {
        "decisions": [{"title": "Use Bronze for raw files", "citations": ["S0101"]}],
        "action_items": [{"text": "Add Bronze database", "citations": ["S0102"]}],
        "open_questions": [],
        "risks": [{"summary": "Schema gap", "citations": []}],
        "sections": [{"title": "Memory context"}],
    }

    outline = ab_report_eval.report_outline(report)

    assert outline == [
        "decision [S0101]: Use Bronze for raw files",
        "task [S0102]: Add Bronze database",
        "risk: Schema gap",
        "section: Memory context",
    ]


def test_report_outline_uses_protocol_ref_when_citations_are_absent():
    report = {
        "decisions": [{"text": "Use Bronze for raw files", "ref": "S0101 · 13:16"}],
        "tasks": [],
        "questions": [],
        "risks": [],
        "sections": [],
    }

    assert ab_report_eval.report_outline(report) == [
        "decision [S0101 · 13:16]: Use Bronze for raw files"
    ]


def test_formal_item_refs_reads_citations_and_protocol_refs():
    report = {
        "decisions": [{"text": "Decision", "ref": "S0101 · 13:16"}],
        "tasks": [{"text": "Task", "citations": ["S0102"]}],
        "questions": [],
        "risks": [{"text": "Risk", "ref": "S0103 13:20-13:25"}],
    }

    assert ab_report_eval.formal_item_refs(report) == {"S0101", "S0102", "S0103"}


def test_semantic_formal_diff_detects_lost_and_matched_items():
    diff = ab_report_eval.semantic_formal_diff(
        [
            "Add Bronze database for uploaded files",
            "Use server account to rent infrastructure",
        ],
        [
            "Create a Bronze level database for uploaded files",
            "Unrelated note about meeting memory",
        ],
    )

    assert diff["semantic_formal_recall"] == 0.5
    assert len(diff["semantic_formal_matches"]) == 1
    assert diff["semantic_formal_lost"] == [
        "Use server account to rent infrastructure"
    ]


def test_should_judge_comparison_selects_borderline_pairs():
    assert ab_report_eval.should_judge_comparison(
        {
            "verdict": "pass",
            "formal_ref_recall": 1.0,
            "semantic_formal_recall": 0.95,
        }
    ) is False
    assert ab_report_eval.should_judge_comparison(
        {
            "verdict": "pass",
            "formal_ref_recall": 0.7,
            "semantic_formal_recall": 0.95,
        }
    ) is True
    assert ab_report_eval.should_judge_comparison(
        {
            "verdict": "review: lost formal evidence",
            "formal_ref_recall": 1.0,
            "semantic_formal_recall": 1.0,
        }
    ) is True


def test_judge_comparison_uses_provider_response(monkeypatch, tmp_path):
    calls = []

    def fake_post_provider_response(*, provider, model, payload, api_key):
        calls.append(
            {
                "provider": provider,
                "model": model,
                "payload_model": payload["model"],
                "api_key": api_key,
            }
        )
        return {
            "output_text": (
                '{"verdict":"with_context_worse",'
                '"current_fact_preservation":2,'
                '"memory_value":3,'
                '"overreach_risk":4,'
                '"rationale":"Context dropped current tasks.",'
                '"missing_current_facts":["Bronze task"],'
                '"useful_memory_additions":["Prior data-layer terminology"]}'
            )
        }

    monkeypatch.setattr(
        ab_report_eval, "post_provider_response", fake_post_provider_response
    )

    result = ab_report_eval.judge_comparison(
        case=ab_report_eval.DEFAULT_CASES[0],
        genre="project",
        provider="openrouter",
        report_model="google/gemini-3.1-pro-preview",
        no_context={
            "status": "ok",
            "formal_total": 2,
            "formal_refs": ["S0101"],
            "outline": ["task [S0101]: Add Bronze database"],
        },
        with_context={
            "status": "ok",
            "formal_total": 1,
            "formal_refs": [],
            "outline": ["section: Memory: data layer"],
        },
        comparison={
            "verdict": "review: lost formal evidence",
            "score": 8,
            "formal_delta": -1,
            "sections_delta": 1,
            "formal_ref_recall": 0.0,
            "semantic_formal_recall": 0.0,
            "formal_refs_lost": ["S0101"],
            "formal_refs_gained": [],
            "semantic_formal_lost": ["Add Bronze database"],
        },
        judge_provider="openrouter",
        judge_model="openai/gpt-5.5",
        judge_api_key="test-key",
        output_dir=tmp_path,
    )

    assert calls == [
        {
            "provider": "openrouter",
            "model": "openai/gpt-5.5",
            "payload_model": "openai/gpt-5.5",
            "api_key": "test-key",
        }
    ]
    assert result["status"] == "ok"
    assert result["verdict"] == "with_context_worse"
    assert result["current_fact_preservation"] == 2
    assert result["missing_current_facts"] == ["Bronze task"]
    assert (tmp_path / "judge.json").exists()


def test_judge_comparison_ignores_stale_cache(monkeypatch, tmp_path):
    ab_report_eval.write_json(
        tmp_path / "judge.json",
        {
            "status": "ok",
            "judge_model": "openrouter:openai/gpt-5.5",
            "input_hash": "old",
            "verdict": "with_context_worse",
        },
    )
    calls = []

    def fake_post_provider_response(*, provider, model, payload, api_key):
        calls.append(payload)
        return {
            "output_text": (
                '{"verdict":"with_context_better",'
                '"current_fact_preservation":5,'
                '"memory_value":4,'
                '"overreach_risk":1,'
                '"rationale":"Fresh judge input.",'
                '"missing_current_facts":[],'
                '"useful_memory_additions":["Memory sidecar"]}'
            )
        }

    monkeypatch.setattr(
        ab_report_eval, "post_provider_response", fake_post_provider_response
    )

    result = ab_report_eval.judge_comparison(
        case=ab_report_eval.DEFAULT_CASES[0],
        genre="project",
        provider="openrouter",
        report_model="google/gemini-3.1-pro-preview",
        no_context={
            "status": "ok",
            "formal_total": 1,
            "formal_refs": ["S0101"],
            "outline": ["decision [S0101]: Keep current fact"],
        },
        with_context={
            "status": "ok",
            "formal_total": 1,
            "formal_refs": ["S0101"],
            "outline": ["decision [S0101]: Keep current fact"],
        },
        comparison={
            "verdict": "pass",
            "score": 10,
            "formal_delta": 0,
            "sections_delta": 3,
            "formal_ref_recall": 1.0,
            "semantic_formal_recall": 1.0,
            "formal_refs_lost": [],
            "formal_refs_gained": [],
            "semantic_formal_lost": [],
        },
        judge_provider="openrouter",
        judge_model="openai/gpt-5.5",
        judge_api_key="test-key",
        output_dir=tmp_path,
    )

    assert len(calls) == 1
    assert result["verdict"] == "with_context_better"
    assert result["input_hash"] != "old"


def test_case_genre_detects_methodology_case():
    toir_case = next(
        case for case in ab_report_eval.DEFAULT_CASES if case.name == "toir_methodology"
    )
    assert ab_report_eval.case_genre(toir_case) == "methodology"
    assert ab_report_eval.case_genre(ab_report_eval.DEFAULT_CASES[0]) == "project"


def test_default_cases_include_extended_project_coverage():
    names = {case.name for case in ab_report_eval.DEFAULT_CASES}

    assert {"data_layer_priority", "task_board_planning"} <= names
    assert ab_report_eval.case_genre(
        next(case for case in ab_report_eval.DEFAULT_CASES if case.name == "data_layer_priority")
    ) == "project"


def test_report_list_accepts_report_to_dict_and_protocol_aliases():
    report = {
        "action_items": [{"text": "canonical"}],
        "tasks": [{"text": "protocol"}],
        "open_questions": [{"text": "canonical question"}],
        "questions": [{"text": "protocol question"}],
    }

    assert ab_report_eval.report_list(report, "action_items", "tasks") == [
        {"text": "canonical"}
    ]
    assert ab_report_eval.report_list(report, "open_questions", "questions") == [
        {"text": "canonical question"}
    ]
    assert ab_report_eval.report_list({"tasks": [{"text": "protocol"}]}, "action_items", "tasks") == [
        {"text": "protocol"}
    ]
