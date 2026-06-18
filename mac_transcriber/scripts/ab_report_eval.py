#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mac_transcriber import memory_db, reporting
from mac_transcriber.reporting import write_report_artifacts


DEFAULT_REPORT_MODEL = "gpt-5-mini"
DEFAULT_PROVIDER = "openai"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUTH_KEY_URL = "https://openrouter.ai/api/v1/auth/key"
DEFAULT_OUTPUT_ROOT = ".local/report_ab_eval"
SEGMENT_NUMBER_RE = re.compile(r"(\d+)")
SEGMENT_REF_RE = re.compile(r"\bS\d{4}\b")
METHODOLOGY_SIGNAL_RE = re.compile(
    r"метод|стратег|тоир|анализ|над[её]жн|обслужив|риск|экономич|границ|критери",
    re.IGNORECASE,
)
USABLE_STATUSES = {"ok", "degraded"}
RAW_TRANSCRIPT_SIGNAL_RE = re.compile(
    r"\b(о['’]?[кк]ей|ну|вот|короче|как бы|типа|просто|давайте|слушай|смотри|то есть)\b",
    re.IGNORECASE,
)
METHODOLOGY_EXAMPLE_AS_RISK_RE = re.compile(
    r"пусть\s+\w*\s*слом|принима\w*\s+риск|прим(ем|имаем)\s+риск|ремонтир\w*\s+по\s+факту\s+отказ|в\s+\d+\s+раз\s+дешевле|процесс\s+от\s+этого\s+не\s+встан",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvalCase:
    name: str
    meeting_id: str
    title: str
    source_filename: str
    start_segment: int
    end_segment: int
    notes: str


DEFAULT_CASES = [
    EvalCase(
        name="bronze_db",
        meeting_id="zoom_z49_oUdiQB_adkDakdbWyQ__",
        title="База данных: bronze/silver/gold и хранение загруженных файлов",
        source_filename="audio.m4a",
        start_segment=101,
        end_segment=118,
        notes="Контрольный кейс: prior_context не должен стирать решения/задачи текущего фрагмента.",
    ),
    EvalCase(
        name="techcard_equipment",
        meeting_id="zoom_z49_oUdiQB_adkDakdbWyQ__",
        title="База данных: техкарты, оборудование и пользовательские действия",
        source_filename="audio.m4a",
        start_segment=195,
        end_segment=210,
        notes="Проверяет, помогает ли память связать техкарты, оборудование и аудит действий.",
    ),
    EvalCase(
        name="toir_methodology",
        meeting_id="zoom_yGUen4FMRP6R4sxPYrMzvA__",
        title="Анализ надежности и ТОиР: стратегия обслуживания",
        source_filename="audio.m4a",
        start_segment=278,
        end_segment=290,
        notes="Методологический/обучающий фрагмент: контекст не должен превращать лекцию в ложные задачи.",
    ),
    EvalCase(
        name="data_layer_priority",
        meeting_id="zoom_wRWdxo7GRYunLctkPgebYA__",
        title="Технический синк: приоритет data-layer и подготовка решений",
        source_filename="audio.m4a",
        start_segment=14,
        end_segment=45,
        notes="Проверяет приоритеты, ownership, административные задачи и подготовку технических решений.",
    ),
    EvalCase(
        name="task_board_planning",
        meeting_id="zoom_m7y0E8c5RUWcc_z8sroMIA__",
        title="Планирование задач: оценка, даты и доска работ",
        source_filename="audio.m4a",
        start_segment=18,
        end_segment=45,
        notes="Операционный planning-кейс: задачи, статусы, старт работ и перемещение по доске.",
    ),
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_files = args.env_file if args.env_file is not None else [Path(".env.local")]
    env = load_env_files(env_files)
    os.environ.update(env)
    # Резолвим дефолтную модель ПОСЛЕ загрузки .env.local, чтобы учитывался
    # MAC_TRANSCRIBER_REPORT_MODEL из файла, а не только дефолт gpt-5-mini.
    if not args.model:
        args.model = [
            os.environ.get("MAC_TRANSCRIBER_REPORT_MODEL", DEFAULT_REPORT_MODEL)
        ]
    database_url = memory_db.database_url_from_env(env)
    if not database_url:
        raise SystemExit("MAC_TRANSCRIBER_DATABASE_URL is not configured")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    cases = selected_cases(args.case)
    if args.limit:
        cases = cases[: args.limit]
    jobs: list[dict[str, Any]] = []
    judge_provider = ""
    judge_model = ""
    judge_api_key = ""
    if args.judge_model:
        judge_provider, judge_model = parse_model_spec(args.judge_model)
        judge_api_key = provider_api_key(judge_provider, env)
        if not judge_api_key:
            raise SystemExit(
                f"{provider_api_key_name(judge_provider)} is not configured"
            )
        if not args.skip_preflight:
            preflight_provider(judge_provider, judge_api_key)
    for spec_index, spec in enumerate(args.model):
        provider, report_model = parse_model_spec(spec)
        api_key = provider_api_key(provider, env)
        if not api_key:
            raise SystemExit(f"{provider_api_key_name(provider)} is not configured")
        if not args.skip_preflight:
            preflight_provider(provider, api_key)
        model_root = output_root / safe_name(f"{provider}_{report_model}")
        for case_index, case in enumerate(cases):
            jobs.append(
                {
                    "order": spec_index * len(cases) + case_index,
                    "case": case,
                    "database_url": database_url,
                    "api_key": api_key,
                    "output_root": model_root,
                    "provider": provider,
                    "report_model": report_model,
                    "context_limit": args.context_limit,
                    "variant_timeout": args.variant_timeout,
                    "force": args.force,
                    "paired_memory": args.paired_memory,
                    "judge_provider": judge_provider,
                    "judge_model": judge_model,
                    "judge_api_key": judge_api_key,
                    "judge_all": args.judge_all,
                }
            )

    if args.parallel <= 1:
        for job in jobs:
            case = job["case"]
            provider = job["provider"]
            report_model = job["report_model"]
            print(
                f"== {provider}:{report_model} / {case.name}: {case.title}", flush=True
            )
            case_result = run_case_job(job)
            results.append(case_result)
            results.sort(key=lambda item: item["_order"])
            write_json(
                output_root / "summary.json", {"results": strip_result_order(results)}
            )
            write_summary_markdown(
                output_root / "summary.md", strip_result_order(results)
            )
            print_case_result(case_result)
    else:
        print(
            f"Running {len(jobs)} model/case jobs with parallel={args.parallel}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_by_order = {}
            for job in jobs:
                case = job["case"]
                print(
                    f"== queued {job['provider']}:{job['report_model']} / {case.name}: {case.title}",
                    flush=True,
                )
                future_by_order[executor.submit(run_case_job, job)] = job["order"]
            for future in as_completed(future_by_order):
                case_result = future.result()
                results.append(case_result)
                results.sort(key=lambda item: item["_order"])
                write_json(
                    output_root / "summary.json",
                    {"results": strip_result_order(results)},
                )
                write_summary_markdown(
                    output_root / "summary.md", strip_result_order(results)
                )
                print_case_result(case_result)

    results.sort(key=lambda item: item["_order"])
    write_json(output_root / "summary.json", {"results": strip_result_order(results)})
    write_summary_markdown(output_root / "summary.md", strip_result_order(results))
    print(f"\nsummary: {output_root / 'summary.md'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real A/B report generation: no prior context vs meeting-memory context."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        action="append",
        # default=None + резолв ниже: иначе action="append" дописывает к дефолту,
        # и .env.local нельзя исключить (классический argparse-gotcha).
        default=None,
        help="Env file to load; may be repeated. Defaults to .env.local if omitted.",
    )
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model to test; may be repeated. Use 'gpt-5-mini' for OpenAI or "
            "'openrouter:provider/model' for OpenRouter."
        ),
    )
    parser.add_argument("--context-limit", type=int, default=3)
    parser.add_argument("--variant-timeout", type=int, default=240)
    parser.add_argument(
        "--case", action="append", choices=[case.name for case in DEFAULT_CASES]
    )
    parser.add_argument(
        "--limit", type=int, help="Only run the first N selected cases."
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing variant artifacts."
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Run model/case jobs concurrently. Each job still runs no_context and with_context sequentially.",
    )
    parser.add_argument(
        "--judge-model",
        help=(
            "Optional model for independent quality judging. Use 'openrouter:provider/model' "
            "or an OpenAI model name. Disabled by default."
        ),
    )
    parser.add_argument(
        "--judge-all",
        action="store_true",
        help="Judge every pair instead of only borderline/regression pairs.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip provider auth preflight checks and let real model calls report errors.",
    )
    parser.add_argument(
        "--paired-memory",
        action="store_true",
        help=(
            "Generate the current report once, save it as no_context, then add memory "
            "enrichment to the same report for with_context. This isolates memory impact "
            "from run-to-run model variation."
        ),
    )
    # Дефолтную модель НЕ резолвим здесь: на этот момент .env.local ещё не загружен в
    # os.environ, иначе MAC_TRANSCRIBER_REPORT_MODEL из .env.local игнорируется. Резолв — в main().
    return parser.parse_args(argv)


def run_case_job(job: dict[str, Any]) -> dict[str, Any]:
    result = run_case(
        case=job["case"],
        database_url=job["database_url"],
        api_key=job["api_key"],
        output_root=job["output_root"],
        provider=job["provider"],
        report_model=job["report_model"],
        context_limit=job["context_limit"],
        variant_timeout=job["variant_timeout"],
        force=job["force"],
        paired_memory=bool(job.get("paired_memory")),
        judge_provider=job.get("judge_provider", ""),
        judge_model=job.get("judge_model", ""),
        judge_api_key=job.get("judge_api_key", ""),
        judge_all=bool(job.get("judge_all")),
    )
    result["_order"] = job["order"]
    return result


def strip_result_order(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for result in results:
        copy_result = dict(result)
        copy_result.pop("_order", None)
        stripped.append(copy_result)
    return stripped


def print_case_result(case_result: dict[str, Any]) -> None:
    no_context = case_result["variants"]["no_context"]
    with_context = case_result["variants"]["with_context"]
    print(
        f"   done {case_result['provider']}:{case_result['report_model']} / {case_result['case']['name']} "
        f"no_context formal={no_context.get('formal_total')} status={no_context.get('status')} "
        f"with_context formal={with_context.get('formal_total')} status={with_context.get('status')} "
        f"delta={case_result['comparison'].get('formal_delta')} "
        f"judge={case_result['comparison'].get('judge', {}).get('verdict', '')}",
        flush=True,
    )


def selected_cases(names: list[str] | None) -> list[EvalCase]:
    if not names:
        return list(DEFAULT_CASES)
    wanted = set(names)
    return [case for case in DEFAULT_CASES if case.name in wanted]


def run_case(
    *,
    case: EvalCase,
    database_url: str,
    api_key: str,
    output_root: Path,
    provider: str,
    report_model: str,
    context_limit: int,
    variant_timeout: int,
    force: bool,
    paired_memory: bool = False,
    judge_provider: str = "",
    judge_model: str = "",
    judge_api_key: str = "",
    judge_all: bool = False,
) -> dict[str, Any]:
    case_dir = output_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    segments = load_case_segments(database_url, case)
    write_json(case_dir / "input_segments.json", segments)
    # Как и production-путь (asr.py): семантический запрос строим из контента встречи,
    # иначе A/B не измеряет реальное поведение поиска контекста по смыслу.
    context_pack = memory_db.build_report_context_pack(
        database_url,
        meeting_id=case.meeting_id,
        title=case.title,
        source_filename=case.source_filename,
        limit=context_limit,
        query_text=memory_db.context_query_text(
            case.title, [str(row.get("text", "")) for row in segments]
        ),
    )
    write_json(case_dir / "context_pack.json", context_pack)

    if paired_memory:
        no_context_dir = case_dir / "no_context"
        with_context_dir = case_dir / "with_context"
        paired_complete = all(
            (variant_dir / "report.json").exists()
            and (variant_dir / "report_health.json").exists()
            for variant_dir in (no_context_dir, with_context_dir)
        )
        if force or not paired_complete:
            generate_paired_variants_with_timeout(
                case_dir=case_dir,
                case=case,
                segments=segments,
                context_pack=context_pack,
                provider=provider,
                report_model=report_model,
                api_key=api_key,
                timeout_seconds=variant_timeout,
            )
    variants: dict[str, dict[str, Any]] = {}
    for variant_name, context in (
        ("no_context", None),
        ("with_context", context_pack),
    ):
        variant_dir = case_dir / variant_name
        complete = (variant_dir / "report.json").exists() and (
            variant_dir / "report_health.json"
        ).exists()
        if not paired_memory and (force or not complete):
            generate_variant_with_timeout(
                variant_dir=variant_dir,
                case=case,
                segments=segments,
                context=context,
                provider=provider,
                report_model=report_model,
                api_key=api_key,
                timeout_seconds=variant_timeout,
            )
        variants[variant_name] = analyze_variant(
            variant_dir=variant_dir,
            selected_segment_ids=[str(row["segment_id"]) for row in segments],
        )

    genre = case_genre(case)
    comparison = compare_variants(
        variants["no_context"], variants["with_context"], genre=genre
    )
    if judge_model and (judge_all or should_judge_comparison(comparison)):
        comparison["judge"] = judge_comparison(
            case=case,
            genre=genre,
            provider=provider,
            report_model=report_model,
            no_context=variants["no_context"],
            with_context=variants["with_context"],
            comparison=comparison,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            output_dir=case_dir / "judge",
        )
    return {
        "case": case.__dict__,
        "genre": genre,
        "provider": provider,
        "report_model": report_model,
        "segment_count": len(segments),
        "context_counts": {
            "facts": len(context_pack.get("facts") or []),
            "segments": len(context_pack.get("segments") or []),
            "embedding_chunks": len(context_pack.get("embedding_chunks") or []),
        },
        "variants": variants,
        "comparison": comparison,
    }


def case_genre(case: EvalCase) -> str:
    text = " ".join([case.name, case.title, case.notes]).lower()
    if "methodology" in text or "методолог" in text or "тоир" in text:
        return "methodology"
    return "project"


def load_case_segments(database_url: str, case: EvalCase) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(database_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT segment_id, start_seconds, end_seconds, speaker, track, text
                FROM meeting_segments
                WHERE meeting_id = %s
                ORDER BY start_seconds ASC, segment_id ASC
                """,
                (case.meeting_id,),
            )
            rows = [dict(row) for row in cur.fetchall()]
    selected = [
        row
        for row in rows
        if case.start_segment
        <= segment_number(str(row.get("segment_id")))
        <= case.end_segment
    ]
    if not selected:
        raise RuntimeError(f"No segments selected for case {case.name}")
    return [
        {
            "segment_id": row["segment_id"],
            "start": float(row.get("start_seconds") or 0),
            "end": float(row.get("end_seconds") or 0),
            "speaker": row.get("speaker") or "Speaker",
            "text": row.get("text") or "",
        }
        for row in selected
    ]


def analyze_variant(
    *, variant_dir: Path, selected_segment_ids: list[str]
) -> dict[str, Any]:
    runtime_path = variant_dir / "ab_runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else {}
    )
    if not (variant_dir / "report.json").exists():
        return {
            "decisions": 0,
            "tasks": 0,
            "questions": 0,
            "risks": 0,
            "sections": 0,
            "timeline": 0,
            "formal_total": 0,
            "status": runtime.get("status", "missing"),
            "alerts": runtime.get("alerts") or [],
            "generated_by": runtime.get("generated_by"),
            "elapsed_seconds": runtime.get("elapsed_seconds"),
            "markdown_bytes": 0,
            "report_path": "",
            "refs_total": 0,
            "invalid_refs": [],
            "ai_call_count": count_openrouter_responses(variant_dir),
            "ai_cost": openrouter_response_cost(variant_dir),
            "memory_sections": 0,
            "methodology_sections": 0,
            "memory_item_count": 0,
            "empty_memory_sections": 0,
            "formal_overreach": False,
            "raw_formal_items": [],
            "methodology_example_formal_items": [],
            "quality_issues": [],
            "paired_memory": bool(runtime.get("paired_memory")),
            "formal_refs": [],
            "formal_texts": [],
            "section_terms": [],
            "outline": [],
        }
    report = json.loads((variant_dir / "report.json").read_text(encoding="utf-8"))
    # report.json пишется раньше report_health.json/report.md и неатомарно: упавший
    # или прерванный по таймауту дочерний процесс может оставить частичные артефакты.
    health_path = variant_dir / "report_health.json"
    health = (
        json.loads(health_path.read_text(encoding="utf-8"))
        if health_path.exists()
        else {}
    )
    markdown_path = variant_dir / "report.md"
    markdown_exists = markdown_path.exists()
    refs = report_refs(report)
    selected = set(selected_segment_ids)
    invalid_refs = sorted({ref for ref in refs if ref not in selected})
    counts = {
        "decisions": len(report_list(report, "decisions")),
        "tasks": len(report_list(report, "action_items", "tasks")),
        "questions": len(report_list(report, "open_questions", "questions")),
        "risks": len(report_list(report, "risks")),
        "sections": len(report_list(report, "sections")),
        "timeline": len(report_list(report, "timeline")),
    }
    formal_total = (
        counts["decisions"] + counts["tasks"] + counts["questions"] + counts["risks"]
    )
    sections = report_list(report, "sections")
    section_titles = [item_summary_text(section, max_chars=400) for section in sections]
    memory_section_items = [
        section
        for section in sections
        if is_memory_section_title(item_summary_text(section, max_chars=400))
    ]
    memory_sections = len(memory_section_items)
    memory_item_count = sum(
        len(section.get("items") or [])
        for section in memory_section_items
        if isinstance(section, dict)
    )
    empty_memory_sections = sum(
        1
        for section in memory_section_items
        if isinstance(section, dict) and not (section.get("items") or [])
    )
    methodology_sections = sum(
        1 for title in section_titles if METHODOLOGY_SIGNAL_RE.search(title)
    )
    formal_refs = formal_item_refs(report)
    formal_texts = formal_item_texts(report)
    raw_formal_items = [
        item["text"]
        for item in formal_items(report)
        if looks_like_raw_transcript_formal_item(item["text"])
    ][:5]
    methodology_example_formal_items = [
        item["text"]
        for item in formal_items(report)
        if looks_like_methodology_example_as_formal_item(item["text"])
    ][:5]
    quality_issues = variant_quality_issues(
        raw_formal_items=raw_formal_items,
        methodology_example_formal_items=methodology_example_formal_items,
        memory_sections=memory_sections,
        memory_item_count=memory_item_count,
    )
    section_terms = section_title_terms(section_titles)
    status = health.get("status") or runtime.get("status") or "partial"
    return {
        **counts,
        "formal_total": formal_total,
        "status": status,
        "alerts": health.get("alerts") or runtime.get("alerts") or [],
        "generated_by": health.get("generated_by") or runtime.get("generated_by"),
        "provider": runtime.get("provider"),
        "elapsed_seconds": runtime.get("elapsed_seconds"),
        "markdown_bytes": markdown_path.stat().st_size if markdown_exists else 0,
        "report_path": str(markdown_path.resolve()) if markdown_exists else "",
        "refs_total": len(refs),
        "invalid_refs": invalid_refs,
        "ai_call_count": count_openrouter_responses(variant_dir),
        "ai_cost": openrouter_response_cost(variant_dir),
        "memory_sections": memory_sections,
        "methodology_sections": methodology_sections,
        "memory_item_count": memory_item_count,
        "empty_memory_sections": empty_memory_sections,
        "formal_overreach": counts["tasks"] > 0
        or bool(methodology_example_formal_items),
        "raw_formal_items": raw_formal_items,
        "methodology_example_formal_items": methodology_example_formal_items,
        "quality_issues": quality_issues,
        "paired_memory": bool(runtime.get("paired_memory")),
        "formal_refs": sorted(formal_refs),
        "formal_texts": formal_texts,
        "section_terms": sorted(section_terms),
        "outline": report_outline(report),
    }


def compare_variants(
    no_context: dict[str, Any], with_context: dict[str, Any], *, genre: str = "project"
) -> dict[str, Any]:
    elapsed_delta = None
    if (
        with_context.get("elapsed_seconds") is not None
        and no_context.get("elapsed_seconds") is not None
    ):
        elapsed_delta = round(
            _number_or_none(with_context.get("elapsed_seconds"))
            - _number_or_none(no_context.get("elapsed_seconds")),
            3,
        )
    comparison = {
        "formal_delta": with_context["formal_total"] - no_context["formal_total"],
        "sections_delta": with_context["sections"] - no_context["sections"],
        "markdown_bytes_delta": with_context["markdown_bytes"]
        - no_context["markdown_bytes"],
        "elapsed_seconds_delta": elapsed_delta,
        "no_context_status": no_context.get("status"),
        "with_context_status": with_context.get("status"),
        "with_context_has_invalid_refs": bool(with_context["invalid_refs"]),
        "with_context_thin_formal": with_context["formal_total"] < 3,
        "genre": genre,
        "with_context_formal_overreach": bool(
            genre == "methodology" and with_context.get("formal_overreach")
        ),
        "methodology_sections_delta": (
            with_context.get("methodology_sections", 0)
            - no_context.get("methodology_sections", 0)
        ),
        "memory_sections_delta": (
            with_context.get("memory_sections", 0)
            - no_context.get("memory_sections", 0)
        ),
        "with_context_quality_issues": with_context.get("quality_issues") or [],
        "quality_issues_added": sorted(
            set(with_context.get("quality_issues") or [])
            - set(no_context.get("quality_issues") or [])
        ),
        "paired_memory": bool(
            no_context.get("paired_memory") and with_context.get("paired_memory")
        ),
        "raw_formal_items_added": max(
            0,
            len(with_context.get("raw_formal_items") or [])
            - len(no_context.get("raw_formal_items") or []),
        ),
        "methodology_example_formal_items_added": max(
            0,
            len(with_context.get("methodology_example_formal_items") or [])
            - len(no_context.get("methodology_example_formal_items") or []),
        ),
    }
    no_formal_refs = set(no_context.get("formal_refs") or [])
    with_formal_refs = set(with_context.get("formal_refs") or [])
    lost_formal_refs = sorted(no_formal_refs - with_formal_refs)
    gained_formal_refs = sorted(with_formal_refs - no_formal_refs)
    comparison["formal_refs_lost"] = lost_formal_refs
    comparison["formal_refs_gained"] = gained_formal_refs
    comparison["formal_ref_recall"] = (
        1.0
        if not no_formal_refs
        else round(len(no_formal_refs & with_formal_refs) / len(no_formal_refs), 3)
    )
    no_section_terms = set(no_context.get("section_terms") or [])
    with_section_terms = set(with_context.get("section_terms") or [])
    comparison["section_terms_lost"] = sorted(no_section_terms - with_section_terms)[:8]
    comparison["section_terms_gained"] = sorted(with_section_terms - no_section_terms)[
        :8
    ]
    semantic = semantic_formal_diff(
        no_context.get("formal_texts") or [],
        with_context.get("formal_texts") or [],
    )
    comparison.update(semantic)
    baseline_quality = baseline_quality_verdict(no_context, genre=genre)
    comparison["baseline_quality_verdict"] = baseline_quality["verdict"]
    comparison["baseline_quality_score"] = baseline_quality["score"]
    comparison["baseline_quality_issues"] = baseline_quality["issues"]
    comparison["score"] = comparison_score(no_context, with_context, comparison)
    comparison["verdict"] = comparison_verdict(comparison)
    comparison["memory_impact_verdict"] = memory_impact_verdict(comparison)
    return comparison


def baseline_quality_verdict(
    variant: dict[str, Any], *, genre: str = "project"
) -> dict[str, Any]:
    issues: list[str] = []
    score = 10
    if not variant_status_usable(variant.get("status")):
        return {
            "verdict": f"fail: baseline {variant.get('status')}",
            "score": 0,
            "issues": [f"status {variant.get('status')}"],
        }
    if variant.get("invalid_refs"):
        issues.append("invalid refs")
        score -= 4
    if variant.get("quality_issues"):
        issues.extend(str(item) for item in variant.get("quality_issues") or [])
        score -= min(3, len(variant.get("quality_issues") or []))
    formal_total = int(variant.get("formal_total") or 0)
    sections = int(variant.get("sections") or 0)
    formal_refs = len(variant.get("formal_refs") or [])
    if genre == "methodology":
        methodology_sections = int(variant.get("methodology_sections") or 0)
        if sections < 2:
            issues.append("thin methodology sections")
            score -= 2
        if methodology_sections < 1:
            issues.append("missing methodology structure")
            score -= 2
        if formal_total > 4:
            issues.append("methodology over-formalized baseline")
            score -= 2
    else:
        if formal_total < 3:
            issues.append("thin formal coverage")
            score -= 3
        if formal_refs < max(1, min(formal_total, 3)):
            issues.append("thin formal evidence refs")
            score -= 2
        if sections < 2:
            issues.append("thin semantic sections")
            score -= 1
    score = max(0, min(10, score))
    if score >= 9:
        verdict = "pass"
    elif score >= 6:
        verdict = "review"
    else:
        verdict = "fail"
    if issues:
        verdict = f"{verdict}: " + ", ".join(issues[:2])
    return {"verdict": verdict, "score": score, "issues": issues}


def comparison_score(
    no_context: dict[str, Any],
    with_context: dict[str, Any],
    comparison: dict[str, Any],
) -> int:
    score = 0
    if variant_status_usable(no_context.get("status")):
        score += 1
    if variant_status_usable(with_context.get("status")):
        score += 2
    if not with_context.get("invalid_refs"):
        score += 2
    if comparison["formal_delta"] >= 0:
        score += 2
    if comparison["sections_delta"] >= 0:
        score += 1
    if comparison.get("quality_issues_added"):
        score -= min(2, len(comparison["quality_issues_added"]))
    if with_context.get("status") not in {"timeout", "failed"}:
        score += 1
    if no_context.get("status") not in {"timeout", "failed"}:
        score += 1
    return score


def comparison_verdict(comparison: dict[str, Any]) -> str:
    if not variant_status_usable(comparison.get("no_context_status")):
        return f"fail: no_context {comparison.get('no_context_status')}"
    if not variant_status_usable(comparison.get("with_context_status")):
        return f"fail: with_context {comparison.get('with_context_status')}"
    if comparison["with_context_has_invalid_refs"]:
        return "fail: invalid refs"
    if comparison.get("methodology_example_formal_items_added"):
        return "review: methodology example formalized"
    if comparison.get("raw_formal_items_added"):
        return "review: raw formal item"
    if "memory sections without concrete items" in comparison.get(
        "quality_issues_added", []
    ):
        return "review: low-value memory sections"
    if (
        comparison.get("genre") == "project"
        and comparison.get("formal_delta", 0) <= 0
        and len(comparison.get("formal_refs_lost") or []) >= 2
        and comparison.get("semantic_formal_recall", 0) < 0.75
    ):
        return "review: lost formal evidence"
    if comparison.get("genre") == "methodology" and comparison.get(
        "with_context_formal_overreach"
    ):
        return "review: methodology over-formalized"
    if (
        comparison.get("genre") == "methodology"
        and comparison.get("formal_delta", 0) < 0
        and comparison.get("semantic_formal_recall", 1.0) < 0.75
    ):
        return "review: methodology formal evidence lost"
    if (
        comparison.get("genre") == "methodology"
        and comparison.get("methodology_sections_delta", 0) >= 0
        and comparison.get("sections_delta", 0) >= 0
    ):
        if comparison["score"] >= 6:
            return "pass"
    if comparison["with_context_thin_formal"]:
        return "review: thin formal coverage"
    if comparison["formal_delta"] < 0:
        if comparison["sections_delta"] > 0:
            return "review: possible consolidation"
        return "regression: formal sections dropped"
    if comparison["sections_delta"] < 0:
        return "review: semantic sections dropped"
    if comparison["score"] >= 9:
        return "pass"
    if comparison["score"] >= 6:
        return "review"
    return "fail"


def memory_impact_verdict(comparison: dict[str, Any]) -> str:
    if not comparison.get("paired_memory"):
        return "not_paired"
    if not variant_status_usable(comparison.get("no_context_status")):
        return f"fail: baseline {comparison.get('no_context_status')}"
    if not variant_status_usable(comparison.get("with_context_status")):
        return f"fail: memory {comparison.get('with_context_status')}"
    if comparison.get("with_context_has_invalid_refs"):
        return "fail: invalid refs"
    if comparison.get("quality_issues_added"):
        return "review: memory added quality issues"
    if comparison.get("formal_delta", 0) < 0:
        return "review: memory removed formal items"
    if comparison.get("formal_ref_recall", 1.0) < 1.0:
        return "review: memory lost formal refs"
    if comparison.get("semantic_formal_recall", 1.0) < 1.0:
        return "review: memory changed formal meaning"
    if (
        comparison.get("memory_sections_delta", 0) > 0
        or comparison.get("sections_delta", 0) > 0
    ):
        return "pass: sidecar added"
    return "pass: unchanged"


def variant_status_usable(status: Any) -> bool:
    return str(status) in USABLE_STATUSES


def should_judge_comparison(comparison: dict[str, Any]) -> bool:
    if str(comparison.get("verdict", "")).startswith(("fail", "regression", "review")):
        return True
    if comparison.get("formal_ref_recall", 1.0) < 0.8:
        return True
    if comparison.get("semantic_formal_recall", 1.0) < 0.8:
        return True
    return False


def judge_comparison(
    *,
    case: EvalCase,
    genre: str,
    provider: str,
    report_model: str,
    no_context: dict[str, Any],
    with_context: dict[str, Any],
    comparison: dict[str, Any],
    judge_provider: str,
    judge_model: str,
    judge_api_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_path = output_dir / "judge.json"
    payload = judge_payload(
        case=case,
        genre=genre,
        provider=provider,
        report_model=report_model,
        no_context=no_context,
        with_context=with_context,
        comparison=comparison,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )
    input_hash = stable_payload_hash(payload)
    if cached_path.exists():
        try:
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            if (
                cached.get("judge_model") == f"{judge_provider}:{judge_model}"
                and cached.get("input_hash") == input_hash
            ):
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    previous_debug_dir = os.environ.get("MAC_TRANSCRIBER_AB_DEBUG_DIR")
    os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = str(output_dir)
    started = time.perf_counter()
    try:
        response = post_provider_response(
            provider=judge_provider,
            model=judge_model,
            payload=payload,
            api_key=judge_api_key,
        )
        # OpenAI Responses возвращает сырой JSON без output_text — текст лежит в
        # output[].content[].text. _extract_output_text покрывает оба провайдера.
        parsed = json.loads(reporting._extract_output_text(response) or "{}")
        result = {
            "status": "ok",
            "judge_model": f"{judge_provider}:{judge_model}",
            "input_hash": input_hash,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            **normalize_judge_result(parsed),
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "failed",
            "judge_model": f"{judge_provider}:{judge_model}",
            "input_hash": input_hash,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }
    finally:
        if previous_debug_dir is None:
            os.environ.pop("MAC_TRANSCRIBER_AB_DEBUG_DIR", None)
        else:
            os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = previous_debug_dir
    write_json(cached_path, result)
    return result


def stable_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def judge_payload(
    *,
    case: EvalCase,
    genre: str,
    provider: str,
    report_model: str,
    no_context: dict[str, Any],
    with_context: dict[str, Any],
    comparison: dict[str, Any],
    judge_provider: str,
    judge_model: str,
) -> dict[str, Any]:
    compact_comparison = {
        key: comparison.get(key)
        for key in (
            "verdict",
            "score",
            "formal_delta",
            "sections_delta",
            "formal_ref_recall",
            "semantic_formal_recall",
            "formal_refs_lost",
            "formal_refs_gained",
            "section_terms_lost",
            "section_terms_gained",
            "semantic_formal_lost",
            "with_context_formal_overreach",
            "methodology_sections_delta",
            "memory_sections_delta",
        )
    }
    evaluation_input = {
        "case": {
            "name": case.name,
            "title": case.title,
            "notes": case.notes,
            "genre": genre,
        },
        "generator": f"{provider}:{report_model}",
        "judge_model": f"{judge_provider}:{judge_model}",
        "metrics": compact_comparison,
        "no_context": {
            "status": no_context.get("status"),
            "formal_total": no_context.get("formal_total"),
            "formal_refs": no_context.get("formal_refs") or [],
            "outline": (no_context.get("outline") or [])[:12],
        },
        "with_context": {
            "status": with_context.get("status"),
            "formal_total": with_context.get("formal_total"),
            "formal_refs": with_context.get("formal_refs") or [],
            "memory_sections": with_context.get("memory_sections"),
            "methodology_sections": with_context.get("methodology_sections"),
            "outline": (with_context.get("outline") or [])[:12],
        },
    }
    return {
        "model": judge_model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are an independent evaluator of Russian meeting reports. "
                    "Compare no_context and with_context outputs. Reward context only when it preserves "
                    "current meeting facts and adds useful prior-memory structure. Penalize invented tasks, "
                    "lost current decisions, and over-formalized methodology discussions. Return strict JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(evaluation_input, ensure_ascii=False, indent=2),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "report_ab_judge",
                "strict": True,
                "schema": judge_schema(),
            }
        },
    }


def judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "with_context_better",
                    "roughly_equal",
                    "with_context_worse",
                    "inconclusive",
                ],
            },
            "current_fact_preservation": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
            "memory_value": {"type": "integer", "minimum": 1, "maximum": 5},
            "overreach_risk": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string"},
            "missing_current_facts": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "useful_memory_additions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
        },
        "required": [
            "verdict",
            "current_fact_preservation",
            "memory_value",
            "overreach_risk",
            "rationale",
            "missing_current_facts",
            "useful_memory_additions",
        ],
    }


def normalize_judge_result(value: dict[str, Any]) -> dict[str, Any]:
    verdict = value.get("verdict")
    if verdict not in {
        "with_context_better",
        "roughly_equal",
        "with_context_worse",
        "inconclusive",
    }:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "current_fact_preservation": bounded_int(
            value.get("current_fact_preservation"), minimum=1, maximum=5
        ),
        "memory_value": bounded_int(value.get("memory_value"), minimum=1, maximum=5),
        "overreach_risk": bounded_int(
            value.get("overreach_risk"), minimum=1, maximum=5
        ),
        "rationale": compact_text(str(value.get("rationale") or ""), max_chars=500),
        "missing_current_facts": compact_string_list(
            value.get("missing_current_facts")
        ),
        "useful_memory_additions": compact_string_list(
            value.get("useful_memory_additions")
        ),
    }


def bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return minimum
    return min(max(number, minimum), maximum)


def compact_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        compact_text(str(item), max_chars=180)
        for item in value[:5]
        if str(item).strip()
    ]


def report_refs(report: dict[str, Any]) -> list[str]:
    refs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            refs.extend(SEGMENT_REF_RE.findall(value))

    walk(report)
    return refs


def count_openrouter_responses(variant_dir: Path) -> int:
    return len(list(variant_dir.glob("openrouter_response_*.json")))


def openrouter_response_cost(variant_dir: Path) -> float | None:
    response_paths = list(variant_dir.glob("openrouter_response_*.json"))
    if not response_paths:
        return None
    total = 0.0
    for path in response_paths:
        try:
            usage = json.loads(path.read_text(encoding="utf-8")).get("usage") or {}
        except (OSError, json.JSONDecodeError):
            continue
        total += float(usage.get("cost") or 0)
    return round(total, 6)


def is_memory_section_title(title: str) -> bool:
    return str(title).strip().lower().startswith("память:")


def formal_item_refs(report: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for keys in (
        ("decisions",),
        ("action_items", "tasks"),
        ("open_questions", "questions"),
        ("risks",),
    ):
        for item in report_list(report, *keys):
            refs.update(item_ref_ids(item))
    return refs


def formal_item_texts(report: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in formal_items(report):
        if item["text"]:
            texts.append(item["text"])
    return texts


def formal_items(report: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for keys in (
        (("decisions",), "decision"),
        (("action_items", "tasks"), "task"),
        (("open_questions", "questions"), "question"),
        (("risks",), "risk"),
    ):
        key_names, label = keys
        for item in report_list(report, *key_names):
            text = item_summary_text(item, max_chars=600)
            if text:
                items.append({"kind": label, "text": text})
    return items


def looks_like_raw_transcript_formal_item(text: str) -> bool:
    value = compact_text(text, max_chars=1000).lower()
    if len(value) < 40:
        return False
    signals = RAW_TRANSCRIPT_SIGNAL_RE.findall(value)
    if re.search(r"\bо[кк]ей\b", value) and len(signals) >= 2:
        return True
    if value.startswith(("просто ", "ну ", "вот ")) and len(signals) >= 2:
        return True
    if len(signals) >= 4:
        return True
    return False


def looks_like_methodology_example_as_formal_item(text: str) -> bool:
    value = compact_text(text, max_chars=1000)
    return bool(METHODOLOGY_EXAMPLE_AS_RISK_RE.search(value))


def variant_quality_issues(
    *,
    raw_formal_items: list[str],
    methodology_example_formal_items: list[str],
    memory_sections: int,
    memory_item_count: int,
) -> list[str]:
    issues: list[str] = []
    if raw_formal_items:
        issues.append("raw transcript-like formal item")
    if methodology_example_formal_items:
        issues.append("methodology example treated as formal item")
    if memory_sections and memory_item_count == 0:
        issues.append("memory sections without concrete items")
    return issues


def item_ref_ids(item: Any) -> set[str]:
    if not isinstance(item, dict):
        return set()
    refs: set[str] = set()
    citations = item.get("citations")
    if isinstance(citations, list):
        refs.update(str(citation) for citation in citations if citation)
    ref = item.get("ref")
    if isinstance(ref, str):
        refs.update(SEGMENT_REF_RE.findall(ref))
    return refs


def semantic_formal_diff(
    no_context_texts: list[str], with_context_texts: list[str]
) -> dict[str, Any]:
    if not no_context_texts:
        return {
            "semantic_formal_recall": 1.0,
            "semantic_formal_lost": [],
            "semantic_formal_matches": [],
        }
    matches: list[dict[str, Any]] = []
    lost: list[str] = []
    for source in no_context_texts:
        best_text = ""
        best_score = 0.0
        for candidate in with_context_texts:
            score = text_similarity(source, candidate)
            if score > best_score:
                best_score = score
                best_text = candidate
        if best_score >= 0.45:
            matches.append(
                {
                    "source": compact_text(source, max_chars=120),
                    "match": compact_text(best_text, max_chars=120),
                    "score": round(best_score, 3),
                }
            )
        else:
            lost.append(compact_text(source, max_chars=160))
    return {
        "semantic_formal_recall": round(
            (len(no_context_texts) - len(lost)) / len(no_context_texts), 3
        ),
        "semantic_formal_lost": lost[:8],
        "semantic_formal_matches": matches[:8],
    }


def text_similarity(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    jaccard = overlap / len(left_tokens | right_tokens)
    return max(f1, jaccard)


def normalized_tokens(value: str) -> set[str]:
    stop_words = {
        "это",
        "как",
        "что",
        "для",
        "или",
        "там",
        "вот",
        "уже",
        "если",
        "нужно",
        "надо",
        "будет",
        "быть",
        "задача",
        "задачи",
        "решение",
        "решили",
        "риск",
    }
    tokens = set()
    for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", value.lower()):
        if token not in stop_words:
            tokens.add(token)
    return tokens


def section_title_terms(section_titles: list[str]) -> set[str]:
    terms: set[str] = set()
    stop_words = {
        "память",
        "контекст",
        "прошлых",
        "встреч",
        "раздел",
        "текущий",
        "текущая",
        "основные",
        "ключевые",
    }
    for title in section_titles:
        if is_memory_section_title(title):
            continue
        for word in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", title.lower()):
            if word not in stop_words:
                terms.add(word)
    return terms


def report_outline(report: dict[str, Any], *, limit_per_bucket: int = 3) -> list[str]:
    outline: list[str] = []
    for keys, label in (
        (("decisions",), "decision"),
        (("action_items", "tasks"), "task"),
        (("open_questions", "questions"), "question"),
        (("risks",), "risk"),
    ):
        for item in report_list(report, *keys)[:limit_per_bucket]:
            text = item_summary_text(item)
            citations = item_refs(item)
            prefix = f"{label}"
            if citations:
                prefix = f"{prefix} [{citations}]"
            outline.append(f"{prefix}: {text}")
    for section in report_list(report, "sections")[:limit_per_bucket]:
        title = item_summary_text(section)
        if title:
            outline.append(f"section: {title}")
    return outline


def report_list(report: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = report.get(key)
        if isinstance(value, list):
            return value
    return []


def item_refs(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    citations = item.get("citations")
    if isinstance(citations, list) and citations:
        return ", ".join(str(citation) for citation in citations if citation)
    ref = item.get("ref")
    return str(ref) if isinstance(ref, str) and ref.strip() else ""


def item_summary_text(item: Any, *, max_chars: int = 180) -> str:
    if not isinstance(item, dict):
        return compact_text(str(item), max_chars=max_chars)
    for key in ("title", "text", "summary", "heading"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return compact_text(value, max_chars=max_chars)
    return compact_text(json.dumps(item, ensure_ascii=False), max_chars=max_chars)


def compact_text(value: str, *, max_chars: int) -> str:
    compacted = re.sub(r"\s+", " ", value).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 1].rstrip() + "…"


def segment_number(segment_id: str) -> int:
    match = SEGMENT_NUMBER_RE.search(segment_id)
    return int(match.group(1)) if match else -1


def write_summary_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Report Memory A/B Evaluation",
        "",
        "| Model | Case | Variant | Status | Formal | Formal refs | Sections | Memory | Method | Quality | Timeline | Invalid refs | Seconds | AI calls | Cost | Report |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        report_model = f"{result['provider']}:{result['report_model']}"
        case_name = result["case"]["name"]
        for variant_name in ("no_context", "with_context"):
            variant = result["variants"][variant_name]
            invalid = (
                ", ".join(variant["invalid_refs"]) if variant["invalid_refs"] else ""
            )
            seconds = (
                ""
                if variant.get("elapsed_seconds") is None
                else str(variant["elapsed_seconds"])
            )
            cost = "" if variant.get("ai_cost") is None else f"{variant['ai_cost']:.6f}"
            lines.append(
                "| "
                f"{report_model} | {case_name} | {variant_name} | {variant['status']} | {variant['formal_total']} | "
                f"{len(variant.get('formal_refs') or [])} | "
                f"{variant['sections']} | {variant.get('memory_sections', 0)} | {variant.get('methodology_sections', 0)} | "
                f"{', '.join(variant.get('quality_issues') or [])} | "
                f"{variant['timeline']} | {invalid} | "
                f"{seconds} | {variant.get('ai_call_count', 0)} | {cost} | {variant['report_path']} |"
            )
        comparison = result["comparison"]
        judge = comparison.get("judge") or {}
        judge_verdict = judge.get("verdict", "")
        judge_scores = ""
        if judge:
            judge_scores = (
                f"; judge={judge_verdict}"
                f" preserve={judge.get('current_fact_preservation', '')}"
                f" memory={judge.get('memory_value', '')}"
                f" risk={judge.get('overreach_risk', '')}"
            )
        lines.append(
            "| "
            f"{report_model} | {case_name} | delta: {comparison['verdict']} ({comparison['score']}/10); baseline: {comparison.get('baseline_quality_verdict', '')} ({comparison.get('baseline_quality_score', '')}/10); memory: {comparison.get('memory_impact_verdict', '')}{judge_scores} |  | {comparison['formal_delta']} | "
            f"{comparison.get('formal_ref_recall', '')} ref / {comparison.get('semantic_formal_recall', '')} sem | "
            f"{comparison['sections_delta']} | {comparison.get('memory_sections_delta', '')} | {comparison.get('methodology_sections_delta', '')} | "
            f"{', '.join(comparison.get('quality_issues_added') or [])} |  | "
            f"{'bad refs' if comparison['with_context_has_invalid_refs'] else ''} | "
            f"{comparison['elapsed_seconds_delta'] if comparison['elapsed_seconds_delta'] is not None else ''} |  |  |  |"
        )
    lines.extend(["", "## Variant Outlines", ""])
    for result in results:
        report_model = f"{result['provider']}:{result['report_model']}"
        case_name = result["case"]["name"]
        lines.append(f"### {report_model} / {case_name}")
        for variant_name in ("no_context", "with_context"):
            variant = result["variants"][variant_name]
            lines.append("")
            lines.append(f"#### {variant_name}")
            outline = variant.get("outline") or []
            if not outline:
                lines.append("- No report outline available.")
                continue
            for item in outline[:12]:
                lines.append(f"- {item}")
        comparison = result["comparison"]
        if comparison.get("formal_refs_lost") or comparison.get("formal_refs_gained"):
            lines.append("")
            lines.append("#### evidence diff")
            if comparison.get("formal_refs_lost"):
                lines.append(
                    "- lost formal refs: " + ", ".join(comparison["formal_refs_lost"])
                )
            if comparison.get("formal_refs_gained"):
                lines.append(
                    "- gained formal refs: "
                    + ", ".join(comparison["formal_refs_gained"])
                )
            if comparison.get("section_terms_lost"):
                lines.append(
                    "- lost section terms: "
                    + ", ".join(comparison["section_terms_lost"])
                )
            if comparison.get("section_terms_gained"):
                lines.append(
                    "- gained section terms: "
                    + ", ".join(comparison["section_terms_gained"])
                )
            if comparison.get("semantic_formal_lost"):
                lines.append("- semantic formal lost:")
                for lost in comparison["semantic_formal_lost"]:
                    lines.append(f"  - {lost}")
        if comparison.get("quality_issues_added"):
            lines.append("")
            lines.append("#### quality issues")
            lines.append("- added: " + ", ".join(comparison["quality_issues_added"]))
            for item in (
                result["variants"]["with_context"].get("raw_formal_items") or []
            )[:3]:
                lines.append(f"- raw formal item: {item}")
            for item in (
                result["variants"]["with_context"].get(
                    "methodology_example_formal_items"
                )
                or []
            )[:3]:
                lines.append(f"- methodology example formal item: {item}")
        judge = comparison.get("judge") or {}
        if judge:
            lines.append("")
            lines.append("#### judge")
            if judge.get("status") != "ok":
                lines.append(
                    f"- status: {judge.get('status')} ({judge.get('error', '')})"
                )
            else:
                lines.append(
                    "- verdict: "
                    f"{judge.get('verdict')} "
                    f"(preserve={judge.get('current_fact_preservation')}/5, "
                    f"memory={judge.get('memory_value')}/5, "
                    f"risk={judge.get('overreach_risk')}/5)"
                )
                if judge.get("rationale"):
                    lines.append(f"- rationale: {judge['rationale']}")
                if judge.get("missing_current_facts"):
                    lines.append("- missing current facts:")
                    for item in judge["missing_current_facts"]:
                        lines.append(f"  - {item}")
                if judge.get("useful_memory_additions"):
                    lines.append("- useful memory additions:")
                    for item in judge["useful_memory_additions"]:
                        lines.append(f"  - {item}")
        lines.append("")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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


def _number_or_none(value: Any) -> float:
    return float(value)


def generate_paired_variants_with_timeout(
    *,
    case_dir: Path,
    case: EvalCase,
    segments: list[dict[str, Any]],
    context_pack: dict[str, Any],
    provider: str,
    report_model: str,
    api_key: str,
    timeout_seconds: int,
) -> None:
    started = time.perf_counter()
    process = multiprocessing.Process(
        target=generate_paired_variants_child,
        kwargs={
            "case_dir": case_dir,
            "case_payload": case.__dict__,
            "segments": segments,
            "context_pack": context_pack,
            "provider": provider,
            "report_model": report_model,
            "api_key": api_key,
        },
    )
    process.start()
    process.join(timeout_seconds if timeout_seconds > 0 else None)
    if process.is_alive():
        process.terminate()
        process.join(5)
        elapsed = time.perf_counter() - started
        for variant_name in ("no_context", "with_context"):
            variant_dir = case_dir / variant_name
            if (variant_dir / "report.json").exists() and (
                variant_dir / "report_health.json"
            ).exists():
                continue
            write_json(
                variant_dir / "ab_runtime.json",
                {
                    "elapsed_seconds": round(elapsed, 3),
                    "provider": provider,
                    "generated_by": report_model,
                    "status": "timeout",
                    "alerts": [f"Paired variants timed out after {timeout_seconds}s"],
                    "paired_memory": True,
                },
            )
    elif process.exitcode != 0:
        elapsed = time.perf_counter() - started
        for variant_name in ("no_context", "with_context"):
            runtime_path = case_dir / variant_name / "ab_runtime.json"
            if runtime_path.exists():
                continue
            write_json(
                runtime_path,
                {
                    "elapsed_seconds": round(elapsed, 3),
                    "provider": provider,
                    "generated_by": report_model,
                    "status": "failed",
                    "alerts": [
                        f"Paired variant process exited with code {process.exitcode}"
                    ],
                    "paired_memory": True,
                },
            )


def generate_paired_variants_child(
    *,
    case_dir: Path,
    case_payload: dict[str, Any],
    segments: list[dict[str, Any]],
    context_pack: dict[str, Any],
    provider: str,
    report_model: str,
    api_key: str,
) -> None:
    no_context_dir = case_dir / "no_context"
    with_context_dir = case_dir / "with_context"
    no_context_dir.mkdir(parents=True, exist_ok=True)
    with_context_dir.mkdir(parents=True, exist_ok=True)
    previous_debug_dir = os.environ.get("MAC_TRANSCRIBER_AB_DEBUG_DIR")
    try:
        with provider_patch(provider, report_model):
            base_report = reporting.build_local_report(
                meeting_id=str(case_payload["meeting_id"]),
                title=str(case_payload["title"]),
                source_filename=str(case_payload["source_filename"]),
                model_name="v3_e2e_rnnt",
                segments=segments,
            )

            current_started = time.perf_counter()
            os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = str(no_context_dir)
            current_report = reporting.build_ai_report(
                base_report=base_report,
                report_model=report_model,
                api_key=api_key,
                context_pack=None,
                recover_base_items=True,
            )
            no_context_artifacts = reporting.write_report_artifacts_from_report(
                output_dir=no_context_dir,
                report=current_report,
                requested_ai=True,
                make_pdf=False,
            )
            write_json(
                no_context_dir / "ab_runtime.json",
                {
                    "elapsed_seconds": round(time.perf_counter() - current_started, 3),
                    "provider": provider,
                    "generated_by": no_context_artifacts.generated_by,
                    "status": no_context_artifacts.status,
                    "alerts": no_context_artifacts.alerts,
                    "paired_memory": True,
                },
            )

            memory_started = time.perf_counter()
            os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = str(with_context_dir)
            enriched_report = reporting._enrich_ai_report_with_memory(
                current_report,
                context_pack=context_pack,
                report_model=report_model,
                api_key=api_key,
                allow_openrouter_fallback=False,
            )
            with_context_artifacts = reporting.write_report_artifacts_from_report(
                output_dir=with_context_dir,
                report=enriched_report,
                requested_ai=True,
                make_pdf=False,
            )
            write_json(
                with_context_dir / "ab_runtime.json",
                {
                    "elapsed_seconds": round(time.perf_counter() - memory_started, 3),
                    "provider": provider,
                    "generated_by": with_context_artifacts.generated_by,
                    "status": with_context_artifacts.status,
                    "alerts": with_context_artifacts.alerts,
                    "paired_memory": True,
                },
            )
    except Exception as exc:  # noqa: BLE001
        for variant_name in ("no_context", "with_context"):
            runtime_path = case_dir / variant_name / "ab_runtime.json"
            if runtime_path.exists():
                continue
            write_json(
                runtime_path,
                {
                    "elapsed_seconds": None,
                    "provider": provider,
                    "generated_by": report_model,
                    "status": "failed",
                    "alerts": [str(exc)],
                    "paired_memory": True,
                },
            )
    finally:
        if previous_debug_dir is None:
            os.environ.pop("MAC_TRANSCRIBER_AB_DEBUG_DIR", None)
        else:
            os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = previous_debug_dir


def generate_variant_with_timeout(
    *,
    variant_dir: Path,
    case: EvalCase,
    segments: list[dict[str, Any]],
    context: dict[str, Any] | None,
    provider: str,
    report_model: str,
    api_key: str,
    timeout_seconds: int,
) -> None:
    started = time.perf_counter()
    process = multiprocessing.Process(
        target=generate_variant_child,
        kwargs={
            "variant_dir": variant_dir,
            "case_payload": case.__dict__,
            "segments": segments,
            "context": context,
            "provider": provider,
            "report_model": report_model,
            "api_key": api_key,
        },
    )
    process.start()
    process.join(timeout_seconds if timeout_seconds > 0 else None)
    if process.is_alive():
        process.terminate()
        process.join(5)
        elapsed = time.perf_counter() - started
        write_json(
            variant_dir / "ab_runtime.json",
            {
                "elapsed_seconds": round(elapsed, 3),
                "provider": provider,
                "generated_by": report_model,
                "status": "timeout",
                "alerts": [f"Variant timed out after {timeout_seconds}s"],
            },
        )
    elif process.exitcode != 0 and not (variant_dir / "ab_runtime.json").exists():
        elapsed = time.perf_counter() - started
        write_json(
            variant_dir / "ab_runtime.json",
            {
                "elapsed_seconds": round(elapsed, 3),
                "provider": provider,
                "generated_by": report_model,
                "status": "failed",
                "alerts": [f"Variant process exited with code {process.exitcode}"],
            },
        )


def generate_variant_child(
    *,
    variant_dir: Path,
    case_payload: dict[str, Any],
    segments: list[dict[str, Any]],
    context: dict[str, Any] | None,
    provider: str,
    report_model: str,
    api_key: str,
) -> None:
    started = time.perf_counter()
    try:
        previous_debug_dir = os.environ.get("MAC_TRANSCRIBER_AB_DEBUG_DIR")
        os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = str(variant_dir)
        with provider_patch(provider, report_model):
            artifacts = write_report_artifacts(
                output_dir=variant_dir,
                meeting_id=str(case_payload["meeting_id"]),
                title=str(case_payload["title"]),
                source_filename=str(case_payload["source_filename"]),
                model_name="v3_e2e_rnnt",
                segments=segments,
                use_ai=True,
                make_pdf=False,
                report_model=report_model,
                api_key=api_key,
                context_pack=context,
            )
        if previous_debug_dir is None:
            os.environ.pop("MAC_TRANSCRIBER_AB_DEBUG_DIR", None)
        else:
            os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = previous_debug_dir
        runtime = {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "provider": provider,
            "generated_by": artifacts.generated_by,
            "status": artifacts.status,
            "alerts": artifacts.alerts,
        }
    except Exception as exc:  # noqa: BLE001
        if "previous_debug_dir" in locals():
            if previous_debug_dir is None:
                os.environ.pop("MAC_TRANSCRIBER_AB_DEBUG_DIR", None)
            else:
                os.environ["MAC_TRANSCRIBER_AB_DEBUG_DIR"] = previous_debug_dir
        runtime = {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "provider": provider,
            "generated_by": report_model,
            "status": "failed",
            "alerts": [str(exc)],
        }
    write_json(variant_dir / "ab_runtime.json", runtime)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def parse_model_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        return DEFAULT_PROVIDER, spec
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    if provider not in {"openai", "openrouter"}:
        raise SystemExit(f"Unsupported provider in model spec: {provider}")
    if not model.strip():
        raise SystemExit(f"Model is missing in spec: {spec}")
    return provider, model.strip()


def provider_api_key(provider: str, env: dict[str, str]) -> str:
    if provider == "openrouter":
        return env.get("OPENROUTER_API_KEY", "").strip()
    return memory_db.openai_api_key_from_env(env)


def provider_api_key_name(provider: str) -> str:
    return "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"


def preflight_provider(provider: str, api_key: str) -> None:
    if provider != "openrouter":
        return
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        OPENROUTER_AUTH_KEY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"OpenRouter key preflight failed with HTTP {exc.code}: {body}"
        ) from exc
    except OSError as exc:
        raise SystemExit(f"OpenRouter key preflight failed: {exc}") from exc


@contextmanager
def provider_patch(provider: str, model: str):
    if provider == "openai" and openai_model_supports_reasoning(model):
        yield
        return
    original = reporting._post_openai_response

    def patched(*, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        selected_model = str(payload.get("model") or model)
        normalized = normalize_payload_for_provider(
            payload, provider=provider, model=selected_model
        )
        if provider == "openrouter":
            return post_openrouter_response(
                payload=normalized, api_key=api_key, model=selected_model
            )
        return original(payload=normalized, api_key=api_key)

    reporting._post_openai_response = patched
    try:
        yield
    finally:
        reporting._post_openai_response = original


def post_provider_response(
    *, provider: str, model: str, payload: dict[str, Any], api_key: str
) -> dict[str, Any]:
    normalized = normalize_payload_for_provider(payload, provider=provider, model=model)
    if provider == "openrouter":
        return post_openrouter_response(
            payload=normalized, api_key=api_key, model=model
        )
    return reporting._post_openai_response(payload=normalized, api_key=api_key)


def normalize_payload_for_provider(
    payload: dict[str, Any], *, provider: str, model: str
) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    if provider == "openrouter" or not openai_model_supports_reasoning(model):
        normalized.pop("reasoning", None)
    return normalized


def openai_model_supports_reasoning(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith(("gpt-5", "o1", "o3", "o4"))


def post_openrouter_response(
    *, payload: dict[str, Any], api_key: str, model: str
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    text_format = payload.get("text", {}).get("format", {})
    data: dict[str, Any] | None = None
    last_body = ""
    for response_format, structured_outputs in (
        (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": text_format.get("name", "meeting_report"),
                    "strict": bool(text_format.get("strict", True)),
                    "schema": text_format.get("schema", {}),
                },
            },
            True,
        ),
        ({"type": "json_object"}, False),
    ):
        messages = list(payload.get("input", []))
        if not structured_outputs:
            messages = [
                reporting._openrouter_json_object_system_message(
                    text_format.get("schema")
                ),
                *messages,
            ]
        request_payload = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
            "temperature": 0,
            # Совпадаем с production-фоллбэком, иначе дефолтный лимит модели режет
            # длинный JSON и искажает качество OpenRouter-моделей в A/B.
            "max_tokens": reporting.openrouter_max_tokens(),
        }
        if structured_outputs:
            request_payload["structured_outputs"] = True
        if "reasoning" in payload:
            request_payload["reasoning"] = payload["reasoning"]
        request = urllib.request.Request(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.gigaam.eval",
                "X-OpenRouter-Title": "GigaAM report A/B eval",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_body = exc.read().decode("utf-8", errors="replace")
            if structured_outputs and reporting._is_openrouter_grammar_error(last_body):
                continue
            raise reporting.ReportGenerationError(
                f"OpenRouter API returned HTTP {exc.code}: {last_body}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise reporting.ReportGenerationError(
                f"OpenRouter API request failed: {exc}"
            ) from exc
    if data is None:
        raise reporting.ReportGenerationError(
            f"OpenRouter API returned invalid response: {last_body}"
        )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise reporting.ReportGenerationError(
            "OpenRouter response did not contain message content"
        ) from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    output_text = str(content or "").strip()
    save_openrouter_debug_response(data=data, content=output_text)
    return {
        "output_text": output_text,
        "_provider_model": f"openrouter:{model}",
        "raw_openrouter": data,
    }


def save_openrouter_debug_response(*, data: dict[str, Any], content: str) -> None:
    debug_dir = os.environ.get("MAC_TRANSCRIBER_AB_DEBUG_DIR")
    if not debug_dir:
        return
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    index = len(list(path.glob("openrouter_response_*.json"))) + 1
    write_json(path / f"openrouter_response_{index:02d}.json", data)
    (path / f"openrouter_content_{index:02d}.txt").write_text(
        content + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
