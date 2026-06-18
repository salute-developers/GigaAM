from __future__ import annotations

import copy
import difflib
import json
import os
import re
import shutil
import subprocess
import html
import base64
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPORT_MODEL = "gpt-5.5"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_PREFIX = "openrouter:"
DEFAULT_OPENROUTER_FALLBACK_MODEL = "openai/gpt-4o-mini"
# Стримим запросы к LLM, чтобы соединение не простаивало молча на длинных
# reasoning-вызовах (иначе прокси рвёт его -> "Remote end closed connection").
# Сам SDK делает ретраи с backoff (max_retries) на обрывах/429/5xx.
AI_HTTP_MAX_RETRIES = int(os.environ.get("MAC_TRANSCRIBER_AI_HTTP_RETRIES", "4"))
AI_HTTP_TIMEOUT = float(os.environ.get("MAC_TRANSCRIBER_AI_HTTP_TIMEOUT", "600"))
# \d{4,}: id'ы генерируются с минимальной шириной 4 (:04d), но встречи с >9999
# сегментами дают S10000+, которые иначе утекали бы в prose без стрипа.
SEGMENT_REF_RE = re.compile(r"\bS\d{4,}\b")
RAW_TRANSCRIPT_FORMAL_SIGNAL_RE = re.compile(
    r"\b(о['’]?[кк]ей|ну|вот|короче|как бы|типа|просто|давайте|слушай|смотри|то есть)\b",
    re.IGNORECASE,
)
RAW_TRANSCRIPT_FILLER_TOKENS = frozenset(
    {
        "да",
        "ага",
        "угу",
        "ну",
        "вот",
        "так",
        "сейчас",
        "ладно",
        "значит",
        "э",
        "ээ",
        "мм",
        "окей",
        "ок",
        "короче",
        "типа",
        "блин",
    }
)
# Фрагменты, которые НЕ являются формальным пунктом: отрицание решения
# ("но мы не решили") и гипотетическая прямая речь ("если ты решил... тогда ты
# говоришь: «...»"). Это давало мусор в Решениях (аудит транскрипт↔отчёт).
NON_DECISION_FRAGMENT_RE = re.compile(
    r"\b(?:мы\s+|пока\s+)*не\s+(?:реши|договор|определ|выбра)"
    r"|тогда\s+ты\s+говоришь"
    r"|если\s+ты\s+(?:реши|скаж|говор)",
    re.IGNORECASE,
)
# Повтор подряд одного слова: "Вадим, Вадим", "да, да" — признак сырой речи.
REPEATED_WORD_RE = re.compile(r"\b(\w{2,})\b[\s,.:;—-]+\b\1\b", re.IGNORECASE)
# Зачины прямой речи в начале фразы. recover из локального regex-отчёта тащит
# дословный record.text, и реплики вроде «Ну, договорились...», «Хорошо. Тогда что
# решим?», «Коллеги, ...» утекают в формальные разделы. В отличие от
# RAW_TRANSCRIPT_FORMAL_SIGNAL_RE здесь ловим именно начало высказывания.
RECOVERED_OPENER_RE = re.compile(
    r"^(?:ну|вот|просто|ладно|хорошо|окей|о['’]?[кк]ей|да|так|тогда|"
    r"слушай(?:те)?|смотри(?:те)?|коллеги|ребят[а]?|друзья)\b",
    re.IGNORECASE,
)
METHODOLOGY_EXAMPLE_FORMAL_RE = re.compile(
    r"пусть\s+\w*\s*слом|принима\w*\s+риск|прим(ем|имаем)\s+риск|ремонтир\w*\s+по\s+факту\s+отказ|в\s+\d+\s+раз\s+дешевле|процесс\s+от\s+этого\s+не\s+встан",
    re.IGNORECASE,
)
MEMORY_SECTION_KINDS = {
    "memory_context",
    "prior_decisions",
    "recurring_risks",
    "open_threads_from_memory",
}
COVERAGE_STATUSES = {"covered", "supporting", "low_signal"}
LOCAL_REPORT_ITEM_LIMIT = 12
AI_DIRECT_SEGMENT_LIMIT = int(
    os.environ.get("MAC_TRANSCRIBER_AI_DIRECT_SEGMENT_LIMIT", "180")
)
AI_CHUNK_SIZE = int(os.environ.get("MAC_TRANSCRIBER_AI_CHUNK_SIZE", "80"))
AI_REQUEST_ATTEMPTS = max(
    1, int(os.environ.get("MAC_TRANSCRIBER_AI_REQUEST_ATTEMPTS", "3"))
)
AI_SYNTHESIS_CHUNK_LIMIT = max(
    1,
    int(os.environ.get("MAC_TRANSCRIBER_AI_SYNTHESIS_CHUNK_LIMIT", "24")),
)
AI_SYNTHESIS_BATCH_SIZE = max(
    1,
    int(os.environ.get("MAC_TRANSCRIBER_AI_SYNTHESIS_BATCH_SIZE", "4")),
)
PACKAGE_DIR = Path(__file__).resolve().parent
PROTOCOL_DIR = PACKAGE_DIR / "protocol"
PROTOCOL_TEMPLATE_PATH = PROTOCOL_DIR / "templates" / "protocol_template.html.j2"
PROTOCOL_FONTS_DIR = PROTOCOL_DIR / "fonts"


@dataclass
class TranscriptRecord:
    segment_id: str
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class TimelineBlock:
    title: str
    summary: str
    citations: list[str]


@dataclass
class ReportItem:
    title: str
    text: str
    citations: list[str]


@dataclass
class ActionItem:
    title: str
    text: str
    owner: str
    due: str
    citations: list[str]


@dataclass
class ReportProfile:
    kind: str
    label: str
    confidence: float
    rationale: str


@dataclass
class AdaptiveSection:
    kind: str
    title: str
    purpose: str
    summary: str
    items: list[ReportItem]
    citations: list[str]
    accent: str = "blue"


@dataclass
class CoverageEntry:
    segment_id: str
    status: str
    section_titles: list[str]
    rationale: str


@dataclass
class MeetingReport:
    meeting_id: str
    title: str
    source_filename: str
    model_name: str
    generated_by: str
    segment_count: int
    duration: float
    overview: str
    timeline: list[TimelineBlock]
    decisions: list[ReportItem]
    action_items: list[ActionItem]
    open_questions: list[ReportItem]
    risks: list[ReportItem]
    notable_quotes: list[ReportItem]
    transcript: list[TranscriptRecord]
    profile: ReportProfile
    adaptive_sections: list[AdaptiveSection]
    coverage: list[CoverageEntry]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReportArtifacts:
    json_path: Path
    markdown_path: Path
    html_path: Path
    typst_path: Path
    health_path: Path
    coverage_path: Path
    slack_summary_path: Path
    slack_text: str
    slack_files: list[dict[str, Any]]
    pdf_path: Path | None
    pdf_error: str | None = None
    generated_by: str = "local"
    status: str = "ok"
    alerts: list[str] = field(default_factory=list)


class ReportGenerationError(RuntimeError):
    pass


class ReportQuotaError(ReportGenerationError):
    """AI-провайдер отклонил запрос из-за исчерпанной квоты/биллинга ("нет денег").

    Подкласс ReportGenerationError, но НЕ должен приводить к local-фоллбэку или
    пропуску чанков: такая ошибка пролетает до сервиса, который ставит встречу на
    паузу (blocked_on_quota) до пополнения, вместо выдачи сырого local-«бреда».
    """


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "insufficient_quota" in lowered
        or "exceeded your current quota" in lowered
        or "check your plan and billing" in lowered
        or "http 402" in lowered  # Payment Required (OpenRouter)
    )


def _loads_ai_json(content: str, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        if "Invalid control character" not in str(exc):
            raise ReportGenerationError(
                f"{context} response was not valid JSON"
            ) from exc
        try:
            payload = json.loads(content, strict=False)
        except json.JSONDecodeError as fallback_exc:
            raise ReportGenerationError(
                f"{context} response was not valid JSON"
            ) from fallback_exc
    if not isinstance(payload, dict):
        raise ReportGenerationError(f"{context} response JSON must be an object")
    return payload


def build_local_report(
    *,
    meeting_id: str,
    title: str,
    source_filename: str,
    model_name: str,
    segments: list[Any],
) -> MeetingReport:
    transcript = transcript_records(segments)
    duration = max((record.end for record in transcript), default=0.0)
    profile = infer_report_profile(transcript)
    decisions = _dedupe_report_items(
        _matching_items(
            transcript,
            title_prefix="Решение",
            patterns=[
                r"\bрешили\b",
                r"\bрешено\b",
                r"\bдоговорились\b",
                r"\bсогласовали\b",
                r"\bутвердили\b",
                r"\bприняли решение\b",
                r"\bagreed\b",
                r"\bdecided\b",
            ],
        )
    )[:LOCAL_REPORT_ITEM_LIMIT]
    action_items = _dedupe_action_items(_local_action_items(transcript))[
        :LOCAL_REPORT_ITEM_LIMIT
    ]
    open_questions = _dedupe_report_items(
        [
            item
            for item in _matching_items(
                transcript,
                title_prefix="Вопрос",
                patterns=[
                    r"\?",
                    r"\bоткрыт\w*\s+вопрос\b",
                    r"\b(?:надо|нужно)\s+(?:понять|решить)\b",
                ],
            )
            if _is_local_open_question(item.text)
        ]
    )[:LOCAL_REPORT_ITEM_LIMIT]
    risks = _dedupe_report_items(
        _matching_items(
            transcript,
            title_prefix="Риск",
            patterns=[
                r"\bриск\b",
                r"\bпроблем[аы]\b",
                r"\bблокер\b",
                r"\bне успе",
                r"\bопасн",
                r"\brisk\b",
                r"\bblocker\b",
            ],
        )
    )[:LOCAL_REPORT_ITEM_LIMIT]
    notable_quotes = _notable_quotes(transcript)
    adaptive_sections = build_adaptive_sections(
        profile=profile,
        transcript=transcript,
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        risks=risks,
        notable_quotes=notable_quotes,
    )
    coverage = build_coverage(
        transcript=transcript, adaptive_sections=adaptive_sections
    )
    report = MeetingReport(
        meeting_id=meeting_id,
        title=title or meeting_id,
        source_filename=source_filename,
        model_name=model_name,
        generated_by="local",
        segment_count=len(transcript),
        duration=duration,
        overview=_local_overview(transcript),
        timeline=_local_timeline(transcript),
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        risks=risks,
        notable_quotes=notable_quotes,
        transcript=transcript,
        profile=profile,
        adaptive_sections=adaptive_sections,
        coverage=coverage,
    )
    validate_report_citations(report)
    return report


def build_report(
    *,
    meeting_id: str,
    title: str,
    source_filename: str,
    model_name: str,
    segments: list[Any],
    use_ai: bool,
    report_model: str | None = None,
    api_key: str | None = None,
    context_pack: dict[str, Any] | None = None,
) -> MeetingReport:
    local_report = build_local_report(
        meeting_id=meeting_id,
        title=title,
        source_filename=source_filename,
        model_name=model_name,
        segments=segments,
    )
    if not use_ai:
        return local_report

    resolved_model = report_model or DEFAULT_REPORT_MODEL
    try:
        ai_report = build_ai_report(
            base_report=local_report,
            report_model=resolved_model,
            api_key=api_key,
            context_pack=context_pack,
        )
    except ReportQuotaError:
        # Нет денег — НЕ отдаём сырой local-«бред»: пробрасываем выше, чтобы сервис
        # поставил встречу на паузу до пополнения.
        raise
    except ReportGenerationError as exc:
        local_report.warnings.append(f"AI report fallback: {exc}")
        # Критика на local-отчёт не запускаем: чистить regex-дамп нечем и незачем.
        return local_report
    # Финальный проход-критик (дедуп/кросс-дедуп/rewrite) — только на успешном
    # AI-отчёте, после memory-enrichment и baseline-upgrade. Выключен по умолчанию.
    return _maybe_apply_critic_pass(
        report=ai_report,
        report_model=resolved_model,
        api_key=api_key,
        allow_openrouter_fallback=api_key is None,
    )


def build_ai_report(
    *,
    base_report: MeetingReport,
    report_model: str,
    api_key: str | None = None,
    context_pack: dict[str, Any] | None = None,
    recover_base_items: bool = False,
    allow_baseline_upgrade: bool = True,
) -> MeetingReport:
    allow_openrouter_fallback = api_key is None
    key = api_key or load_openai_api_key()
    if not key:
        if not (allow_openrouter_fallback and load_openrouter_api_key()):
            raise ReportGenerationError("OPENAI_API_KEY is not configured")
        key = ""

    if context_pack:
        current_report = build_ai_report(
            base_report=base_report,
            report_model=report_model,
            api_key=api_key,
            context_pack=None,
            recover_base_items=True,
            allow_baseline_upgrade=True,
        )
        return _enrich_ai_report_with_memory(
            current_report,
            context_pack=context_pack,
            report_model=report_model,
            api_key=key,
            allow_openrouter_fallback=allow_openrouter_fallback,
        )

    if len(base_report.transcript) > AI_DIRECT_SEGMENT_LIMIT:
        return build_chunked_ai_report(
            base_report=base_report,
            report_model=report_model,
            api_key=key,
            context_pack=context_pack,
            allow_openrouter_fallback=allow_openrouter_fallback,
            recover_base_items=recover_base_items or context_pack is not None,
        )

    last_direct_error: ReportGenerationError | None = None
    for attempt in range(1, AI_REQUEST_ATTEMPTS + 1):
        try:
            report = build_direct_ai_report(
                base_report=base_report,
                report_model=report_model,
                api_key=key,
                context_pack=context_pack,
                allow_openrouter_fallback=allow_openrouter_fallback,
                recover_base_items=recover_base_items or context_pack is not None,
            )
            return _maybe_upgrade_baseline_report(
                report=report,
                base_report=base_report,
                report_model=report_model,
                api_key=api_key,
                allow_openrouter_fallback=allow_openrouter_fallback,
                allow_baseline_upgrade=allow_baseline_upgrade,
            )
        except ReportGenerationError as exc:
            last_direct_error = exc
            if attempt < AI_REQUEST_ATTEMPTS and _should_retry_direct_ai(exc):
                continue
            if not _should_retry_chunked(exc):
                raise
            break
    if last_direct_error is not None:
        report = build_chunked_ai_report(
            base_report=base_report,
            report_model=report_model,
            api_key=key,
            context_pack=context_pack,
            allow_openrouter_fallback=allow_openrouter_fallback,
            recover_base_items=recover_base_items or context_pack is not None,
        )
        return _maybe_upgrade_baseline_report(
            report=report,
            base_report=base_report,
            report_model=report_model,
            api_key=api_key,
            allow_openrouter_fallback=allow_openrouter_fallback,
            allow_baseline_upgrade=allow_baseline_upgrade,
        )
    raise ReportGenerationError("AI direct report failed without an error")


def _maybe_upgrade_baseline_report(
    *,
    report: MeetingReport,
    base_report: MeetingReport,
    report_model: str,
    api_key: str | None,
    allow_openrouter_fallback: bool,
    allow_baseline_upgrade: bool,
) -> MeetingReport:
    if not allow_baseline_upgrade:
        return report
    upgrade_model = baseline_upgrade_model()
    if not upgrade_model or upgrade_model == report_model:
        return report
    quality = baseline_report_quality(report)
    if quality["verdict"] == "pass":
        return report
    upgraded = build_ai_report(
        base_report=base_report,
        report_model=upgrade_model,
        api_key=api_key,
        context_pack=None,
        recover_base_items=True,
        allow_baseline_upgrade=False,
    )
    upgraded_quality = baseline_report_quality(upgraded)
    if upgraded_quality["score"] >= quality["score"]:
        upgraded.warnings.append(
            "AI baseline upgraded: "
            f"{report_model} -> {upgrade_model}; "
            f"{quality['verdict']} ({quality['score']}/10) -> "
            f"{upgraded_quality['verdict']} ({upgraded_quality['score']}/10)."
        )
        return upgraded
    report.warnings.append(
        "AI baseline upgrade skipped: "
        f"{report_model} -> {upgrade_model} did not improve "
        f"{quality['score']}/10 -> {upgraded_quality['score']}/10."
    )
    return report


def baseline_report_quality(report: MeetingReport) -> dict[str, Any]:
    issues: list[str] = []
    score = 10
    if any(
        _current_report_item_allowed(item) is False
        for item in report.decisions + report.open_questions + report.risks
    ):
        issues.append("formal quality issue")
        score -= 3
    if any(_current_action_item_allowed(item) is False for item in report.action_items):
        issues.append("action quality issue")
        score -= 3
    formal_total = (
        len(report.decisions)
        + len(report.action_items)
        + len(report.open_questions)
        + len(report.risks)
    )
    formal_refs = {
        citation
        for item in [*report.decisions, *report.open_questions, *report.risks]
        for citation in item.citations
    }
    formal_refs.update(
        citation for item in report.action_items for citation in item.citations
    )
    is_methodology = report.profile.kind in {
        "consultation",
        "lecture",
        "methodology",
        "training",
    } or any(
        METHODOLOGY_EXAMPLE_FORMAL_RE.search(section.summary)
        for section in report.adaptive_sections
    )
    methodology_sections = sum(
        1
        for section in report.adaptive_sections
        if re.search(
            r"метод|стратег|тоир|анализ|над[её]жн|обслужив|риск|экономич|границ|критери",
            " ".join([section.kind, section.title, section.summary]),
            re.IGNORECASE,
        )
    )
    if is_methodology:
        if len(report.adaptive_sections) < 2:
            issues.append("thin methodology sections")
            score -= 2
        if methodology_sections < 1:
            issues.append("missing methodology structure")
            score -= 2
        if formal_total > 4:
            issues.append("methodology over-formalized")
            score -= 2
    else:
        if formal_total < 3:
            issues.append("thin formal coverage")
            score -= 3
        if len(formal_refs) < max(1, min(formal_total, 3)):
            issues.append("thin formal evidence refs")
            score -= 2
        if len(report.adaptive_sections) < 2:
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


VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}


def _reasoning_effort(default: str) -> str:
    """Глубина reasoning у моделей gpt-5/o-серии.

    MAC_TRANSCRIBER_REASONING_EFFORT (minimal|low|medium|high) переопределяет
    отчётные проходы (прямой отчёт, синтез, enrichment памятью). Если не задан
    или некорректен — используется default конкретного прохода.
    """
    value = (os.environ.get("MAC_TRANSCRIBER_REASONING_EFFORT") or "").strip().lower()
    return value if value in VALID_REASONING_EFFORTS else default


def build_direct_ai_report(
    *,
    base_report: MeetingReport,
    report_model: str,
    api_key: str,
    context_pack: dict[str, Any] | None = None,
    allow_openrouter_fallback: bool = True,
    recover_base_items: bool = False,
) -> MeetingReport:
    payload = {
        "model": report_model,
        "reasoning": {"effort": _reasoning_effort("medium")},
        "input": [
            {
                "role": "system",
                "content": (
                    "Ты готовишь подробные русскоязычные отчеты по встречам. "
                    "Используй предоставленный транскрипт как источник фактов текущей встречи. "
                    "Если дан prior_context, используй его только как фон из прошлых встреч: "
                    "для связности, открытых хвостов и повторяющихся рисков. Не выдавай prior_context "
                    "за факты текущей встречи и не цитируй его segment_id как citation текущего отчета. "
                    "Сначала извлеки decisions, action_items, open_questions и risks только из "
                    "текущего транскрипта. prior_context не должен удалять или понижать такие "
                    "пункты: если в текущем транскрипте есть 'договорились', 'нужно', "
                    "'добавлю', 'не хватает', 'уточнить', сохрани это в формальных разделах "
                    "с citations текущих segment_id. "
                    "action_items создавай только для реальных next steps текущего проекта: "
                    "должен быть конкретный deliverable, изменение, проверка или commitment. "
                    "Методологические фразы вроде 'нужно понимать/учитывать/помнить', "
                    "'в отчете нужно показывать' или объяснение правил не являются задачами; "
                    "помещай их в adaptive_sections как методологию, критерии или оговорки. "
                    "Не выдумывай факты, "
                    "имена, решения, сроки или задачи. Каждый содержательный пункт "
                    "обязательно подкрепляй citation segment_id из транскрипта. "
                    "Сначала определи тип записи и выбери структуру отчета под смысл: "
                    "проектная встреча, лекция, интервью, техническое обсуждение, консультация "
                    "или общий разговор. Верни adaptive_sections под этот тип записи. "
                    "Coverage должен покрывать каждый segment_id ровно один раз: "
                    "covered, supporting или low_signal. Не пропускай сегменты, даже если они "
                    "попали только в полный транскрипт. Отчет должен быть полным: решения, задачи, "
                    "риски, вопросы, определения, примеры, аргументы и важные оговорки нельзя "
                    "сжимать до общих фраз. Если информации для раздела нет, верни пустой список."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "meeting_id": base_report.meeting_id,
                        "title": base_report.title,
                        "source_filename": base_report.source_filename,
                        "segments": [
                            asdict(record) for record in base_report.transcript
                        ],
                        **({"prior_context": context_pack} if context_pack else {}),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "meeting_report",
                "schema": ai_report_schema(),
                "strict": True,
            }
        },
    }
    response = _post_openai_response_with_context(
        payload=payload,
        api_key=api_key,
        context="AI direct report",
        allow_openrouter_fallback=allow_openrouter_fallback,
    )
    content = _extract_output_text(response)
    if not content:
        raise ReportGenerationError("OpenAI response did not contain output text")

    ai_payload = _loads_ai_json(content, context="OpenAI")
    ai_payload = _filter_payload_citations(
        ai_payload,
        {record.segment_id for record in base_report.transcript},
    )

    actual_report_model = str(response.get("_provider_model") or report_model)
    report = _merge_ai_payload(
        base_report,
        ai_payload,
        report_model=actual_report_model,
        recover_base_items=recover_base_items or context_pack is not None,
    )
    try:
        validate_report_citations(report)
    except ValueError as exc:
        raise ReportGenerationError(f"AI report failed validation: {exc}") from exc
    return report


def build_chunked_ai_report(
    *,
    base_report: MeetingReport,
    report_model: str,
    api_key: str,
    context_pack: dict[str, Any] | None = None,
    allow_openrouter_fallback: bool = True,
    recover_base_items: bool = False,
) -> MeetingReport:
    chunk_notes: list[dict[str, Any]] = []
    provider_model: str | None = None
    skipped_chunks: list[int] = []
    chunks = _chunks(base_report.transcript, size=max(1, AI_CHUNK_SIZE))
    for index, records in enumerate(chunks, start=1):
        payload = {
            "model": report_model,
            "reasoning": {"effort": "low"},
            "input": [
                {
                    "role": "system",
                    "content": (
                        "Ты аналитик встреч. Сожми один фрагмент русского транскрипта "
                        "в подробные структурированные заметки. Не делай общий отчёт. "
                        "Не теряй содержательные детали: технические сущности, таблицы, "
                        "аргументы, договорённости, риски, вопросы, ограничения и next steps. "
                        "Короткие шумовые реплики можно не выносить. Каждый пункт цитируй "
                        "только точными segment_id из входного фрагмента. Ограничения: summary до "
                        "900 знаков; key_points до 10 пунктов; decisions до 6; action_items до 8; "
                        "open_questions до 8; risks до 6; notable_quotes до 3. Текст каждого пункта "
                        "1-2 плотных предложения без воды."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "meeting_id": base_report.meeting_id,
                            "chunk_index": index,
                            "chunks_total": len(chunks),
                            "segments": [asdict(record) for record in records],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "meeting_report_chunk",
                    "schema": ai_chunk_notes_schema(),
                    "strict": True,
                }
            },
        }
        try:
            response = _post_openai_response_with_context(
                payload=payload,
                api_key=api_key,
                context=f"AI chunk {index}/{len(chunks)}",
                allow_openrouter_fallback=allow_openrouter_fallback,
            )
            provider_model = response.get("_provider_model") or provider_model
            content = _extract_output_text(response)
            if not content:
                raise ReportGenerationError(
                    f"AI chunk {index} did not contain output text"
                )
            chunk_payload = _loads_ai_json(content, context=f"AI chunk {index}")
        except ReportQuotaError:
            # Нет денег — не пропускаем как «сбойный чанк», а пробрасываем выше.
            raise
        except ReportGenerationError:
            # Один сбойный/пустой чанк не должен ронять весь AI-отчёт длинной встречи
            # и откатывать его в сырой local-дамп: пропускаем чанк, собираем из остальных.
            skipped_chunks.append(index)
            continue
        chunk_notes.append(
            _filter_payload_citations(
                chunk_payload, {record.segment_id for record in records}
            )
        )

    if not chunk_notes:
        raise ReportGenerationError(
            f"AI chunked report produced no usable chunk notes "
            f"({len(skipped_chunks)}/{len(chunks)} chunks failed)"
        )
    chunk_skip_warning = (
        f"AI chunks skipped: {len(skipped_chunks)}/{len(chunks)} returned no usable "
        f"output (chunks {skipped_chunks}); report assembled from the rest."
        if skipped_chunks
        else None
    )

    if len(chunks) > AI_SYNTHESIS_CHUNK_LIMIT:
        synthesis_skip_warning = (
            f"AI synthesis skipped: {len(chunks)} chunks exceed "
            f"MAC_TRANSCRIBER_AI_SYNTHESIS_CHUNK_LIMIT={AI_SYNTHESIS_CHUNK_LIMIT}; "
            "assembled report from chunk notes."
        )
        return _merge_ai_chunk_notes_payload(
            base_report,
            chunk_notes,
            report_model=report_model,
            warning="; ".join(
                w for w in [synthesis_skip_warning, chunk_skip_warning] if w
            ),
            provider_model=provider_model,
        )

    try:
        ai_payload = _build_chunked_synthesis_payload(
            base_report=base_report,
            chunk_notes=chunk_notes,
            report_model=report_model,
            api_key=api_key,
            context_pack=context_pack,
            allow_openrouter_fallback=allow_openrouter_fallback,
        )
    except ReportQuotaError:
        # Нет денег на синтезе — не собираем «как есть» из заметок, пробрасываем выше.
        raise
    except ReportGenerationError as exc:
        return _merge_ai_chunk_notes_payload(
            base_report,
            chunk_notes,
            report_model=report_model,
            warning="; ".join(
                w for w in [f"AI synthesis fallback: {exc}", chunk_skip_warning] if w
            ),
            provider_model=provider_model,
        )

    known_ids = {record.segment_id for record in base_report.transcript}
    ai_payload = _filter_payload_citations(ai_payload, known_ids)
    actual_report_model = str(
        ai_payload.pop("_provider_model", report_model) or report_model
    )
    synthesis_warning = ai_payload.pop("_synthesis_warning", None)
    report = _merge_ai_synthesis_payload(
        base_report,
        ai_payload,
        report_model=actual_report_model,
        recover_base_items=recover_base_items or context_pack is not None,
    )
    try:
        validate_report_citations(report)
    except ValueError as exc:
        raise ReportGenerationError(
            f"AI chunked report failed validation: {exc}"
        ) from exc
    for warning in (chunk_skip_warning, synthesis_warning):
        if warning:
            report.warnings.append(warning)
    return report


def _build_chunked_synthesis_payload(
    *,
    base_report: MeetingReport,
    chunk_notes: list[dict[str, Any]],
    report_model: str,
    api_key: str,
    context_pack: dict[str, Any] | None = None,
    allow_openrouter_fallback: bool = True,
) -> dict[str, Any]:
    if len(chunk_notes) <= AI_SYNTHESIS_BATCH_SIZE:
        return _request_ai_synthesis_payload(
            base_report=base_report,
            chunk_notes=chunk_notes,
            report_model=report_model,
            api_key=api_key,
            context="AI synthesis",
            context_pack=context_pack,
            allow_openrouter_fallback=allow_openrouter_fallback,
        )

    batch_payloads: list[dict[str, Any]] = []
    skipped_batches: list[int] = []
    batches = _chunks(chunk_notes, size=AI_SYNTHESIS_BATCH_SIZE)
    for index, batch in enumerate(batches, start=1):
        try:
            batch_payloads.append(
                _request_ai_synthesis_payload(
                    base_report=base_report,
                    chunk_notes=batch,
                    report_model=report_model,
                    api_key=api_key,
                    context=f"AI synthesis batch {index}/{len(batches)}",
                    context_pack=context_pack,
                    allow_openrouter_fallback=allow_openrouter_fallback,
                )
            )
        except ReportQuotaError:
            # Нет денег — пробрасываем, а не сшиваем частично.
            raise
        except ReportGenerationError:
            # Один сбойный батч (напр. сетевой дроп) не должен ронять весь синтез в
            # шумную сборку-из-заметок: пропускаем его и синтезируем из выживших.
            skipped_batches.append(index)
            continue
    if not batch_payloads:
        raise ReportGenerationError(
            f"AI synthesis failed: all {len(batches)} batches errored"
        )
    try:
        merged = _merge_ai_synthesis_payloads(base_report, batch_payloads)
    except Exception as exc:  # noqa: BLE001
        raise ReportGenerationError(f"AI synthesis batch merge failed: {exc}") from exc
    if skipped_batches:
        merged["_synthesis_warning"] = (
            f"AI synthesis: {len(skipped_batches)}/{len(batches)} batches skipped "
            f"(batches {skipped_batches}); synthesized from the rest."
        )
    return merged


def _request_ai_synthesis_payload(
    *,
    base_report: MeetingReport,
    chunk_notes: list[dict[str, Any]],
    report_model: str,
    api_key: str,
    context: str,
    context_pack: dict[str, Any] | None = None,
    allow_openrouter_fallback: bool = True,
) -> dict[str, Any]:
    synthesis_payload = {
        "model": report_model,
        "reasoning": {"effort": _reasoning_effort("medium")},
        "input": [
            {
                "role": "system",
                "content": (
                    "Ты готовишь подробный русскоязычный отчёт по встрече из набора "
                    "пофрагментных заметок. Используй заметки и citations как источник фактов "
                    "текущей встречи. Если дан prior_context, используй его только как фон из "
                    "прошлых встреч: для связности, открытых хвостов и повторяющихся рисков. "
                    "Не выдавай prior_context за факты текущей встречи и не цитируй его segment_id "
                    "как citation текущего отчета. "
                    "Сначала восстанови decisions, action_items, open_questions и risks только из "
                    "заметок текущей встречи. prior_context не должен удалять или понижать такие "
                    "пункты: если текущие заметки содержат 'договорились', 'нужно', 'добавить', "
                    "'не хватает', 'уточнить', вынеси это в формальные разделы с текущими segment_id. "
                    "action_items создавай только для реальных next steps текущего проекта: "
                    "должен быть конкретный deliverable, изменение, проверка или commitment. "
                    "Методологические фразы вроде 'нужно понимать/учитывать/помнить', "
                    "'в отчете нужно показывать' или объяснение правил не являются задачами; "
                    "помещай их в adaptive_sections как методологию, критерии или оговорки. "
                    "Структуру адаптируй под смысл записи: проектная встреча, лекция, "
                    "интервью, техническое обсуждение, консультация или общий разговор. "
                    "Не обобщай до пустых фраз: сохраняй важные детали, термины, связи, "
                    "варианты, решения, задачи, риски и открытые вопросы. Дедуплицируй "
                    "повторяющиеся пункты внутри входных заметок. "
                    "Каждый факт упоминай РОВНО ОДИН раз в наиболее подходящем месте: не "
                    "повторяй один и тот же пункт в TL;DR/overview, смысловых разделах и "
                    "decisions/tasks одновременно. TL;DR — это новые сжатые тезисы, а не "
                    "копии буллетов разделов. При этом не теряй крупные самостоятельные темы "
                    "встречи (отдельные подсистемы/разделы продукта): каждая значимая тема "
                    "должна быть отражена хотя бы в одном разделе. "
                    "Каждый пункт должен иметь "
                    "точные segment_id из заметок. Ограничения: overview до 900 знаков; "
                    "adaptive_sections до 5 секций по 5-8 пунктов; timeline до 8 блоков; "
                    "decisions/action_items/open_questions/risks до 12 пунктов на раздел; "
                    "notable_quotes до 5."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "meeting_id": base_report.meeting_id,
                        "title": base_report.title,
                        "source_filename": base_report.source_filename,
                        "segment_count": base_report.segment_count,
                        "duration": base_report.duration,
                        "chunk_notes": chunk_notes,
                        **({"prior_context": context_pack} if context_pack else {}),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "meeting_report_synthesis",
                "schema": ai_synthesis_schema(),
                "strict": True,
            }
        },
    }
    response = _post_openai_response_with_context(
        payload=synthesis_payload,
        api_key=api_key,
        context=context,
        allow_openrouter_fallback=allow_openrouter_fallback,
    )
    content = _extract_output_text(response)
    if not content:
        raise ReportGenerationError(f"{context} did not contain output text")
    payload = _loads_ai_json(content, context=context)
    if response.get("_provider_model"):
        payload["_provider_model"] = response["_provider_model"]
    return payload


def _enrich_ai_report_with_memory(
    report: MeetingReport,
    *,
    context_pack: dict[str, Any],
    report_model: str,
    api_key: str,
    allow_openrouter_fallback: bool,
) -> MeetingReport:
    payload = {
        "model": report_model,
        "reasoning": {"effort": _reasoning_effort("low")},
        "input": [
            {
                "role": "system",
                "content": (
                    "Ты добавляешь память прошлых встреч к уже готовому отчету по текущей встрече. "
                    "Не переписывай решения, задачи, вопросы, риски, timeline и основные смысловые "
                    "разделы текущего отчета. prior_context можно использовать только как фон: "
                    "связи с прежними решениями, повторяющиеся риски, открытые хвосты и полезные "
                    "ссылки на контекст. Не выдавай prior_context за факты текущей встречи. "
                    "Если ссылаешься на текущую встречу, используй только segment_id текущего "
                    "транскрипта. "
                    "Главная задача — преемственность: для каждого открытого вопроса, задачи, "
                    "решения и риска из prior_context определи по текущему отчету и транскрипту его "
                    "статус и явно подпиши одним из: 'всё ещё открыто', 'закрыто на этой встрече', "
                    "'повторяется снова' или 'пересмотрено/заменено'. Открытые хвосты прошлых встреч "
                    "выноси в open_threads_from_memory, повторяющиеся риски — в recurring_risks, "
                    "связи с прежними решениями — в prior_decisions. Не выдумывай статус: если по "
                    "текущим материалам он не ясен, помечай 'всё ещё открыто'. Опирайся только на "
                    "пункты, реально присутствующие в prior_context. "
                    "memory_sections используй только с kind из: "
                    "memory_context, prior_decisions, recurring_risks, open_threads_from_memory. "
                    "Не возвращай kind текущих разделов вроде actions, decisions, questions, "
                    "architecture или methodology. Верни только overview_addendum и memory_sections."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "meeting_id": report.meeting_id,
                        "title": report.title,
                        "segments": [asdict(record) for record in report.transcript],
                        "current_report": _report_memory_enrichment_payload(report),
                        "prior_context": context_pack,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "meeting_report_memory_enrichment",
                "schema": ai_memory_enrichment_schema(),
                "strict": True,
            }
        },
    }
    try:
        response = _post_openai_response_with_context(
            payload=payload,
            api_key=api_key,
            context="AI memory enrichment",
            allow_openrouter_fallback=allow_openrouter_fallback,
        )
        content = _extract_output_text(response)
        if not content:
            raise ReportGenerationError(
                "AI memory enrichment did not contain output text"
            )
        enrichment_payload = _loads_ai_json(content, context="AI memory enrichment")
        enrichment_payload = _filter_payload_citations(
            enrichment_payload,
            {record.segment_id for record in report.transcript},
        )
        enrichment_payload = _sanitize_memory_enrichment_payload(enrichment_payload)
        return _merge_memory_enrichment_payload(report, enrichment_payload)
    except ReportGenerationError as exc:
        report.warnings.append(f"AI memory enrichment skipped: {exc}")
        return report


def _report_memory_enrichment_payload(report: MeetingReport) -> dict[str, Any]:
    return {
        "overview": report.overview,
        "decisions": [asdict(item) for item in report.decisions],
        "action_items": [asdict(item) for item in report.action_items],
        "open_questions": [asdict(item) for item in report.open_questions],
        "risks": [asdict(item) for item in report.risks],
        "adaptive_sections": [
            {
                "kind": section.kind,
                "title": section.title,
                "summary": section.summary,
                "citations": section.citations,
            }
            for section in report.adaptive_sections
        ],
    }


_CRITIC_SYSTEM_PROMPT = (
    "Ты редактор-критик уже готового русскоязычного отчёта по встрече. На вход даны "
    "формальные разделы (decisions, action_items, open_questions, risks), где у каждого "
    "пункта есть стабильный id, а также read-only контекст: overview, context_sections "
    "(смысловые разделы) и cited_segments (исходные реплики транскрипта по id сегментов). "
    "Твоя задача — навести порядок, ничего не выдумывая:\n"
    "1) Дедуп: если несколько пунктов об одном и том же — объедини их операцией merge.\n"
    "2) Кросс-дедуп: если формальный пункт лишь повторяет то, что уже есть в overview или "
    "context_sections и не несёт отдельной формальной ценности — удали его операцией drop.\n"
    "3) Rewrite: почини ASR-ошибки и косноязычие, ужми до делового тезиса (1-2 предложения). "
    "Опирайся на cited_segments, чтобы восстановить смысл, но НЕ добавляй фактов, которых "
    "в них нет.\n"
    "Запреты: НЕ удаляй пункт только потому, что он 'неважный' — режь лишь дубли и повторы. "
    "НЕ переноси пункты между разделами. НЕ трогай и не возвращай citations — они "
    "подставятся автоматически из исходных пунктов. Не упоминай segment_id в тексте.\n"
    "Формат: по каждому разделу верни список операций над его пунктами. "
    "keep (оставить как есть — поле id), drop (удалить — id + reason), "
    "rewrite (переписать — id + новый title/text), merge (склеить дубли — ids + новый "
    "title/text). Для каждого исходного пункта — ровно одна операция keep/drop/rewrite, "
    "либо участие в одном merge. notes — короткое резюме правок."
)


def _maybe_apply_critic_pass(
    *,
    report: MeetingReport,
    report_model: str,
    api_key: str | None,
    allow_openrouter_fallback: bool,
) -> MeetingReport:
    """Запускает критика, если задан MAC_TRANSCRIBER_REPORT_CRITIC_MODEL; иначе no-op."""
    critic_model = report_critic_model()
    if not critic_model:
        return report
    key = api_key or load_openai_api_key()
    if not key:
        if not (allow_openrouter_fallback and load_openrouter_api_key()):
            report.warnings.append(
                "AI critic skipped: OPENAI_API_KEY is not configured"
            )
            return report
        key = ""
    return _apply_report_critic(
        report=report,
        report_model=critic_model,
        api_key=key,
        allow_openrouter_fallback=allow_openrouter_fallback,
    )


def _apply_report_critic(
    *,
    report: MeetingReport,
    report_model: str,
    api_key: str,
    allow_openrouter_fallback: bool,
) -> MeetingReport:
    if not any(
        [
            report.decisions,
            report.action_items,
            report.open_questions,
            report.risks,
        ]
    ):
        return report  # нечего чистить
    payload = {
        "model": report_model,
        "reasoning": {"effort": _reasoning_effort("medium")},
        "input": [
            {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    _report_critic_payload(report), ensure_ascii=False
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "meeting_report_critic",
                "schema": ai_report_critic_schema(),
                "strict": True,
            }
        },
    }
    try:
        response = _post_openai_response_with_context(
            payload=payload,
            api_key=api_key,
            context="AI critic",
            allow_openrouter_fallback=allow_openrouter_fallback,
        )
        content = _extract_output_text(response)
        if not content:
            raise ReportGenerationError("AI critic did not contain output text")
        ops_payload = _loads_ai_json(content, context="AI critic")
        critiqued = _apply_critic_ops(report, ops_payload)
        validate_report_citations(critiqued)
    except ReportQuotaError:
        # Критик — финальная полировка уже готового отчёта; нет денег на него не повод
        # ронять/ставить на паузу хороший отчёт. Помечаем и отдаём доcritic-версию.
        report.warnings.append("AI critic skipped: provider quota exhausted")
        return report
    except (ReportGenerationError, ValueError) as exc:
        report.warnings.append(f"AI critic skipped: {exc}")
        return report
    critiqued.warnings.append(
        _critic_summary(report, critiqued, report_model, ops_payload)
    )
    return critiqued


def _report_critic_payload(report: MeetingReport) -> dict[str, Any]:
    def ided_items(items: list[ReportItem], prefix: str) -> list[dict[str, Any]]:
        return [
            {"id": f"{prefix}{index}", "title": item.title, "text": item.text}
            for index, item in enumerate(items, start=1)
        ]

    cited: set[str] = set()
    for group in (
        report.decisions,
        report.open_questions,
        report.risks,
        report.action_items,
    ):
        for item in group:
            cited.update(item.citations)
    text_by_id = {record.segment_id: record.text for record in report.transcript}
    cited_segments = {
        segment_id: text_by_id[segment_id]
        for segment_id in sorted(cited)
        if segment_id in text_by_id
    }
    return {
        "overview": report.overview,
        "decisions": ided_items(report.decisions, "D"),
        "action_items": [
            {
                "id": f"T{index}",
                "title": item.title,
                "text": item.text,
                "owner": item.owner,
                "due": item.due,
            }
            for index, item in enumerate(report.action_items, start=1)
        ],
        "open_questions": ided_items(report.open_questions, "Q"),
        "risks": ided_items(report.risks, "R"),
        "context_sections": [
            {
                "title": section.title,
                "summary": section.summary,
                "points": [item.text for item in section.items],
            }
            for section in report.adaptive_sections
        ],
        "cited_segments": cited_segments,
    }


def _apply_critic_ops(
    report: MeetingReport, ops_payload: dict[str, Any]
) -> MeetingReport:
    known_ids = {record.segment_id for record in report.transcript}
    decisions = _dedupe_report_items(
        _apply_section_ops(
            report.decisions, "D", ops_payload.get("decisions"), known_ids
        )
    )
    open_questions = _dedupe_report_items(
        _apply_section_ops(
            report.open_questions, "Q", ops_payload.get("open_questions"), known_ids
        )
    )
    risks = _dedupe_report_items(
        _apply_section_ops(report.risks, "R", ops_payload.get("risks"), known_ids)
    )
    action_items = _dedupe_action_items(
        _apply_action_section_ops(
            report.action_items, "T", ops_payload.get("action_items"), known_ids
        )
    )
    # coverage зависит только от adaptive_sections; формальные правки её не меняют,
    # но пересчитываем для единообразия с остальными merge-проходами.
    coverage = build_coverage(
        transcript=report.transcript, adaptive_sections=report.adaptive_sections
    )
    return MeetingReport(
        meeting_id=report.meeting_id,
        title=report.title,
        source_filename=report.source_filename,
        model_name=report.model_name,
        generated_by=report.generated_by,
        segment_count=report.segment_count,
        duration=report.duration,
        overview=report.overview,
        timeline=report.timeline,
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        risks=risks,
        notable_quotes=report.notable_quotes,
        transcript=report.transcript,
        profile=report.profile,
        adaptive_sections=report.adaptive_sections,
        coverage=coverage,
        warnings=list(report.warnings),
    )


def _apply_section_ops(
    items: list[ReportItem],
    prefix: str,
    ops: Any,
    known_ids: set[str],
) -> list[ReportItem]:
    by_id = {f"{prefix}{index}": item for index, item in enumerate(items, start=1)}
    if not isinstance(ops, list):
        # Критик не дал операций по разделу — оставляем его без изменений.
        return [
            ReportItem(title=item.title, text=item.text, citations=list(item.citations))
            for item in items
        ]
    referenced: set[str] = set()
    result: list[ReportItem] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op") or "").strip().lower()
        if kind == "merge":
            ids = _unique_strings(
                [str(value) for value in (op.get("ids") or []) if str(value) in by_id]
            )
            referenced.update(ids)
            sources = [by_id[item_id] for item_id in ids]
            if not sources:
                continue
            citations = _unique_strings(
                [citation for source in sources for citation in source.citations]
            )
            text = (
                _strip_unknown_segment_refs(
                    str(op.get("text") or "").strip(), known_ids
                )
                or sources[0].text
            )
            title = str(op.get("title") or "").strip() or sources[0].title
            result.append(ReportItem(title=title, text=text, citations=citations))
            continue
        item_id = str(op.get("id") or "")
        if item_id not in by_id:
            continue
        referenced.add(item_id)
        source = by_id[item_id]
        if kind == "drop":
            continue
        if kind == "rewrite":
            text = (
                _strip_unknown_segment_refs(
                    str(op.get("text") or "").strip(), known_ids
                )
                or source.text
            )
            title = str(op.get("title") or "").strip() or source.title
            result.append(
                ReportItem(title=title, text=text, citations=list(source.citations))
            )
            continue
        # keep или неизвестная операция — сохраняем пункт как есть.
        result.append(
            ReportItem(
                title=source.title, text=source.text, citations=list(source.citations)
            )
        )
    # Пункты, которых критик не коснулся ни одной операцией — keep по умолчанию,
    # чтобы никогда не терять пункт молча.
    for item_id, source in by_id.items():
        if item_id not in referenced:
            result.append(
                ReportItem(
                    title=source.title,
                    text=source.text,
                    citations=list(source.citations),
                )
            )
    return result


def _apply_action_section_ops(
    items: list[ActionItem],
    prefix: str,
    ops: Any,
    known_ids: set[str],
) -> list[ActionItem]:
    by_id = {f"{prefix}{index}": item for index, item in enumerate(items, start=1)}

    def clone(source: ActionItem) -> ActionItem:
        return ActionItem(
            title=source.title,
            text=source.text,
            owner=source.owner,
            due=source.due,
            citations=list(source.citations),
        )

    if not isinstance(ops, list):
        return [clone(item) for item in items]
    referenced: set[str] = set()
    result: list[ActionItem] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op") or "").strip().lower()
        if kind == "merge":
            ids = _unique_strings(
                [str(value) for value in (op.get("ids") or []) if str(value) in by_id]
            )
            referenced.update(ids)
            sources = [by_id[item_id] for item_id in ids]
            if not sources:
                continue
            citations = _unique_strings(
                [citation for source in sources for citation in source.citations]
            )
            text = (
                _strip_unknown_segment_refs(
                    str(op.get("text") or "").strip(), known_ids
                )
                or sources[0].text
            )
            title = str(op.get("title") or "").strip() or sources[0].title
            owner = next((source.owner for source in sources if source.owner), "")
            due = next((source.due for source in sources if source.due), "")
            result.append(
                ActionItem(
                    title=title, text=text, owner=owner, due=due, citations=citations
                )
            )
            continue
        item_id = str(op.get("id") or "")
        if item_id not in by_id:
            continue
        referenced.add(item_id)
        source = by_id[item_id]
        if kind == "drop":
            continue
        if kind == "rewrite":
            text = (
                _strip_unknown_segment_refs(
                    str(op.get("text") or "").strip(), known_ids
                )
                or source.text
            )
            title = str(op.get("title") or "").strip() or source.title
            result.append(
                ActionItem(
                    title=title,
                    text=text,
                    owner=source.owner,
                    due=source.due,
                    citations=list(source.citations),
                )
            )
            continue
        result.append(clone(source))
    for item_id, source in by_id.items():
        if item_id not in referenced:
            result.append(clone(source))
    return result


def _critic_summary(
    before: MeetingReport,
    after: MeetingReport,
    model: str,
    ops_payload: dict[str, Any],
) -> str:
    parts = (
        f"decisions {len(before.decisions)}->{len(after.decisions)}",
        f"action_items {len(before.action_items)}->{len(after.action_items)}",
        f"open_questions {len(before.open_questions)}->{len(after.open_questions)}",
        f"risks {len(before.risks)}->{len(after.risks)}",
    )
    summary = f"AI critic ({model}): " + ", ".join(parts)
    notes = str((ops_payload or {}).get("notes") or "").strip()
    if notes:
        summary += f"; {notes}"
    return summary


def _merge_memory_enrichment_payload(
    report: MeetingReport,
    payload: dict[str, Any],
) -> MeetingReport:
    memory_sections = _adaptive_sections_from_payload(
        payload.get("memory_sections"), fallback=[]
    )
    overview_addendum = str(payload.get("overview_addendum") or "").strip()
    if overview_addendum:
        memory_sections = [
            _memory_overview_section(overview_addendum),
            *memory_sections,
        ]
    if not memory_sections and not overview_addendum:
        return report
    memory_sections = _dedupe_adaptive_sections(memory_sections)
    adaptive_sections = [*report.adaptive_sections, *memory_sections]
    coverage = build_coverage(
        transcript=report.transcript, adaptive_sections=adaptive_sections
    )
    enriched = MeetingReport(
        meeting_id=report.meeting_id,
        title=report.title,
        source_filename=report.source_filename,
        model_name=report.model_name,
        generated_by=report.generated_by,
        segment_count=report.segment_count,
        duration=report.duration,
        overview=report.overview,
        timeline=report.timeline,
        decisions=report.decisions,
        action_items=report.action_items,
        open_questions=report.open_questions,
        risks=report.risks,
        notable_quotes=report.notable_quotes,
        transcript=report.transcript,
        profile=report.profile,
        adaptive_sections=adaptive_sections,
        coverage=coverage,
        warnings=list(report.warnings),
    )
    validate_report_citations(enriched)
    return enriched


def _sanitize_memory_enrichment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(payload)
    sanitized["overview_addendum"] = _strip_all_segment_refs(
        str(sanitized.get("overview_addendum") or "")
    )
    sections = sanitized.get("memory_sections")
    if not isinstance(sections, list):
        sanitized["memory_sections"] = []
        return sanitized
    sanitized_sections = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section = _strip_all_segment_refs(section)
        kind = str(section.get("kind") or "memory_context").strip()
        if kind not in MEMORY_SECTION_KINDS:
            kind = "memory_context"
        title = str(section.get("title") or "Контекст прошлых встреч").strip()
        if not title.lower().startswith("память:"):
            title = f"Память: {title}"
        section["kind"] = kind
        section["title"] = title
        section["citations"] = []
        items = section.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item["citations"] = []
        sanitized_sections.append(section)
    sanitized["memory_sections"] = sanitized_sections
    return sanitized


def _memory_overview_section(overview_addendum: str) -> AdaptiveSection:
    return AdaptiveSection(
        kind="memory_context",
        title="Память: контекст прошлых встреч",
        purpose="Отделить фон прошлых встреч от фактов текущего транскрипта.",
        summary=overview_addendum,
        items=[],
        citations=[],
        accent="violet",
    )


def _strip_all_segment_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_all_segment_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_all_segment_refs(item) for item in value]
    if isinstance(value, str):
        return _strip_unknown_segment_refs(value, set())
    return value


def _merge_ai_synthesis_payloads(
    base_report: MeetingReport,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    if not payloads:
        raise ReportGenerationError("AI synthesis did not return any batch payloads")

    profile = next(
        (
            payload.get("profile")
            for payload in payloads
            if isinstance(payload.get("profile"), dict)
        ),
        {
            "kind": base_report.profile.kind,
            "label": base_report.profile.label,
            "confidence": base_report.profile.confidence,
            "rationale": base_report.profile.rationale,
        },
    )
    decisions = _dedupe_payload_items(_flatten_payload_list(payloads, "decisions"))[:20]
    action_items = _dedupe_payload_actions(
        _flatten_payload_list(payloads, "action_items")
    )[:20]
    open_questions = _dedupe_payload_items(
        _flatten_payload_list(payloads, "open_questions")
    )[:20]
    risks = _dedupe_payload_items(_flatten_payload_list(payloads, "risks"))[:20]
    notable_quotes = _dedupe_payload_items(
        _flatten_payload_list(payloads, "notable_quotes")
    )[:8]
    adaptive_sections = _dedupe_payload_sections(
        _flatten_payload_list(payloads, "adaptive_sections")
    )[:7]
    if not adaptive_sections:
        adaptive_sections = _payload_sections_from_top_level(
            decisions=decisions,
            action_items=action_items,
            open_questions=open_questions,
            risks=risks,
        )

    merged = {
        "profile": profile,
        "overview": _shorten(
            " ".join(str(payload.get("overview") or "") for payload in payloads),
            limit=1200,
        ),
        "adaptive_sections": adaptive_sections,
        "timeline": _dedupe_payload_timeline(
            _flatten_payload_list(payloads, "timeline")
        )[:14],
        "decisions": decisions,
        "action_items": action_items,
        "open_questions": open_questions,
        "risks": risks,
        "notable_quotes": notable_quotes,
    }
    # Сохраняем тег провайдера из батчей, чтобы generated_by не терял
    # "openrouter:<model>" на длинных стенограммах (батчевый путь синтеза).
    provider_model = next(
        (
            payload.get("_provider_model")
            for payload in payloads
            if payload.get("_provider_model")
        ),
        None,
    )
    if provider_model:
        merged["_provider_model"] = provider_model
    return merged


def write_report_artifacts(
    *,
    output_dir: Path,
    meeting_id: str,
    title: str,
    source_filename: str,
    model_name: str,
    segments: list[Any],
    use_ai: bool,
    make_pdf: bool,
    report_model: str | None = None,
    api_key: str | None = None,
    context_pack: dict[str, Any] | None = None,
) -> ReportArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        meeting_id=meeting_id,
        title=title,
        source_filename=source_filename,
        model_name=model_name,
        segments=segments,
        use_ai=use_ai,
        report_model=report_model,
        api_key=api_key,
        context_pack=context_pack,
    )

    return write_report_artifacts_from_report(
        output_dir=output_dir,
        report=report,
        requested_ai=use_ai,
        make_pdf=make_pdf,
    )


def write_report_artifacts_from_report(
    *,
    output_dir: Path,
    report: MeetingReport,
    requested_ai: bool,
    make_pdf: bool,
) -> ReportArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    typst_path = output_dir / "report.typ"
    health_path = output_dir / "report_health.json"
    coverage_path = output_dir / "coverage.json"
    slack_summary_path = output_dir / "slack_summary.md"
    protocol = build_protocol_data(report)
    json_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_report_markdown(report), encoding="utf-8")
    html_path.write_text(render_report_html(report), encoding="utf-8")
    typst_path.write_text(render_report_typst(report), encoding="utf-8")
    coverage_path.write_text(
        json.dumps(build_coverage_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pdf_path = None
    pdf_error = None
    if make_pdf:
        try:
            pdf_path = render_report_pdf(
                html_path=html_path, pdf_path=output_dir / "report.pdf"
            )
        except ReportGenerationError as exc:
            pdf_error = str(exc)

    health = build_report_health(
        report=report,
        output_dir=output_dir,
        requested_ai=requested_ai,
        requested_pdf=make_pdf,
        pdf_path=pdf_path,
        pdf_error=pdf_error,
    )
    health_path.write_text(
        json.dumps(health, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    slack_summary = render_slack_summary(
        protocol=protocol,
        health=health,
        output_dir=output_dir,
        pdf_path=pdf_path,
    )
    slack_files = _slack_artifact_files(output_dir=output_dir, pdf_path=pdf_path)
    slack_summary_path.write_text(slack_summary, encoding="utf-8")

    return ReportArtifacts(
        json_path=json_path,
        markdown_path=markdown_path,
        html_path=html_path,
        typst_path=typst_path,
        health_path=health_path,
        coverage_path=coverage_path,
        slack_summary_path=slack_summary_path,
        slack_text=slack_summary,
        slack_files=slack_files,
        pdf_path=pdf_path,
        pdf_error=pdf_error,
        generated_by=report.generated_by,
        status=str(health["status"]),
        alerts=list(health["alerts"]),
    )


def transcript_records(segments: list[Any]) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    ordered = sorted(
        segments,
        key=lambda item: (
            _segment_float(item, "start"),
            _segment_float(item, "end"),
        ),
    )
    for index, segment in enumerate(ordered, start=1):
        text = " ".join(str(_segment_value(segment, "text", "")).split()).strip()
        if not text:
            continue
        records.append(
            TranscriptRecord(
                segment_id=str(
                    _segment_value(segment, "segment_id", "") or f"S{index:04d}"
                ),
                start=_segment_float(segment, "start"),
                end=_segment_float(segment, "end"),
                speaker=str(_segment_value(segment, "speaker", "Speaker") or "Speaker"),
                text=text,
            )
        )
    return records


def _segment_value(segment: Any, key: str, default: Any = None) -> Any:
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


def _segment_float(segment: Any, key: str) -> float:
    try:
        return float(_segment_value(segment, key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def infer_report_profile(records: list[TranscriptRecord]) -> ReportProfile:
    text = " ".join(record.text.lower() for record in records)
    profiles = [
        (
            "lecture",
            "Лекция / обучение",
            [
                "лекция",
                "сегодня разбер",
                "определение",
                "термин",
                "например",
                "формула",
                "концепц",
                "метрик",
                "запомн",
                "разберем",
            ],
        ),
        (
            "research_interview",
            "Исследовательское интервью",
            [
                "расскажите",
                "как вы",
                "почему вы",
                "пользуетесь",
                "опыт",
                "интервью",
                "респондент",
                "боль",
                "инсайт",
            ],
        ),
        (
            "technical_discussion",
            "Техническое обсуждение",
            [
                "архитектур",
                "api",
                "база данных",
                "интеграц",
                "модель",
                "пайплайн",
                "сервис",
                "код",
                "деплой",
                "тест",
            ],
        ),
        (
            "consultation",
            "Консультация",
            [
                "рекоменд",
                "диагноз",
                "проблема клиента",
                "совет",
                "вариант",
                "консультац",
                "лучше сделать",
            ],
        ),
        (
            "project_sync",
            "Проектная встреча",
            [
                "договорились",
                "решили",
                "нужно",
                "задач",
                "следующий шаг",
                "риск",
                "блокер",
                "mvp",
                "запуск",
                "план",
                "срок",
            ],
        ),
    ]
    scores: list[tuple[int, str, str, list[str]]] = []
    for kind, label, markers in profiles:
        score = sum(text.count(marker) for marker in markers)
        scores.append((score, kind, label, markers))
    score, kind, label, markers = max(scores, key=lambda item: item[0])
    if score <= 0:
        return ReportProfile(
            kind="general",
            label="Общий разговор",
            confidence=0.35,
            rationale="Явных маркеров жанра не найдено; выбран универсальный отчёт.",
        )
    confidence = min(0.92, 0.45 + score * 0.08)
    matched = [marker for marker in markers if marker in text][:4]
    return ReportProfile(
        kind=kind,
        label=label,
        confidence=round(confidence, 2),
        rationale="Профиль выбран по маркерам: " + ", ".join(matched),
    )


def build_adaptive_sections(
    *,
    profile: ReportProfile,
    transcript: list[TranscriptRecord],
    decisions: list[ReportItem],
    action_items: list[ActionItem],
    open_questions: list[ReportItem],
    risks: list[ReportItem],
    notable_quotes: list[ReportItem],
) -> list[AdaptiveSection]:
    if profile.kind == "lecture":
        notes = [
            ReportItem(
                title=f"Тезис {index}",
                text=record.text,
                citations=[record.segment_id],
            )
            for index, record in enumerate(transcript, start=1)
        ]
        terms = _matching_items(
            transcript,
            title_prefix="Термин",
            patterns=[r"\bопределение\b", r"\bэто\b", r"\bтермин\b", r"\bметрик"],
        )
        examples = _matching_items(
            transcript,
            title_prefix="Пример",
            patterns=[r"\bнапример\b", r"\bпример\b", r"\bесли\b"],
        )
        return [
            AdaptiveSection(
                kind="lecture_notes",
                title="Конспект",
                purpose="Сохранить ход объяснения без потери деталей.",
                summary=_shorten(
                    " ".join(record.text for record in transcript), limit=900
                ),
                items=notes,
                citations=[record.segment_id for record in transcript],
                accent="blue",
            ),
            AdaptiveSection(
                kind="terms",
                title="Термины и определения",
                purpose="Вытащить понятия, которые нужно помнить после лекции.",
                summary="Ключевые определения и понятия из записи.",
                items=terms,
                citations=_unique_citations(terms),
                accent="violet",
            ),
            AdaptiveSection(
                kind="examples",
                title="Примеры",
                purpose="Сохранить иллюстрации и практические объяснения.",
                summary="Примеры, которыми спикер раскрывал материал.",
                items=examples,
                citations=_unique_citations(examples),
                accent="green",
            ),
            AdaptiveSection(
                kind="questions",
                title="Вопросы слушателей",
                purpose="Не потерять вопросы, которые меняют смысл материала.",
                summary="Вопросы и уточнения из аудитории.",
                items=_clone_report_items(open_questions),
                citations=_unique_citations(open_questions),
                accent="amber",
            ),
        ]

    if profile.kind == "research_interview":
        insights = _matching_items(
            transcript,
            title_prefix="Инсайт",
            patterns=[
                r"\bпотому что\b",
                r"\bне удобно\b",
                r"\bболь\b",
                r"\bхочу\b",
                r"\bнужно\b",
            ],
        )
        return [
            AdaptiveSection(
                "interview_context",
                "Контекст респондента",
                "Понять ситуацию собеседника.",
                _shorten(" ".join(r.text for r in transcript[:4]), limit=700),
                _clone_report_items(notable_quotes[:3]),
                _unique_citations(notable_quotes[:3]),
                "blue",
            ),
            AdaptiveSection(
                "insights",
                "Инсайты",
                "Выделить наблюдения, которые полезны продукту.",
                "Повторяющиеся боли, мотивы и формулировки.",
                insights,
                _unique_citations(insights),
                "green",
            ),
            AdaptiveSection(
                "quotes",
                "Доказательные цитаты",
                "Сохранить голос пользователя.",
                "Цитаты для дальнейшего анализа.",
                _clone_report_items(notable_quotes),
                _unique_citations(notable_quotes),
                "violet",
            ),
            AdaptiveSection(
                "questions",
                "Открытые вопросы",
                "Что стоит уточнить в следующих интервью.",
                "Вопросы и неполные места.",
                _clone_report_items(open_questions),
                _unique_citations(open_questions),
                "amber",
            ),
        ]

    if profile.kind == "technical_discussion":
        return [
            AdaptiveSection(
                "architecture",
                "Архитектура и варианты",
                "Сохранить технические варианты и аргументы.",
                _shorten(" ".join(record.text for record in transcript), limit=900),
                _clone_report_items(notable_quotes),
                _unique_citations(notable_quotes),
                "blue",
            ),
            AdaptiveSection(
                "decisions",
                "Технические решения",
                "Отделить принятые решения от обсуждения.",
                "Принятые или предложенные технические решения.",
                _clone_report_items(decisions),
                _unique_citations(decisions),
                "green",
            ),
            AdaptiveSection(
                "risks",
                "Риски и блокеры",
                "Зафиксировать места, где возможна поломка.",
                "Технические риски, ограничения и неясности.",
                _clone_report_items(risks),
                _unique_citations(risks),
                "red",
            ),
            AdaptiveSection(
                "questions",
                "Открытые вопросы",
                "Что требует дополнительной проверки.",
                "Незакрытые технические вопросы.",
                _clone_report_items(open_questions),
                _unique_citations(open_questions),
                "amber",
            ),
        ]

    if profile.kind == "consultation":
        recommendations = _matching_items(
            transcript,
            title_prefix="Рекомендация",
            patterns=[
                r"\bрекоменд",
                r"\bлучше\b",
                r"\bстоит\b",
                r"\bнадо\b",
                r"\bнужно\b",
            ],
        )
        return [
            AdaptiveSection(
                "diagnosis",
                "Ситуация и диагноз",
                "Собрать проблему и контекст.",
                _shorten(" ".join(record.text for record in transcript), limit=800),
                _clone_report_items(notable_quotes[:4]),
                _unique_citations(notable_quotes[:4]),
                "blue",
            ),
            AdaptiveSection(
                "recommendations",
                "Рекомендации",
                "Сделать выводы применимыми.",
                "Рекомендованные действия и варианты.",
                recommendations,
                _unique_citations(recommendations),
                "green",
            ),
            AdaptiveSection(
                "actions",
                "План действий",
                "Превратить консультацию в next steps.",
                "Практические шаги после консультации.",
                _action_items_as_report_items(action_items),
                _unique_action_citations(action_items),
                "amber",
            ),
        ]

    return [
        AdaptiveSection(
            "decisions",
            "Решения",
            "Что было зафиксировано как договоренность или направление.",
            "Решения и договоренности из транскрипта.",
            _clone_report_items(decisions),
            _unique_citations(decisions),
            "green",
        ),
        AdaptiveSection(
            "actions",
            "Задачи и next steps",
            "Что нужно сделать после разговора.",
            "Задачи, владельцы и дальнейшие шаги.",
            _action_items_as_report_items(action_items),
            _unique_action_citations(action_items),
            "amber",
        ),
        AdaptiveSection(
            "risks",
            "Риски",
            "Что может сорвать результат или требует внимания.",
            "Риски, блокеры и слабые места.",
            _clone_report_items(risks),
            _unique_citations(risks),
            "red",
        ),
        AdaptiveSection(
            "questions",
            "Открытые вопросы",
            "Что осталось не закрыто.",
            "Вопросы, требующие ответа или проверки.",
            _clone_report_items(open_questions),
            _unique_citations(open_questions),
            "violet",
        ),
    ]


def build_coverage(
    *,
    transcript: list[TranscriptRecord],
    adaptive_sections: list[AdaptiveSection],
) -> list[CoverageEntry]:
    section_by_segment: dict[str, list[str]] = {
        record.segment_id: [] for record in transcript
    }
    for section in adaptive_sections:
        for segment_id in section.citations:
            section_by_segment.setdefault(segment_id, []).append(section.title)
        for item in section.items:
            for segment_id in item.citations:
                section_by_segment.setdefault(segment_id, []).append(section.title)
    entries: list[CoverageEntry] = []
    for record in transcript:
        titles = sorted(set(section_by_segment.get(record.segment_id, [])))
        if titles:
            status = "covered"
            rationale = "Сегмент использован в смысловых секциях отчёта."
        elif _is_low_signal(record.text):
            status = "low_signal"
            rationale = (
                "Сегмент похож на короткую реплику, повтор или техническую связку."
            )
        else:
            status = "supporting"
            rationale = (
                "Сегмент сохранён в полном транскрипте и поддерживает общий контекст."
            )
        entries.append(
            CoverageEntry(
                segment_id=record.segment_id,
                status=status,
                section_titles=titles,
                rationale=rationale,
            )
        )
    return entries


def validate_report_citations(report: MeetingReport) -> None:
    known = {record.segment_id for record in report.transcript}
    for section_name, items in [
        ("timeline", report.timeline),
        ("decisions", report.decisions),
        ("action_items", report.action_items),
        ("open_questions", report.open_questions),
        ("risks", report.risks),
        ("notable_quotes", report.notable_quotes),
    ]:
        for item in items:
            for segment_id in item.citations:
                if segment_id not in known:
                    raise ValueError(
                        f"{section_name} references unknown segment id: {segment_id}"
                    )
    for section in report.adaptive_sections:
        for segment_id in section.citations:
            if segment_id not in known:
                raise ValueError(
                    f"adaptive_sections references unknown segment id: {segment_id}"
                )
        for item in section.items:
            for segment_id in item.citations:
                if segment_id not in known:
                    raise ValueError(
                        f"adaptive_sections references unknown segment id: {segment_id}"
                    )
    for entry in report.coverage:
        if entry.segment_id not in known:
            raise ValueError(
                f"coverage references unknown segment id: {entry.segment_id}"
            )
        if entry.status not in COVERAGE_STATUSES:
            raise ValueError(f"coverage has unsupported status: {entry.status}")

    coverage_ids = [entry.segment_id for entry in report.coverage]
    duplicates = sorted(
        {
            segment_id
            for segment_id in coverage_ids
            if coverage_ids.count(segment_id) > 1
        }
    )
    if duplicates:
        raise ValueError(f"coverage has duplicate segment ids: {', '.join(duplicates)}")
    missing = sorted(known.difference(coverage_ids))
    if missing:
        raise ValueError(f"coverage is missing segment ids: {', '.join(missing)}")


def render_report_markdown(report: MeetingReport) -> str:
    data = build_protocol_data(report)
    meeting = data["meeting"]
    lines = [
        f"# {meeting['title']}",
        "",
        "<!-- Generated meeting protocol. Full transcript and coverage audit are separate artifacts. -->",
        "",
        f"- ID: `{meeting['doc_id']}`",
        f"- Date: `{meeting['date_short']}`",
        f"- Duration: `{meeting['duration']}`",
        f"- Segments: `{meeting['segments']}`",
        f"- Source: `{meeting['source']}`",
        f"- Transcript: `{meeting['transcript']}`",
        f"- Coverage: `{meeting['coverage']}`",
        f"- Generator: `{meeting['generator']}`",
        "",
    ]
    if data["warnings"]:
        lines.extend(["## Alerts", ""])
        lines.extend(f"- {warning}" for warning in data["warnings"])
        lines.append("")

    lines.extend(["## TL;DR", ""])
    lines.extend(f"- {item}" for item in data["tldr"])
    lines.append("")
    lines.extend(["## KPI", ""])
    lines.extend(f"- **{item['num']}** {item['cap']}" for item in data["kpis"])
    lines.append("")
    _append_protocol_sections_markdown(lines, data.get("sections", []))
    _append_protocol_items_markdown(lines, "Решения", data["decisions"])
    _append_protocol_items_markdown(lines, "Задачи", data["tasks"], include_owner=True)
    if data.get("tasks_footnote"):
        lines.extend([f"_Примечание: {data['tasks_footnote']}_", ""])
    _append_protocol_items_markdown(lines, "Открытые вопросы", data["questions"])
    _append_protocol_items_markdown(lines, "Риски", data["risks"])
    lines.extend(["## Хроника", ""])
    for block in data["timeline"]:
        lines.extend(
            [f"### {block['range']} · {block['title']}", "", block["summary"], ""]
        )
    lines.extend(["## Участники", ""])
    lines.extend(f"- {participant}" for participant in data["participants"])
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_report_html(report: MeetingReport) -> str:
    return render_protocol_html(build_protocol_data(report))


def build_protocol_data(report: MeetingReport) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    participants = sorted(
        {record.speaker for record in report.transcript if record.speaker}
    )
    decisions = _protocol_report_items("D", report.decisions, report)
    tasks = _protocol_action_items(report.action_items, report)
    questions = _protocol_report_items("Q", report.open_questions, report)
    risks = _protocol_report_items("R", report.risks, report)
    _dedupe_protocol_sections(decisions, tasks, questions, risks)
    timeline = _protocol_timeline(report)
    data = {
        "meeting": {
            "doc_id": _protocol_doc_id(report, generated_at),
            "title": report.title,
            "date_human": _protocol_date_human(report.source_filename, generated_at),
            "date_short": _protocol_date_short(report.source_filename, generated_at),
            "time_human": _protocol_time_human(report.source_filename),
            "duration": fmt_time(report.duration),
            "segments": report.segment_count,
            "source": report.source_filename,
            "transcript": "transcript.md",
            "coverage": "coverage.json",
            "generator": f"{report.generated_by} · {generated_at.strftime('%d.%m.%Y %H:%M UTC')}",
        },
        "participants": participants,
        "tldr": _protocol_tldr(report),
        "sections": _protocol_adaptive_sections(report),
        "decisions": decisions,
        "tasks": tasks,
        "tasks_footnote": "Сроки, не прозвучавшие в записи, не заполняются. Задачи без ответственного требуют назначения.",
        "questions": questions,
        "risks": risks,
        "timeline": timeline,
        "warnings": list(report.warnings),
    }
    data["kpis"] = [
        {"num": len(decisions), "cap": "Решений"},
        {"num": len(tasks), "cap": "Задач"},
        {"num": len(questions), "cap": "Вопросов"},
        {"num": len(risks), "cap": "Рисков"},
    ]
    return data


def _protocol_adaptive_sections(report: MeetingReport) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section in report.adaptive_sections:
        items = [
            {
                "title": item.title,
                "text": _shorten(item.text, limit=520),
                "ref": _citation_text(report, item.citations),
            }
            for item in section.items
        ]
        if not items and not section.summary:
            continue
        sections.append(
            {
                "title": section.title,
                "purpose": section.purpose,
                "summary": _shorten(section.summary, limit=700),
                "items": items,
            }
        )
    return sections


def build_coverage_payload(report: MeetingReport) -> dict[str, Any]:
    index = {record.segment_id: record for record in report.transcript}
    return {
        "meeting_id": report.meeting_id,
        "source": report.source_filename,
        "segment_count": report.segment_count,
        "coverage": [
            {
                "segment_id": entry.segment_id,
                "status": entry.status,
                "time": fmt_time(index[entry.segment_id].start)
                if entry.segment_id in index
                else "",
                "sections": entry.section_titles,
                "rationale": entry.rationale,
            }
            for entry in report.coverage
        ],
    }


def render_slack_summary(
    *,
    protocol: dict[str, Any],
    health: dict[str, Any],
    output_dir: Path,
    pdf_path: Path | None,
) -> str:
    meeting = protocol.get("meeting", {})
    title = str(meeting.get("title") or "Протокол встречи")
    date_parts = [
        str(meeting.get("date_human") or meeting.get("date_short") or "").strip()
    ]
    time_human = str(meeting.get("time_human") or "").strip()
    if time_human:
        date_parts.append(time_human)
    duration = str(meeting.get("duration") or "").strip()
    if duration:
        date_parts.append(f"длительность {duration}")
    participants = ", ".join(
        str(item) for item in protocol.get("participants", []) if item
    )
    kpis = {
        str(item.get("cap", "")).lower(): item.get("num")
        for item in protocol.get("kpis", [])
    }
    status = _slack_status_label(str(health.get("status") or "unknown"))
    alerts = _slack_health_alerts(
        [str(item) for item in health.get("alerts", []) if item]
    )
    files = _slack_artifact_files(output_dir=output_dir, pdf_path=pdf_path)

    lines = [
        f"*{title}*",
        " · ".join(part for part in date_parts if part),
    ]
    if participants:
        lines.append(f"*Участники:* {participants}")
    lines.append(
        "*Итоги:* "
        f"{_ru_count(kpis.get('решений', 0), ('решение', 'решения', 'решений'))} · "
        f"{_ru_count(kpis.get('задач', 0), ('задача', 'задачи', 'задач'))} · "
        f"{_ru_count(kpis.get('вопросов', 0), ('вопрос', 'вопроса', 'вопросов'))} · "
        f"{_ru_count(kpis.get('рисков', 0), ('риск', 'риска', 'рисков'))}"
    )
    lines.append(f"*Статус отчёта:* {status}")
    if alerts:
        lines.append("*Проверить:* " + "; ".join(alerts[:3]))
    lines.extend(["", "*Главное:*"])
    for item in protocol.get("tldr", [])[:4]:
        lines.append(f"• {_shorten(str(item), limit=260)}")
    lines.extend(["", "*Файлы:*"])
    for file_info in files:
        lines.append(
            f"• `{file_info['filename']}` — {file_info['title']} ({file_info['size_human']})"
        )
    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def _slack_status_label(status: str) -> str:
    return {
        "ok": "готов",
        "degraded": "требует проверки",
        "failed": "ошибка",
    }.get(status.strip().lower(), status or "неизвестно")


def _slack_health_alerts(alerts: list[str]) -> list[str]:
    has_openai_quota = any(_is_openai_quota_alert(alert) for alert in alerts)
    humanized: list[str] = []
    for alert in alerts:
        message = _humanize_slack_alert(alert)
        if (
            has_openai_quota
            and message
            == "AI-обогащение не выполнено; использован локальный резервный режим."
        ):
            continue
        if message not in humanized:
            humanized.append(message)
    return humanized


def _humanize_slack_alert(alert: str) -> str:
    lowered = alert.lower()
    if _is_openai_quota_alert(alert):
        return "OpenAI: превышена квота или не настроен биллинг; отчёт собран локальным резервным режимом."
    if "openai_api_key is not configured" in lowered:
        return "OpenAI: ключ не настроен; отчёт собран локальным резервным режимом."
    if "remote end closed connection" in lowered:
        return "OpenAI: соединение оборвалось во время генерации; отчёт собран локальным резервным режимом."
    if "ai report requested but generation fell back" in lowered:
        return "AI-обогащение не выполнено; использован локальный резервный режим."
    if "pdf generation failed" in lowered:
        return "PDF не собрался; Markdown/HTML артефакты сохранены."
    return _shorten(alert, limit=220)


def _is_openai_quota_alert(alert: str) -> bool:
    lowered = alert.lower()
    return "insufficient_quota" in lowered or "exceeded your current quota" in lowered


def _slack_artifact_files(
    *, output_dir: Path, pdf_path: Path | None
) -> list[dict[str, Any]]:
    candidates = [
        (
            "report_pdf",
            pdf_path or output_dir / "report.pdf",
            "PDF-протокол",
            "application/pdf",
        ),
        ("report_md", output_dir / "report.md", "Markdown-отчёт", "text/markdown"),
        (
            "transcript_md",
            output_dir / "transcript.md",
            "Markdown-транскрипт",
            "text/markdown",
        ),
    ]
    files: list[dict[str, Any]] = []
    for kind, path, title, mime_type in candidates:
        if path is None or not path.exists():
            continue
        size = path.stat().st_size
        files.append(
            {
                "kind": kind,
                "filename": path.name,
                "title": title,
                "mime_type": mime_type,
                "size_bytes": size,
                "size_human": _format_bytes(size),
            }
        )
    return files


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _ru_count(value: object, forms: tuple[str, str, str]) -> str:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        number = 0
    last_two = abs(number) % 100
    last = abs(number) % 10
    if 11 <= last_two <= 14:
        form = forms[2]
    elif last == 1:
        form = forms[0]
    elif 2 <= last <= 4:
        form = forms[1]
    else:
        form = forms[2]
    return f"{number} {form}"


def render_protocol_html(data: dict[str, Any]) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    if PROTOCOL_TEMPLATE_PATH.exists():
        env = Environment(
            loader=FileSystemLoader(PROTOCOL_TEMPLATE_PATH.parent),
            autoescape=select_autoescape(["html", "j2"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        rendered = env.get_template(PROTOCOL_TEMPLATE_PATH.name).render(**data)
    else:
        env = Environment(
            autoescape=select_autoescape(["html", "j2"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        rendered = env.from_string(PROTOCOL_TEMPLATE).render(**data, css=PROTOCOL_CSS)
    return _inject_protocol_font_faces(rendered)


def _inject_protocol_font_faces(html_text: str) -> str:
    font_css = _protocol_font_css()
    if not font_css or "<style>" not in html_text:
        return html_text
    return html_text.replace("<style>", f"<style>\n{font_css}\n", 1)


def _protocol_font_css() -> str:
    fonts = [
        ("Golos Text", "normal", "400 700", "GolosTextVar.ttf"),
        ("IBM Plex Mono", "normal", "400", "IBMPlexMono-Regular.ttf"),
        ("IBM Plex Mono", "normal", "600", "IBMPlexMono-Medium.ttf"),
    ]
    rules: list[str] = []
    for family, style, weight, filename in fonts:
        path = PROTOCOL_FONTS_DIR / filename
        if not path.exists():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            "@font-face { "
            f"font-family: '{family}'; font-style: {style}; font-weight: {weight}; "
            f"src: url(data:font/ttf;base64,{encoded}) format('truetype'); "
            "}"
        )
    return "\n".join(rules)


def _append_protocol_items_markdown(
    lines: list[str],
    title: str,
    items: list[dict[str, Any]],
    *,
    include_owner: bool = False,
) -> None:
    lines.extend([f"## {title}", ""])
    if not items:
        lines.extend(["Нет явно зафиксированных пунктов.", ""])
        return
    for item in items:
        owner = ""
        if include_owner:
            owner_value = item.get("owner") or "не назначен"
            owner = f" Ответственный: **{owner_value}**."
        note = f" _{item['note']}_" if item.get("note") else ""
        lines.append(f"- **{item['id']}** {item['text']}{owner} `{item['ref']}`{note}")
    lines.append("")


def _append_protocol_sections_markdown(
    lines: list[str], sections: list[dict[str, Any]]
) -> None:
    if not sections:
        return
    lines.extend(["## Смысловые разделы", ""])
    for section in sections:
        lines.extend([f"### {section['title']}", ""])
        if section.get("purpose"):
            lines.extend([str(section["purpose"]), ""])
        if section.get("summary"):
            lines.extend([str(section["summary"]), ""])
        items = section.get("items") or []
        if items:
            for item in items:
                ref = f" `{item['ref']}`" if item.get("ref") else ""
                lines.append(f"- **{item['title']}** {item['text']}{ref}")
    lines.append("")


def _protocol_report_items(
    prefix: str, items: list[ReportItem], report: MeetingReport
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for item in items:
        ref = _protocol_ref(report, item.citations)
        if not ref:
            continue
        rendered.append(
            {
                "id": f"{prefix}-{len(rendered) + 1:02d}",
                "text": _shorten(item.text, limit=360),
                "note": "",
                "ref": ref,
            }
        )
    return rendered


def _protocol_action_items(
    items: list[ActionItem], report: MeetingReport
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for item in items:
        ref = _protocol_ref(report, item.citations)
        if not ref:
            continue
        rendered.append(
            {
                "id": f"T-{len(rendered) + 1:02d}",
                "text": _shorten(item.text, limit=360),
                "note": f"Срок: {item.due}" if item.due else "",
                "owner": item.owner or None,
                "ref": ref,
            }
        )
    return rendered


def _dedupe_protocol_sections(*sections: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for section in sections:
        kept: list[dict[str, Any]] = []
        for item in section:
            key = re.sub(r"\W+", "", str(item.get("text", "")).lower())[:160]
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append(item)
        section[:] = kept


def _protocol_ref(report: MeetingReport, citations: list[str]) -> str:
    records = {record.segment_id: record for record in report.transcript}
    for segment_id in citations:
        record = records.get(segment_id)
        if record is not None:
            return f"{segment_id} · {fmt_time(record.start)}"
    return ""


def _protocol_tldr(report: MeetingReport) -> list[str]:
    points: list[str] = []
    for section in report.adaptive_sections:
        if section.kind in {
            "key_points",
            "architecture",
            "lecture_notes",
            "insights",
            "recommendations",
        }:
            for item in section.items:
                points.append(_shorten(item.text, limit=230))
                if len(points) >= 4:
                    return points
    for sentence in re.split(r"(?<=[.!?])\s+", report.overview):
        clean = sentence.strip()
        if clean:
            points.append(_shorten(clean, limit=230))
        if len(points) >= 4:
            break
    return points or ["Содержательные тезисы не выделены; смотрите полный транскрипт."]


def _protocol_timeline(report: MeetingReport) -> list[dict[str, str]]:
    blocks = [
        {
            "range": _timeline_range(block, report),
            "title": block.title,
            "summary": _shorten(block.summary, limit=520),
        }
        for block in report.timeline
    ]
    if len(blocks) <= 8:
        return blocks
    target = 6
    group_size = max(1, (len(blocks) + target - 1) // target)
    grouped: list[dict[str, str]] = []
    for index in range(0, len(blocks), group_size):
        group = blocks[index : index + group_size]
        grouped.append(
            {
                "range": f"{group[0]['range'].split('–')[0].strip()} – {group[-1]['range'].split('–')[-1].strip()}",
                "title": _shorten(group[0]["title"], limit=90),
                "summary": _shorten(
                    " ".join(item["summary"] for item in group), limit=760
                ),
            }
        )
    return grouped[:8]


def _timeline_range(block: TimelineBlock, report: MeetingReport) -> str:
    records = {record.segment_id: record for record in report.transcript}
    cited = [records[citation] for citation in block.citations if citation in records]
    if cited:
        return f"{fmt_time(min(record.start for record in cited))} – {fmt_time(max(record.end for record in cited))}"
    return block.title.replace("-", " – ")


def _protocol_doc_id(report: MeetingReport, generated_at: datetime) -> str:
    clean = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "", report.meeting_id.upper())[:6]
    return f"PRT-{generated_at.strftime('%y%m%d')}-{clean or 'MEET'}"


def _protocol_date_short(source_filename: str, fallback: datetime) -> str:
    parsed = _protocol_datetime_from_source(source_filename)
    if parsed is not None:
        return parsed.strftime("%d.%m.%Y")
    return fallback.strftime("%d.%m.%Y")


def _protocol_date_human(source_filename: str, fallback: datetime) -> str:
    parsed = _protocol_datetime_from_source(source_filename)
    if parsed is None:
        return _protocol_date_short(source_filename, fallback)
    weekdays = [
        "Понедельник",
        "Вторник",
        "Среда",
        "Четверг",
        "Пятница",
        "Суббота",
        "Воскресенье",
    ]
    months = [
        "",
        "января",
        "февраля",
        "марта",
        "апреля",
        "мая",
        "июня",
        "июля",
        "августа",
        "сентября",
        "октября",
        "ноября",
        "декабря",
    ]
    return f"{weekdays[parsed.weekday()]}, {parsed.day} {months[parsed.month]} {parsed.year}"


def _protocol_time_human(source_filename: str) -> str:
    parsed = _protocol_datetime_from_source(source_filename)
    if parsed is None:
        return ""
    match = re.search(
        r"\b(20\d{2})[-_\.](\d{2})[-_\.](\d{2})\s+(\d{2})[-_:](\d{2})\s*UTC\b",
        source_filename,
    )
    if not match:
        return ""
    msk = parsed + timedelta(hours=3)
    return f"{msk:%H:%M} МСК ({parsed:%H:%M} UTC)"


def _protocol_datetime_from_source(source_filename: str) -> datetime | None:
    match = re.search(
        r"\b(20\d{2})[-_\.](\d{2})[-_\.](\d{2})(?:\s+(\d{2})[-_:](\d{2})\s*UTC)?\b",
        source_filename,
    )
    if not match:
        return None
    year, month, day, hour, minute = match.groups()
    return datetime(
        int(year),
        int(month),
        int(day),
        int(hour or "0"),
        int(minute or "0"),
        tzinfo=UTC,
    )


PROTOCOL_TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{{ meeting.title }}</title>
  <style>{{ css }}</style>
</head>
<body>
  <main class="protocol-page">
    <header class="cover">
      <div class="doc-id">{{ meeting.doc_id }}</div>
      <h1>{{ meeting.title }}</h1>
      <dl class="meta-grid">
        <div><dt>Дата</dt><dd>{{ meeting.date_short }}</dd></div>
        <div><dt>Длительность</dt><dd>{{ meeting.duration }}</dd></div>
        <div><dt>Сегментов</dt><dd>{{ meeting.segments }}</dd></div>
        <div><dt>Генератор</dt><dd>{{ meeting.generator }}</dd></div>
      </dl>
      <p class="artifact-note">Полный транскрипт: {{ meeting.transcript }} · Coverage: {{ meeting.coverage }}</p>
    </header>

    {% if warnings %}
    <section class="alert">
      <h2>Проверить</h2>
      <ul>{% for warning in warnings %}<li>{{ warning }}</li>{% endfor %}</ul>
    </section>
    {% endif %}

    <section class="kpis">
      {% for kpi in kpis %}
      <div><strong>{{ kpi.num }}</strong><span>{{ kpi.cap }}</span></div>
      {% endfor %}
    </section>

    <section>
      <h2>TL;DR</h2>
      <ul class="tldr">{% for item in tldr %}<li>{{ item }}</li>{% endfor %}</ul>
    </section>

    <section>
      <h2>Решения</h2>
      {% for item in decisions %}
      <article class="item-row"><b>{{ item.id }}</b><p>{{ item.text }}{% if item.note %}<em>{{ item.note }}</em>{% endif %}</p><span>{{ item.ref }}</span></article>
      {% else %}<p class="empty">Нет явно зафиксированных решений.</p>{% endfor %}
    </section>

    <section>
      <h2>Задачи</h2>
      {% for item in tasks %}
      <article class="item-row"><b>{{ item.id }}</b><p>{{ item.text }}<small>Ответственный: {{ item.owner or "не назначен" }}</small>{% if item.note %}<em>{{ item.note }}</em>{% endif %}</p><span>{{ item.ref }}</span></article>
      {% else %}<p class="empty">Нет явно зафиксированных задач.</p>{% endfor %}
      <p class="footnote">{{ tasks_footnote }}</p>
    </section>

    <section>
      <h2>Открытые вопросы</h2>
      {% for item in questions %}
      <article class="item-row"><b>{{ item.id }}</b><p>{{ item.text }}</p><span>{{ item.ref }}</span></article>
      {% else %}<p class="empty">Нет явно зафиксированных вопросов.</p>{% endfor %}
    </section>

    <section>
      <h2>Риски</h2>
      {% for item in risks %}
      <article class="item-row risk"><b>{{ item.id }}</b><p>{{ item.text }}{% if item.note %}<em>{{ item.note }}</em>{% endif %}</p><span>{{ item.ref }}</span></article>
      {% else %}<p class="empty">Нет явно зафиксированных рисков.</p>{% endfor %}
    </section>

    <section>
      <h2>Хроника</h2>
      {% for block in timeline %}
      <article class="timeline-block"><time>{{ block.range }}</time><h3>{{ block.title }}</h3><p>{{ block.summary }}</p></article>
      {% endfor %}
    </section>

    <footer>
      <span>Источник: {{ meeting.source }}</span>
      <span>Участники: {{ participants|join(", ") }}</span>
    </footer>
  </main>
</body>
</html>
"""


PROTOCOL_CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
* { box-sizing: border-box; }
body {
  margin: 0;
  color: #171717;
  background: #fff;
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.42;
}
.protocol-page { max-width: 178mm; margin: 0 auto; }
.cover {
  border-bottom: 1.4pt solid #171717;
  margin-bottom: 8mm;
  padding-bottom: 5mm;
}
.doc-id {
  color: #666;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 8.5pt;
  font-weight: 700;
  letter-spacing: .08em;
}
h1 {
  margin: 7mm 0 5mm;
  max-width: 150mm;
  font-size: 23pt;
  line-height: 1.08;
}
h2 {
  break-after: avoid;
  margin: 8mm 0 3mm;
  border-bottom: .55pt solid #d4d4d4;
  padding-bottom: 1.5mm;
  font-size: 13pt;
}
h3 { margin: 0 0 1mm; font-size: 10.5pt; }
.meta-grid, .kpis {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 2mm;
}
.meta-grid { margin: 0; }
.meta-grid div, .kpis div {
  border: .6pt solid #d4d4d4;
  padding: 2.5mm;
}
dt, .kpis span {
  color: #737373;
  font-size: 7.5pt;
  font-weight: 700;
  text-transform: uppercase;
}
dd { margin: 1mm 0 0; }
.kpis strong { display: block; font-size: 18pt; }
.artifact-note, .footnote, footer {
  color: #666;
  font-size: 8.5pt;
}
.artifact-note { margin: 4mm 0 0; }
.alert {
  border: .7pt solid #f59e0b;
  background: #fffbeb;
  padding: 3mm 4mm;
}
.alert h2 { margin-top: 0; border: 0; padding: 0; color: #92400e; }
.tldr { padding-left: 5mm; }
.tldr li { margin-bottom: 1.8mm; }
.item-row {
  display: grid;
  grid-template-columns: 15mm 1fr 24mm;
  gap: 3mm;
  break-inside: avoid;
  border-bottom: .45pt solid #e5e5e5;
  padding: 2.5mm 0;
}
.item-row b {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 9pt;
}
.item-row p { margin: 0; }
.item-row span {
  color: #666;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 8pt;
  text-align: right;
}
.item-row small, .item-row em {
  display: block;
  margin-top: 1mm;
  color: #666;
  font-size: 8.5pt;
}
.risk b { color: #b91c1c; }
.timeline-block {
  break-inside: avoid;
  border-left: 1.4pt solid #171717;
  margin: 0 0 4mm;
  padding-left: 4mm;
}
.timeline-block time {
  color: #666;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 8.5pt;
}
.timeline-block p { margin: 0; }
.empty { color: #737373; }
footer {
  display: grid;
  gap: 1mm;
  margin-top: 9mm;
  border-top: .55pt solid #d4d4d4;
  padding-top: 3mm;
}
"""


EXECUTIVE_MEMO_CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #f1f2f4;
  color: #111827;
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Arial, sans-serif;
}
.page {
  width: 794px;
  min-height: 1123px;
  margin: 0 auto;
  padding: 74px 82px 64px;
  background: #fff;
}
header {
  border-bottom: 2px solid #111827;
  padding-bottom: 18px;
  margin-bottom: 26px;
}
.kicker {
  font-size: 12px;
  font-weight: 700;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: .08em;
}
h1 {
  margin: 34px 0 12px;
  max-width: 570px;
  font-size: 38px;
  line-height: 1.04;
  letter-spacing: -0.02em;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 13px;
  color: #475569;
}
.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border-top: 1px solid #d1d5db;
  border-bottom: 1px solid #d1d5db;
  margin: 22px 0 26px;
}
.stats div { padding: 12px 14px 11px 0; }
.stats span {
  display: block;
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: .05em;
}
.stats strong {
  display: block;
  margin-top: 6px;
  font-size: 24px;
}
.summary {
  margin: 0 0 26px;
  font-size: 16px;
  line-height: 1.5;
}
.report-section {
  border-top: 1px solid #d9dde5;
  padding-top: 4px;
  break-inside: avoid;
}
h2 {
  margin: 28px 0 10px;
  font-size: 18px;
  letter-spacing: -0.01em;
}
.section-purpose {
  margin: -4px 0 10px;
  color: #475569;
  font-size: 13px;
}
.item {
  display: grid;
  grid-template-columns: 112px 1fr;
  gap: 16px;
  padding: 10px 0;
  border-bottom: 1px solid #edf0f4;
  break-inside: avoid;
}
.item p {
  margin: 0;
  line-height: 1.42;
}
.item .cite {
  grid-column: 2;
  margin-top: -4px;
}
.cite {
  color: #64748b;
  font-size: 11px;
  white-space: nowrap;
}
.coverage { margin-top: 26px; }
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  text-align: left;
  border-bottom: 1px solid #e5e7eb;
  padding: 7px 0;
  vertical-align: top;
}
th {
  font-size: 11px;
  color: #64748b;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .03em;
}
footer {
  margin-top: 26px;
  color: #64748b;
  font-size: 12px;
  border-top: 1px solid #e5e7eb;
  padding-top: 12px;
}
""".strip()


def render_report_typst(report: MeetingReport) -> str:
    lines = [
        '#import "@preview/basic-report:0.5.0": *',
        "",
        "#show: it => basic-report(",
        '  doc-category: "Отчёт по транскрипту",',
        f"  doc-title: {typst_report_title(report.title)},",
        '  author: "GigaAM",',
        f"  affiliation: {typst_string(report.source_filename + ' · ' + report.model_name + ' · ' + report.generated_by)},",
        '  language: "ru",',
        "  show-outline: false,",
        "  compact-mode: true,",
        '  heading-color: rgb("#2457B8"),',
        '  heading-font: ("Helvetica Neue", "Arial"),',
        '  body-font: ("Helvetica Neue", "Arial"),',
        "  it,",
        ")",
        "",
        "#set table(",
        "  stroke: 0.35pt + luma(78%),",
        "  inset: (x: 5pt, y: 4pt),",
        ")",
        '#show table.cell.where(y: 0): set text(weight: "bold")',
        "",
        "= Сводка отчета",
        "",
        "#table(",
        "  columns: (1fr, 1fr, 1fr, 1fr),",
        "  [Длительность], [Сегментов], [Решений], [Задач],",
        f"  [{typst_markup(fmt_time(report.duration))}],",
        f"  [{typst_markup(str(report.segment_count))}],",
        f"  [{typst_markup(str(len(report.decisions)))}],",
        f"  [{typst_markup(str(len(report.action_items)))}],",
        ")",
        "",
        typst_markup(report.overview),
        "",
        "= Структура отчета",
        "",
        f"*Тип записи:* {typst_markup(report.profile.label)}",
        "",
        f"*Уверенность:* {report.profile.confidence}",
        "",
        typst_markup(report.profile.rationale),
        "",
    ]
    if report.warnings:
        lines.extend(["= Предупреждения", ""])
        for warning in report.warnings:
            lines.extend([f"- {typst_markup(warning)}", ""])
        lines.append("")
    lines.extend(["= Разделы отчета", ""])
    _append_adaptive_sections_typst(lines, report)
    _append_timeline_typst(lines, report)
    _append_coverage_typst(lines, report)
    _append_pdf_source_note_typst(lines, report)
    return "\n".join(lines).rstrip() + "\n"


def render_report_pdf(*, html_path: Path, pdf_path: Path) -> Path | None:
    _ensure_fontconfig_cache()
    try:
        from weasyprint import HTML
    except Exception:
        HTML = None
    if HTML is not None:
        try:
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise ReportGenerationError(f"weasyprint failed: {exc}") from exc
        return pdf_path

    playwright_bin = os.environ.get("MAC_TRANSCRIBER_PLAYWRIGHT") or shutil.which(
        "playwright"
    )
    if playwright_bin is None:
        return None
    completed = subprocess.run(
        [
            playwright_bin,
            "pdf",
            "--paper-format",
            "A4",
            "--wait-for-selector",
            ".hero, body",
            html_path.resolve().as_uri(),
            str(pdf_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "pdf render failed").strip()
        raise ReportGenerationError(message)
    return pdf_path


def _ensure_fontconfig_cache() -> None:
    if os.environ.get("XDG_CACHE_HOME"):
        return
    cache_dir = Path(
        os.environ.get(
            "MAC_TRANSCRIBER_FONT_CACHE_DIR", "/tmp/mac-transcriber-font-cache"
        )
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def report_to_dict(report: MeetingReport) -> dict[str, Any]:
    return asdict(report)


def build_report_health(
    *,
    report: MeetingReport,
    output_dir: Path,
    requested_ai: bool,
    requested_pdf: bool,
    pdf_path: Path | None,
    pdf_error: str | None,
) -> dict[str, Any]:
    checks: list[dict[str, object]] = []
    alerts: list[str] = []

    def add_check(
        name: str, ok: bool, fail_detail: str, ok_detail: str | None = None
    ) -> None:
        details = ok_detail if ok and ok_detail is not None else fail_detail
        checks.append({"name": name, "ok": ok, "details": details})
        if not ok and fail_detail not in alerts:
            alerts.append(fail_detail)

    artifact_files = {
        "report_json": output_dir / "report.json",
        "report_md": output_dir / "report.md",
        "report_html": output_dir / "report.html",
        "report_typ": output_dir / "report.typ",
        "report_pdf": output_dir / "report.pdf",
        "coverage_json": output_dir / "coverage.json",
    }
    artifact_status = {
        name: {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in artifact_files.items()
    }

    add_check(
        "transcript_has_segments",
        report.segment_count > 0,
        "Transcript has no speech segments; report cannot contain meeting details.",
        f"Transcript contains {report.segment_count} speech segments.",
    )
    coverage_ids = [entry.segment_id for entry in report.coverage]
    known_ids = {record.segment_id for record in report.transcript}
    add_check(
        "coverage_complete",
        set(coverage_ids) == known_ids and len(coverage_ids) == len(set(coverage_ids)),
        "Coverage audit does not cover every transcript segment exactly once.",
        "Coverage audit covers every transcript segment exactly once.",
    )
    add_check(
        "report_markdown_nonempty",
        artifact_status["report_md"]["size_bytes"]
        >= (512 if report.segment_count else 1),
        "Markdown report is unexpectedly small.",
        f"Markdown report size is {artifact_status['report_md']['size_bytes']} bytes.",
    )
    add_check(
        "report_html_exists",
        bool(artifact_status["report_html"]["exists"]),
        "HTML report artifact is missing.",
        "HTML report artifact exists.",
    )
    add_check(
        "report_json_exists",
        bool(artifact_status["report_json"]["exists"]),
        "JSON report artifact is missing.",
        "JSON report artifact exists.",
    )
    protocol_ok, protocol_detail = _protocol_json_shape_status(
        artifact_files["report_json"]
    )
    add_check(
        "protocol_json_shape",
        protocol_ok,
        protocol_detail,
        protocol_detail,
    )
    add_check(
        "coverage_json_exists",
        bool(artifact_status["coverage_json"]["exists"]),
        "Coverage JSON artifact is missing.",
        "Coverage JSON artifact exists.",
    )
    html_ok, html_detail = _protocol_html_status(artifact_files["report_html"])
    add_check(
        "protocol_html_template",
        html_ok,
        html_detail,
        html_detail,
    )
    transcript_path = output_dir / "transcript.md"
    if transcript_path.exists():
        inventory_ok, inventory_detail = _transcript_inventory_status(
            transcript_path,
            expected_segments=report.segment_count,
        )
        add_check(
            "transcript_markdown_inventory",
            inventory_ok,
            inventory_detail,
            inventory_detail,
        )
    blocking_warnings = _report_blocking_warnings(report.warnings)
    add_check(
        "report_has_no_blocking_warnings",
        not blocking_warnings,
        _report_warning_alert(blocking_warnings),
        "Report was generated without blocking warnings.",
    )
    if requested_ai:
        add_check(
            "ai_report_succeeded",
            report.generated_by != "local"
            and not any(
                "AI report fallback" in warning for warning in blocking_warnings
            ),
            "AI report requested but generation fell back to the local report.",
            f"AI report generated by {report.generated_by}.",
        )
        synthesis_warnings = [
            warning
            for warning in report.warnings
            if "AI synthesis fallback" in warning or "AI synthesis skipped" in warning
        ]
        if report.generated_by.endswith("/chunked") or synthesis_warnings:
            add_check(
                "ai_synthesis_succeeded",
                not synthesis_warnings,
                _report_warning_alert(synthesis_warnings),
                "AI chunk synthesis completed without warnings.",
            )
    if requested_pdf:
        add_check(
            "pdf_generated",
            pdf_path is not None and pdf_path.exists() and pdf_error is None,
            f"PDF generation failed: {pdf_error}"
            if pdf_error
            else "PDF report artifact is missing.",
            "PDF report artifact exists.",
        )
        if pdf_path is not None and pdf_path.exists() and pdf_error is None:
            pdf_text_ok, pdf_text_detail = _pdf_content_status(pdf_path)
            add_check(
                "pdf_content_gate",
                pdf_text_ok,
                pdf_text_detail,
                pdf_text_detail,
            )

    status = "ok" if all(bool(check["ok"]) for check in checks) else "degraded"
    if report.segment_count == 0:
        status = "failed"

    return {
        "status": status,
        "generated_by": report.generated_by,
        "requested_ai": requested_ai,
        "requested_pdf": requested_pdf,
        "segment_count": report.segment_count,
        "duration_seconds": report.duration,
        "alerts": alerts,
        "warnings": list(report.warnings),
        "checks": checks,
        "artifacts": artifact_status,
    }


def _report_warning_alert(warnings: list[str]) -> str:
    if not warnings:
        return "Report was generated with warnings; inspect report_health.json."
    visible = "; ".join(_shorten(str(warning), limit=240) for warning in warnings[:3])
    if len(warnings) > 3:
        visible = f"{visible}; +{len(warnings) - 3} more warning(s)"
    return f"Report was generated with warnings: {visible}"


def _report_blocking_warnings(warnings: list[str]) -> list[str]:
    return [
        warning
        for warning in warnings
        if not str(warning).startswith("AI baseline upgraded:")
    ]


def _protocol_json_shape_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "Protocol JSON artifact is missing."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"Protocol JSON is not readable: {exc}"
    required_top = {
        "meeting",
        "tldr",
        "decisions",
        "tasks",
        "questions",
        "risks",
        "timeline",
        "kpis",
    }
    missing = sorted(required_top.difference(data))
    if missing:
        return False, f"Protocol JSON misses required sections: {', '.join(missing)}."
    meeting = data.get("meeting")
    if not isinstance(meeting, dict) or not meeting.get("title"):
        return False, "Protocol JSON meeting block is missing title."
    for key in ("tldr", "decisions", "tasks", "questions", "risks", "timeline", "kpis"):
        if not isinstance(data.get(key), list):
            return False, f"Protocol JSON section {key} must be a list."
    return True, "Protocol JSON has the required protocol sections."


def _protocol_html_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "Protocol HTML artifact is missing."
    text = path.read_text(encoding="utf-8", errors="replace")
    required = ["Протокол встречи", "Главное", "@font-face"]
    missing = [token for token in required if token not in text]
    if missing:
        return False, f"Protocol HTML misses template markers: {', '.join(missing)}."
    if "Coverage audit" in text or "## Полный транскрипт" in text:
        return False, "Protocol HTML contains an embedded audit/transcript section."
    return True, "Protocol HTML uses the bundled protocol template and embedded fonts."


def _transcript_inventory_status(
    path: Path, *, expected_segments: int
) -> tuple[bool, str]:
    if not path.exists():
        return False, "Transcript Markdown artifact is missing."
    text = path.read_text(encoding="utf-8", errors="replace")
    declared = None
    match = re.search(r"Segments:\s*(\d+)", text[:2000])
    if match:
        declared = int(match.group(1))
    rows = 0
    raw_count = 0
    explicit_ids = 0
    for line in text.splitlines():
        match = re.match(
            r"^(?:\[(S\d{4})(?:\s*-\s*(S\d{4}))?\]\s*)?"
            r"\[\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\]\s*\*\*.+?:\*\*",
            line.strip(),
        )
        if not match:
            continue
        rows += 1
        sid_a, sid_b = match.groups()
        if sid_a:
            explicit_ids += 1
            raw_count += int((sid_b or sid_a)[1:]) - int(sid_a[1:]) + 1
        else:
            raw_count += 1
    if rows == 0:
        return False, "Transcript Markdown has no timestamped transcript rows."
    expected = declared or expected_segments
    if raw_count != expected_segments or expected != expected_segments:
        return (
            False,
            f"Transcript inventory mismatch: {rows} rows cover {raw_count} raw segments; expected {expected_segments}, declared {declared}.",
        )
    if explicit_ids == 0 and rows != expected_segments:
        return (
            False,
            "Transcript Markdown merged rows without explicit raw S-ID ranges.",
        )
    return True, f"Transcript inventory OK: {rows} rows cover {raw_count} raw segments."


def _pdf_content_status(path: Path) -> tuple[bool, str]:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        return False, "Cannot validate PDF text because pdftotext is not installed."
    completed = subprocess.run(
        [pdftotext, str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "pdftotext failed").strip()
        return False, f"Cannot extract PDF text: {message}"
    text = completed.stdout
    forbidden_patterns = [
        r"(?m)^\[S\d{4}(?:-S\d{4})?\]\s+\[\d{2}:\d{2}:\d{2}",
        r"Coverage audit",
        r"## Полный транскрипт",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, text):
            return (
                False,
                "PDF contains embedded transcript rows or coverage audit text.",
            )
    if "Протокол встречи" not in text and "ПРОТОКОЛ ВСТРЕЧИ" not in text:
        return False, "PDF text does not contain the protocol heading."
    return (
        True,
        "PDF content gate OK: protocol text extracts, full transcript and coverage audit are not embedded.",
    )


def load_openai_api_key(env_path: Path | None = None) -> str | None:
    return _load_env_value(
        "OPENAI_API_KEY",
        env_path=env_path,
        configured_env_file="MAC_TRANSCRIBER_OPENAI_ENV_FILE",
    )


def load_openrouter_api_key(env_path: Path | None = None) -> str | None:
    return _load_env_value(
        "OPENROUTER_API_KEY",
        env_path=env_path,
        configured_env_file="MAC_TRANSCRIBER_OPENROUTER_ENV_FILE",
    )


def openrouter_fallback_model(env_path: Path | None = None) -> str:
    return (
        _load_env_value(
            "MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL",
            env_path=env_path,
            configured_env_file="MAC_TRANSCRIBER_OPENROUTER_ENV_FILE",
        )
        or DEFAULT_OPENROUTER_FALLBACK_MODEL
    )


def baseline_upgrade_model(env_path: Path | None = None) -> str:
    return (
        _load_env_value(
            "MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL",
            env_path=env_path,
            configured_env_file="MAC_TRANSCRIBER_OPENAI_ENV_FILE",
        )
        or ""
    ).strip()


def report_critic_model(env_path: Path | None = None) -> str:
    """Модель финального прохода-критика; пусто ⇒ критик выключен (дефолт).

    MAC_TRANSCRIBER_REPORT_CRITIC_MODEL включает дедуп/кросс-дедуп/rewrite формальных
    разделов поверх готового AI-отчёта. Принимает обычную OpenAI-модель или
    'openrouter:<model>'.
    """
    return (
        _load_env_value(
            "MAC_TRANSCRIBER_REPORT_CRITIC_MODEL",
            env_path=env_path,
            configured_env_file="MAC_TRANSCRIBER_OPENAI_ENV_FILE",
        )
        or ""
    ).strip()


def openrouter_max_tokens(env_path: Path | None = None) -> int:
    raw_value = _load_env_value(
        "MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS",
        env_path=env_path,
        configured_env_file="MAC_TRANSCRIBER_OPENROUTER_ENV_FILE",
    )
    if not raw_value:
        return 12000
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 12000


def _load_env_value(
    key_name: str,
    *,
    env_path: Path | None = None,
    configured_env_file: str | None = None,
) -> str | None:
    key = os.environ.get(key_name)
    if key:
        return key.strip()
    configured = os.environ.get(configured_env_file) if configured_env_file else None
    candidates = [Path(configured)] if configured else [env_path or Path(".env.local")]
    for candidate in candidates:
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == key_name and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def ai_report_schema() -> dict[str, Any]:
    item = _ai_report_item_schema()
    action = _ai_action_item_schema()
    timeline = _ai_timeline_schema()
    profile = _ai_profile_schema()
    adaptive_section = _ai_adaptive_section_schema(item)
    coverage = {
        "type": "object",
        "properties": {
            "segment_id": {"type": "string"},
            "status": {"type": "string", "enum": sorted(COVERAGE_STATUSES)},
            "section_titles": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["segment_id", "status", "section_titles", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "profile": profile,
            "overview": {"type": "string"},
            "adaptive_sections": {"type": "array", "items": adaptive_section},
            "coverage": {"type": "array", "items": coverage},
            "timeline": {"type": "array", "items": timeline},
            "decisions": {"type": "array", "items": item},
            "action_items": {"type": "array", "items": action},
            "open_questions": {"type": "array", "items": item},
            "risks": {"type": "array", "items": item},
            "notable_quotes": {"type": "array", "items": item},
        },
        "required": [
            "profile",
            "overview",
            "adaptive_sections",
            "coverage",
            "timeline",
            "decisions",
            "action_items",
            "open_questions",
            "risks",
            "notable_quotes",
        ],
        "additionalProperties": False,
    }


def ai_chunk_notes_schema() -> dict[str, Any]:
    item = _ai_report_item_schema()
    action = _ai_action_item_schema()
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": item},
            "decisions": {"type": "array", "items": item},
            "action_items": {"type": "array", "items": action},
            "open_questions": {"type": "array", "items": item},
            "risks": {"type": "array", "items": item},
            "notable_quotes": {"type": "array", "items": item},
        },
        "required": [
            "summary",
            "key_points",
            "decisions",
            "action_items",
            "open_questions",
            "risks",
            "notable_quotes",
        ],
        "additionalProperties": False,
    }


def ai_synthesis_schema() -> dict[str, Any]:
    item = _ai_report_item_schema()
    action = _ai_action_item_schema()
    timeline = _ai_timeline_schema()
    profile = _ai_profile_schema()
    adaptive_section = _ai_adaptive_section_schema(item)
    return {
        "type": "object",
        "properties": {
            "profile": profile,
            "overview": {"type": "string"},
            "adaptive_sections": {"type": "array", "items": adaptive_section},
            "timeline": {"type": "array", "items": timeline},
            "decisions": {"type": "array", "items": item},
            "action_items": {"type": "array", "items": action},
            "open_questions": {"type": "array", "items": item},
            "risks": {"type": "array", "items": item},
            "notable_quotes": {"type": "array", "items": item},
        },
        "required": [
            "profile",
            "overview",
            "adaptive_sections",
            "timeline",
            "decisions",
            "action_items",
            "open_questions",
            "risks",
            "notable_quotes",
        ],
        "additionalProperties": False,
    }


def ai_memory_enrichment_schema() -> dict[str, Any]:
    item = _ai_report_item_schema()
    adaptive_section = _ai_adaptive_section_schema(item)
    return {
        "type": "object",
        "properties": {
            "overview_addendum": {"type": "string"},
            "memory_sections": {"type": "array", "items": adaptive_section},
        },
        "required": ["overview_addendum", "memory_sections"],
        "additionalProperties": False,
    }


def ai_report_critic_schema() -> dict[str, Any]:
    op = _ai_critic_op_schema()
    ops = {"type": "array", "items": op}
    return {
        "type": "object",
        "properties": {
            "decisions": ops,
            "action_items": ops,
            "open_questions": ops,
            "risks": ops,
            "notes": {"type": "string"},
        },
        "required": ["decisions", "action_items", "open_questions", "risks", "notes"],
        "additionalProperties": False,
    }


def _ai_critic_op_schema() -> dict[str, Any]:
    # strict json_schema требует все поля в required; неиспользуемые передаются
    # пустыми ("" / []). Интерпретация зависит от op (см. _apply_critic_ops):
    #   keep/drop/rewrite -> id; merge -> ids; rewrite/merge -> title/text.
    return {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["keep", "drop", "rewrite", "merge"]},
            "id": {"type": "string"},
            "ids": {"type": "array", "items": {"type": "string"}},
            "title": {"type": "string"},
            "text": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["op", "id", "ids", "title", "text", "reason"],
        "additionalProperties": False,
    }


def _ai_report_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "text": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "text", "citations"],
        "additionalProperties": False,
    }


def _ai_action_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "text": {"type": "string"},
            "owner": {"type": "string"},
            "due": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "text", "owner", "due", "citations"],
        "additionalProperties": False,
    }


def _ai_timeline_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "summary", "citations"],
        "additionalProperties": False,
    }


def _ai_profile_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "project_sync",
                    "lecture",
                    "research_interview",
                    "technical_discussion",
                    "consultation",
                    "general",
                ],
            },
            "label": {"type": "string"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["kind", "label", "confidence", "rationale"],
        "additionalProperties": False,
    }


def _ai_adaptive_section_schema(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "title": {"type": "string"},
            "purpose": {"type": "string"},
            "summary": {"type": "string"},
            "items": {"type": "array", "items": item},
            "citations": {"type": "array", "items": {"type": "string"}},
            "accent": {"type": "string"},
        },
        "required": [
            "kind",
            "title",
            "purpose",
            "summary",
            "items",
            "citations",
            "accent",
        ],
        "additionalProperties": False,
    }


def fmt_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def typst_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def typst_report_title(value: str) -> str:
    return typst_string(_wrap_report_title(str(value)))


def _wrap_report_title(value: str) -> str:
    clean = " ".join(value.split())
    if len(clean) <= 28:
        return clean
    words = clean.split()
    if len(words) <= 1:
        return clean

    lines: list[str] = []
    current: list[str] = []
    max_chars = 18

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        line = " ".join(current)
        if not lines and len(current) == 2 and len(line) < 14:
            lines.append(current[0])
            current = [current[1]]
            return
        lines.append(line)
        current = []

    for word in words:
        candidate = " ".join([*current, word])
        if not current or len(candidate) <= max_chars:
            current.append(word)
            continue
        flush_current()
        current.append(word)
    flush_current()
    return "\n".join(lines)


def typst_markup(value: str) -> str:
    text = str(value)
    for char in ["\\", "#", "$", "%", "&", "_", "*", "@", "`"]:
        text = text.replace(char, "\\" + char)
    return text


def _local_overview(records: list[TranscriptRecord]) -> str:
    if not records:
        return "Транскрипт пустой: содержательных речевых сегментов не найдено."
    speakers = ", ".join(sorted({record.speaker for record in records}))
    return (
        "Отчёт собран по транскрипту: выделены ключевые темы, таймлайн, решения, "
        "задачи, вопросы и риски. Полный текст записи остаётся в Markdown-артефакте. "
        f"Участники/спикеры: {speakers or 'не определены'}."
    )


def _local_timeline(records: list[TranscriptRecord]) -> list[TimelineBlock]:
    blocks: list[TimelineBlock] = []
    for chunk in _chunks(records, size=8):
        title = f"{fmt_time(chunk[0].start)}-{fmt_time(chunk[-1].end)}"
        summary = _shorten(" ".join(record.text for record in chunk), limit=700)
        blocks.append(
            TimelineBlock(
                title=title,
                summary=summary,
                citations=[record.segment_id for record in chunk],
            )
        )
    return blocks


def _matching_items(
    records: list[TranscriptRecord],
    *,
    title_prefix: str,
    patterns: list[str],
) -> list[ReportItem]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    items: list[ReportItem] = []
    for record in records:
        if any(pattern.search(record.text) for pattern in compiled):
            items.append(
                ReportItem(
                    title=f"{title_prefix} {len(items) + 1}",
                    text=record.text,
                    citations=[record.segment_id],
                )
            )
    return items


def _local_action_items(records: list[TranscriptRecord]) -> list[ActionItem]:
    compiled = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\b(?:нужно|надо)\s+(?:добавить|обновить|перенести|подвинуть|отметить|оценить|заполнить|создать|проверить|подготовить|расписать|закрыть|согласовать|завести|разбить|сделать|сохранить|поправить|исправить|пройтись|загрузить|убрать|уменьшить)\b",
            r"\b(?:сделаю|добавлю|создам|проверю|подготовлю|распишу|перенесу|обновлю|сохраню|поправлю|исправлю)\b",
            r"\bследующий шаг\b",
            r"\baction\b",
            r"\btodo\b",
        ]
    ]
    items: list[ActionItem] = []
    for record in records:
        if re.search(r"\bне\s+(?:надо|нужно)\b", record.text, re.IGNORECASE):
            continue
        if _looks_like_methodology_action_noise(record.text):
            continue
        if any(pattern.search(record.text) for pattern in compiled):
            items.append(
                ActionItem(
                    title=f"Задача {len(items) + 1}",
                    text=record.text,
                    owner=record.speaker,
                    due="",
                    citations=[record.segment_id],
                )
            )
    return items


def _looks_like_methodology_action_noise(text: str) -> bool:
    lowered = str(text).lower()
    noise_patterns = [
        r"\bметодологическ\w*\b",
        r"\bв\s+(?:отч[её]те|протоколе)\s+(?:нужно|надо|следует)\b",
        r"\b(?:нужно|надо|следует)\s+(?:понимать|помнить|учитывать|иметь в виду)\b",
        r"\bважно\s+(?:понимать|помнить|учитывать)\b",
        r"\bследует\s+понимать\b",
    ]
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in noise_patterns)


def _is_local_open_question(text: str) -> bool:
    clean = " ".join(str(text).split())
    lowered = clean.lower()
    words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", clean)
    if len(words) < 4 or len(clean) < 14:
        return False
    if re.search(r"\bоткрыт\w*\s+вопрос\b", lowered):
        return True
    if re.search(r"\b(?:надо|нужно)\s+(?:понять|решить)\b", lowered):
        return True
    if "?" not in clean:
        return False
    if re.search(
        r"\bчто\b.*\b(?:делать|идти|должно|дальше|после|будет|план)\b", lowered
    ):
        return True
    if re.search(
        r"\bкак\b.*\b(?:проверять|отображать|попадать|согласовать|закрыть)\b", lowered
    ):
        return True
    if re.search(r"\bкогда\b.*\b(?:начн|будет|законч)\b", lowered):
        return True
    return bool(re.search(r"\b(?:можем ли|надо ли|нужно ли|стоит ли)\b", lowered))


def _notable_quotes(records: list[TranscriptRecord]) -> list[ReportItem]:
    scored = sorted(records, key=lambda record: len(record.text), reverse=True)
    return [
        ReportItem(
            title=f"Цитата {index}",
            text=record.text,
            citations=[record.segment_id],
        )
        for index, record in enumerate(scored[: min(5, len(scored))], start=1)
    ]


def _unique_citations(items: list[ReportItem]) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for item in items:
        for segment_id in item.citations:
            if segment_id not in seen:
                seen.add(segment_id)
                citations.append(segment_id)
    return citations


def _clone_report_items(items: list[ReportItem]) -> list[ReportItem]:
    return [
        ReportItem(
            title=item.title,
            text=item.text,
            citations=list(item.citations),
        )
        for item in items
    ]


def _unique_action_citations(items: list[ActionItem]) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for item in items:
        for segment_id in item.citations:
            if segment_id not in seen:
                seen.add(segment_id)
                citations.append(segment_id)
    return citations


def _action_items_as_report_items(items: list[ActionItem]) -> list[ReportItem]:
    return [
        ReportItem(
            title=item.title,
            text=f"{item.text} Ответственный: {item.owner or 'не указан'}",
            citations=item.citations,
        )
        for item in items
    ]


def _is_low_signal(text: str) -> bool:
    clean = " ".join(text.lower().split())
    if len(clean) < 24:
        return True
    return clean in {"да", "нет", "угу", "окей", "хорошо", "спасибо"}


def _chunks(
    records: list[TranscriptRecord], *, size: int
) -> list[list[TranscriptRecord]]:
    return [records[index : index + size] for index in range(0, len(records), size)]


def _shorten(text: str, *, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _coverage_counts(entries: list[CoverageEntry]) -> dict[str, int]:
    return {
        status: sum(1 for entry in entries if entry.status == status)
        for status in sorted(COVERAGE_STATUSES)
    }


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _citation_text(report: MeetingReport, citations: list[str]) -> str:
    index = {record.segment_id: record for record in report.transcript}
    labels = []
    for segment_id in citations:
        record = index.get(segment_id)
        if record is None:
            labels.append(segment_id)
        else:
            labels.append(
                f"{segment_id} {fmt_time(record.start)}-{fmt_time(record.end)}"
            )
    return ", ".join(labels)


def html_escape(value: str) -> str:
    return html.escape(str(value), quote=True)


def _html_stat_grid(report: MeetingReport) -> str:
    stats = [
        ("Длительность", fmt_time(report.duration)),
        ("Сегментов", str(report.segment_count)),
        ("Решений", str(len(report.decisions))),
        ("Задач", str(len(report.action_items))),
    ]
    cells = "\n".join(
        f"<div><span>{html_escape(label)}</span><strong>{html_escape(value)}</strong></div>"
        for label, value in stats
    )
    return f'<section class="stats">{cells}</section>'


def _html_adaptive_sections(report: MeetingReport) -> str:
    rendered_sections: list[str] = []
    for section in report.adaptive_sections:
        items = section.items
        if items:
            body = "\n".join(_html_report_item(report, item) for item in items)
        else:
            body = '<div class="item"><b>Нет пунктов</b><p>Нет явно зафиксированных пунктов.</p></div>'
        rendered_sections.append(
            "\n".join(
                [
                    '<section class="report-section">',
                    f"<h2>{html_escape(section.title)}</h2>",
                    f'<p class="section-purpose">{html_escape(section.purpose)}</p>',
                    body,
                    "</section>",
                ]
            )
        )
    return "\n".join(rendered_sections)


def _html_report_item(report: MeetingReport, item: ReportItem) -> str:
    return "\n".join(
        [
            '<div class="item">',
            f"<b>{html_escape(item.title)}</b>",
            f"<p>{html_escape(item.text)}</p>",
            f'<span class="cite">{html_escape(_citation_text(report, item.citations))}</span>',
            "</div>",
        ]
    )


def _html_coverage(report: MeetingReport) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html_escape(entry.segment_id)}</td>"
        f"<td>{html_escape(entry.status)}</td>"
        f"<td>{html_escape(', '.join(entry.section_titles) if entry.section_titles else 'full transcript')}</td>"
        "</tr>"
        for entry in report.coverage
    )
    return "\n".join(
        [
            '<section class="coverage">',
            "<h2>Coverage audit</h2>",
            "<table>",
            "<thead><tr><th>Segment</th><th>Status</th><th>Used in</th></tr></thead>",
            f"<tbody>{rows}</tbody>",
            "</table>",
            "</section>",
        ]
    )


def _append_profile_markdown(lines: list[str], report: MeetingReport) -> None:
    lines.extend(
        [
            "## Структура отчета",
            "",
            f"- Тип записи: **{report.profile.label}** (`{report.profile.kind}`)",
            f"- Уверенность: `{report.profile.confidence}`",
            f"- Почему так: {report.profile.rationale}",
            "",
        ]
    )


def _append_adaptive_sections_markdown(lines: list[str], report: MeetingReport) -> None:
    for section in report.adaptive_sections:
        lines.extend([f"## {section.title}", "", section.purpose, ""])
        if section.summary:
            lines.extend([section.summary, ""])
        if not section.items:
            lines.extend(["Нет явно зафиксированных пунктов.", ""])
            continue
        for item in section.items:
            lines.append(
                f"- **{item.title}:** {item.text} ({_citation_text(report, item.citations)})"
            )
        lines.append("")


def _append_coverage_markdown(lines: list[str], report: MeetingReport) -> None:
    lines.extend(["## Coverage audit", ""])
    counts = _coverage_counts(report.coverage)
    lines.append(
        f"Covered: `{counts['covered']}` · Supporting: `{counts['supporting']}` · "
        f"Low signal: `{counts['low_signal']}`"
    )
    lines.append("")
    lines.append("| Segment | Status | Used in | Note |")
    lines.append("| --- | --- | --- | --- |")
    for entry in report.coverage:
        used_in = (
            ", ".join(entry.section_titles)
            if entry.section_titles
            else "full transcript"
        )
        lines.append(
            f"| {_markdown_cell(entry.segment_id)} | {_markdown_cell(entry.status)} | "
            f"{_markdown_cell(used_in)} | {_markdown_cell(entry.rationale)} |"
        )
    lines.append("")


def _append_timeline_markdown(lines: list[str], report: MeetingReport) -> None:
    lines.extend(["## Таймлайн", ""])
    if not report.timeline:
        lines.extend(["Нет содержательных сегментов для таймлайна.", ""])
        return
    for block in report.timeline:
        lines.append(
            f"- **{block.title}:** {block.summary} ({_citation_text(report, block.citations)})"
        )
    lines.append("")


def _append_items_markdown(
    lines: list[str],
    title: str,
    items: list[ReportItem],
    report: MeetingReport,
) -> None:
    lines.extend([f"## {title}", ""])
    if not items:
        lines.extend(["Нет явно зафиксированных пунктов.", ""])
        return
    for item in items:
        lines.append(
            f"- **{item.title}:** {item.text} ({_citation_text(report, item.citations)})"
        )
    lines.append("")


def _append_actions_markdown(lines: list[str], report: MeetingReport) -> None:
    lines.extend(["## Задачи", ""])
    if not report.action_items:
        lines.extend(["Нет явно зафиксированных задач.", ""])
        return
    for item in report.action_items:
        due = f"; срок: {item.due}" if item.due else ""
        lines.append(
            f"- **{item.title}:** {item.text} "
            f"(ответственный: {item.owner or 'не указан'}{due}; {_citation_text(report, item.citations)})"
        )
    lines.append("")


def _append_adaptive_sections_typst(lines: list[str], report: MeetingReport) -> None:
    for section in report.adaptive_sections:
        lines.extend([f"== {typst_markup(section.title)}", ""])
        lines.extend(
            [
                f"_{typst_markup(section.purpose)}_",
                "",
            ]
        )
        if section.summary:
            lines.extend([typst_markup(section.summary), ""])
        if section.items:
            for item in section.items:
                lines.extend(
                    [
                        f"*{typst_markup(item.title)}.* {typst_markup(item.text)}",
                        f"#text(size: 7pt, fill: luma(45))[{typst_markup(_citation_text(report, item.citations))}]",
                        "",
                    ]
                )
        else:
            lines.extend(["Нет явно зафиксированных пунктов.", ""])


def _append_timeline_typst(lines: list[str], report: MeetingReport) -> None:
    lines.extend(["== Таймлайн", ""])
    if not report.timeline:
        lines.extend(["Нет содержательных сегментов для таймлайна.", ""])
        return
    for block in report.timeline:
        lines.extend(
            [
                f"*{typst_markup(block.title)}.* {typst_markup(block.summary)}",
                f"#text(size: 7pt, fill: luma(45))[{typst_markup(_citation_text(report, block.citations))}]",
                "",
            ]
        )


def _append_items_typst(
    lines: list[str],
    title: str,
    items: list[ReportItem],
    report: MeetingReport,
    *,
    accent: str,
) -> None:
    lines.extend([f"== {typst_markup(title)}", ""])
    if not items:
        lines.extend(["Нет явно зафиксированных пунктов.", ""])
        return
    for item in items:
        lines.extend(
            [
                f"*{typst_markup(item.title)}.* {typst_markup(item.text)}",
                f"#text(size: 7pt, fill: luma(45))[{typst_markup(_citation_text(report, item.citations))}]",
                "",
            ]
        )


def _append_actions_typst(lines: list[str], report: MeetingReport) -> None:
    lines.extend(["== Задачи", ""])
    if not report.action_items:
        lines.extend(["Нет явно зафиксированных задач.", ""])
        return
    for item in report.action_items:
        due = f"; срок: {item.due}" if item.due else ""
        lines.extend(
            [
                f"*{typst_markup(item.title)}.* {typst_markup(item.text)}",
                f"#text(size: 7pt, fill: luma(45))[{typst_markup((item.owner or 'не указан') + due)} · {typst_markup(_citation_text(report, item.citations))}]",
                "",
            ]
        )


def _append_coverage_typst(lines: list[str], report: MeetingReport) -> None:
    lines.extend(["= Coverage audit", ""])
    counts = _coverage_counts(report.coverage)
    lines.extend(
        [
            "#table(",
            "  columns: (1fr, 1fr, 1fr),",
            "  [Covered], [Supporting], [Low signal],",
            f"  [{typst_markup(str(counts['covered']))}],",
            f"  [{typst_markup(str(counts['supporting']))}],",
            f"  [{typst_markup(str(counts['low_signal']))}],",
            ")",
            "",
            "#table(",
            "  columns: (auto, 24mm, 1fr),",
            "  table.header(",
            '    [#text(weight: "bold")[Segment]],',
            '    [#text(weight: "bold")[Status]],',
            '    [#text(weight: "bold")[Used in]],',
            "  ),",
        ]
    )
    for entry in report.coverage:
        used_in = (
            ", ".join(entry.section_titles)
            if entry.section_titles
            else "full transcript"
        )
        lines.extend(
            [
                f"  [{typst_markup(entry.segment_id)}],",
                f"  [{typst_markup(entry.status)}],",
                f"  [{typst_markup(used_in)}],",
            ]
        )
    lines.extend([")", ""])


def _append_pdf_source_note_typst(lines: list[str], report: MeetingReport) -> None:
    lines.extend(
        [
            "= Артефакты",
            "",
            "Полный транскрипт хранится в Markdown-артефакте. PDF содержит отчётный слой: выводы, решения, задачи, риски, вопросы и проверку покрытия сегментов.",
            "",
            f"Источник аудио: {typst_markup(report.source_filename)}",
            "",
            "Citations вида S0001 указывают на сегменты полного транскрипта.",
            "",
        ]
    )


def _openai_client(api_key: str, *, base_url: str | None = None):
    """Клиент openai SDK со стримингом и встроенными ретраями/backoff."""
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=AI_HTTP_MAX_RETRIES,
        timeout=AI_HTTP_TIMEOUT,
    )


def _provider_error_body(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if body is not None:
        try:
            return json.dumps(body, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(body)
    return str(exc)


def _raise_for_provider_status(exc: Exception, *, provider: str) -> None:
    """APIStatusError -> ReportQuotaError (нет денег) или ReportGenerationError."""
    status = getattr(exc, "status_code", None)
    body = _provider_error_body(exc)
    message = f"{provider} API returned HTTP {status}: {body}"
    if status in (402, 429) and _is_quota_error(body):
        raise ReportQuotaError(message) from exc
    raise ReportGenerationError(message) from exc


def _post_openai_response(*, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    import openai

    client = _openai_client(api_key)
    kwargs: dict[str, Any] = {
        "model": payload["model"],
        "input": payload.get("input", []),
    }
    if payload.get("reasoning") is not None:
        kwargs["reasoning"] = payload["reasoning"]
    if payload.get("text") is not None:
        kwargs["text"] = payload["text"]
    try:
        # Стриминг держит соединение живым на длинных reasoning-вызовах.
        with client.responses.stream(**kwargs) as stream:
            for _event in stream:
                pass
            final = stream.get_final_response()
    except openai.APIStatusError as exc:
        _raise_for_provider_status(exc, provider="OpenAI")
    except openai.OpenAIError as exc:
        raise ReportGenerationError(f"OpenAI API request failed: {exc}") from exc
    return {"output_text": final.output_text or ""}


def _stream_openrouter_content(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any],
    structured_outputs: bool,
) -> str:
    extra_body = {"structured_outputs": True} if structured_outputs else None
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0,
        max_tokens=openrouter_max_tokens(),
        stream=True,
        extra_headers={
            "HTTP-Referer": "https://local.gigaam.transcriber",
            "X-Title": "GigaAM transcriber report fallback",
        },
        extra_body=extra_body,
    )
    parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        piece = getattr(delta, "content", None) if delta else None
        if piece:
            parts.append(piece)
    return "".join(parts).strip()


def _post_openrouter_response(
    *,
    payload: dict[str, Any],
    api_key: str,
    model: str,
) -> dict[str, Any]:
    import openai

    text_format = payload.get("text", {}).get("format", {})
    client = _openai_client(api_key, base_url=OPENROUTER_BASE_URL)
    last_grammar_body = ""
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
                _openrouter_json_object_system_message(text_format.get("schema")),
                *messages,
            ]
        try:
            content = _stream_openrouter_content(
                client,
                model=model,
                messages=messages,
                response_format=response_format,
                structured_outputs=structured_outputs,
            )
        except openai.APIStatusError as exc:
            body = _provider_error_body(exc)
            if structured_outputs and _is_openrouter_grammar_error(body):
                # Модель не вытягивает строгую json_schema — повторяем как json_object.
                last_grammar_body = body
                continue
            _raise_for_provider_status(exc, provider="OpenRouter")
        except openai.OpenAIError as exc:
            raise ReportGenerationError(
                f"OpenRouter API request failed: {exc}"
            ) from exc
        if not content:
            raise ReportGenerationError(
                "OpenRouter response did not contain message content"
            )
        return {
            "output_text": content,
            "_provider_model": f"openrouter:{model}",
        }
    raise ReportGenerationError(
        f"OpenRouter API returned invalid response: {last_grammar_body}"
    )


def _openrouter_model_from_report_model(model: Any) -> str:
    if not isinstance(model, str):
        return ""
    stripped = model.strip()
    if not stripped.lower().startswith(OPENROUTER_MODEL_PREFIX):
        return ""
    return stripped[len(OPENROUTER_MODEL_PREFIX) :].strip()


def _post_openai_response_with_context(
    *,
    payload: dict[str, Any],
    api_key: str,
    context: str,
    allow_openrouter_fallback: bool = True,
) -> dict[str, Any]:
    openrouter_model = _openrouter_model_from_report_model(payload.get("model"))
    if openrouter_model:
        openrouter_key = load_openrouter_api_key() or api_key
        if not openrouter_key:
            raise ReportGenerationError(
                f"{context} failed: OPENROUTER_API_KEY is not configured"
            )
        last_error: ReportGenerationError | None = None
        attempts_used = 0
        for attempt in range(1, AI_REQUEST_ATTEMPTS + 1):
            attempts_used = attempt
            try:
                return _post_openrouter_response(
                    payload=payload,
                    api_key=openrouter_key,
                    model=openrouter_model,
                )
            except ReportGenerationError as exc:
                last_error = exc
                if attempt >= AI_REQUEST_ATTEMPTS or not _should_retry_chunked(exc):
                    break
        detail = str(last_error) if last_error else "unknown error"
        raise _provider_error(
            f"{context} failed after {attempts_used} OpenRouter attempt(s): {detail}",
            last_error,
        ) from last_error

    last_error: ReportGenerationError | None = None
    attempts_used = 0
    if api_key:
        for attempt in range(1, AI_REQUEST_ATTEMPTS + 1):
            attempts_used = attempt
            try:
                return _post_openai_response(payload=payload, api_key=api_key)
            except ReportGenerationError as exc:
                last_error = exc
                if attempt >= AI_REQUEST_ATTEMPTS or not _should_retry_chunked(exc):
                    break
    else:
        last_error = ReportGenerationError("OPENAI_API_KEY is not configured")
    detail = str(last_error) if last_error else "unknown error"
    openai_error = _provider_error(
        f"{context} failed after {attempts_used} OpenAI attempt(s): {detail}",
        last_error,
    )
    if not allow_openrouter_fallback:
        raise openai_error from last_error

    fallback_key = load_openrouter_api_key()
    fallback_model = openrouter_fallback_model()
    if not fallback_key or not fallback_model:
        raise openai_error from last_error

    fallback_last_error: ReportGenerationError | None = None
    fallback_attempts_used = 0
    for attempt in range(1, AI_REQUEST_ATTEMPTS + 1):
        fallback_attempts_used = attempt
        try:
            return _post_openrouter_response(
                payload=payload,
                api_key=fallback_key,
                model=fallback_model,
            )
        except ReportGenerationError as exc:
            fallback_last_error = exc
            if attempt >= AI_REQUEST_ATTEMPTS or not _should_retry_chunked(exc):
                break
    fallback_detail = (
        str(fallback_last_error) if fallback_last_error else "unknown error"
    )
    raise _provider_error(
        f"{context} failed after {attempts_used} OpenAI attempt(s) and "
        f"{fallback_attempts_used} OpenRouter fallback attempt(s): "
        f"OpenAI: {detail}; OpenRouter: {fallback_detail}",
        last_error,
        fallback_last_error,
    ) from fallback_last_error


def _extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(
                content.get("text"), str
            ):
                chunks.append(content["text"])
    return "".join(chunks).strip()


def _merge_ai_payload(
    base_report: MeetingReport,
    payload: dict[str, Any],
    *,
    report_model: str,
    recover_base_items: bool = False,
) -> MeetingReport:
    profile_payload = payload.get("profile") or {}
    profile = ReportProfile(
        kind=str(profile_payload.get("kind") or base_report.profile.kind),
        label=str(profile_payload.get("label") or base_report.profile.label),
        confidence=_profile_confidence(
            profile_payload.get("confidence"), base_report.profile.confidence
        ),
        rationale=str(
            profile_payload.get("rationale") or base_report.profile.rationale
        ),
    )
    adaptive_sections = _adaptive_sections_from_payload(
        payload.get("adaptive_sections"),
        fallback=base_report.adaptive_sections,
    )
    coverage = build_coverage(
        transcript=base_report.transcript,
        adaptive_sections=adaptive_sections,
    )
    decisions = _merge_current_report_items(
        _report_items(payload.get("decisions", [])),
        base_report.decisions,
        recover_base=recover_base_items,
    )
    action_items = _merge_current_action_items(
        _action_items_from_payload(payload.get("action_items", [])),
        base_report.action_items,
        recover_base=recover_base_items,
    )
    open_questions = _merge_current_report_items(
        _report_items(payload.get("open_questions", [])),
        base_report.open_questions,
        recover_base=recover_base_items,
    )
    risks = _merge_current_report_items(
        _report_items(payload.get("risks", [])),
        base_report.risks,
        recover_base=recover_base_items,
    )
    return MeetingReport(
        meeting_id=base_report.meeting_id,
        title=base_report.title,
        source_filename=base_report.source_filename,
        model_name=base_report.model_name,
        generated_by=report_model,
        segment_count=base_report.segment_count,
        duration=base_report.duration,
        overview=str(payload.get("overview") or base_report.overview),
        timeline=_timeline_blocks(payload.get("timeline")),
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        risks=risks,
        notable_quotes=_dedupe_report_items(
            _report_items(payload.get("notable_quotes", []))
        ),
        transcript=base_report.transcript,
        profile=profile,
        adaptive_sections=adaptive_sections,
        coverage=coverage,
    )


def _merge_ai_synthesis_payload(
    base_report: MeetingReport,
    payload: dict[str, Any],
    *,
    report_model: str,
    recover_base_items: bool = False,
) -> MeetingReport:
    profile_payload = payload.get("profile") or {}
    profile = ReportProfile(
        kind=str(profile_payload.get("kind") or base_report.profile.kind),
        label=str(profile_payload.get("label") or base_report.profile.label),
        confidence=_profile_confidence(
            profile_payload.get("confidence"), base_report.profile.confidence
        ),
        rationale=str(
            profile_payload.get("rationale") or base_report.profile.rationale
        ),
    )
    adaptive_sections = _adaptive_sections_from_payload(
        payload.get("adaptive_sections"),
        fallback=base_report.adaptive_sections,
    )
    coverage = build_coverage(
        transcript=base_report.transcript,
        adaptive_sections=adaptive_sections,
    )
    decisions = _merge_current_report_items(
        _report_items(payload.get("decisions", [])),
        base_report.decisions,
        recover_base=recover_base_items,
    )
    action_items = _merge_current_action_items(
        _action_items_from_payload(payload.get("action_items", [])),
        base_report.action_items,
        recover_base=recover_base_items,
    )
    open_questions = _merge_current_report_items(
        _report_items(payload.get("open_questions", [])),
        base_report.open_questions,
        recover_base=recover_base_items,
    )
    risks = _merge_current_report_items(
        _report_items(payload.get("risks", [])),
        base_report.risks,
        recover_base=recover_base_items,
    )
    return MeetingReport(
        meeting_id=base_report.meeting_id,
        title=base_report.title,
        source_filename=base_report.source_filename,
        model_name=base_report.model_name,
        generated_by=report_model,
        segment_count=base_report.segment_count,
        duration=base_report.duration,
        overview=str(payload.get("overview") or base_report.overview),
        timeline=_timeline_blocks(payload.get("timeline")),
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        risks=risks,
        notable_quotes=_dedupe_report_items(
            _report_items(payload.get("notable_quotes", []))
        ),
        transcript=base_report.transcript,
        profile=profile,
        adaptive_sections=adaptive_sections,
        coverage=coverage,
    )


def _merge_ai_chunk_notes_payload(
    base_report: MeetingReport,
    chunk_notes: list[dict[str, Any]],
    *,
    report_model: str,
    warning: str,
    provider_model: str | None = None,
) -> MeetingReport:
    key_points = _dedupe_report_items(
        _chunk_report_items(chunk_notes, "key_points", title_prefix="Тема")
    )
    decisions = _dedupe_report_items(
        _chunk_report_items(chunk_notes, "decisions", title_prefix="Решение")
    )
    action_items = _dedupe_action_items(_chunk_action_items(chunk_notes))
    open_questions = _dedupe_report_items(
        _chunk_report_items(chunk_notes, "open_questions", title_prefix="Вопрос")
    )
    risks = _dedupe_report_items(
        _chunk_report_items(chunk_notes, "risks", title_prefix="Риск")
    )
    notable_quotes = _dedupe_report_items(
        _chunk_report_items(chunk_notes, "notable_quotes", title_prefix="Цитата")
    )[:8]
    adaptive_sections = [
        AdaptiveSection(
            "key_points",
            "Ключевые темы",
            "Сохранить содержательные детали по всем блокам записи.",
            _shorten(
                " ".join(str(note.get("summary", "")) for note in chunk_notes),
                limit=1200,
            ),
            key_points,
            _unique_citations(key_points),
            "blue",
        ),
        AdaptiveSection(
            "decisions",
            "Решения",
            "Отделить зафиксированные решения от обсуждения.",
            "Решения и договорённости, извлечённые из блоков транскрипта.",
            decisions,
            _unique_citations(decisions),
            "green",
        ),
        AdaptiveSection(
            "actions",
            "Задачи и next steps",
            "Зафиксировать последующие действия.",
            "Задачи, владельцы и дальнейшие шаги.",
            _action_items_as_report_items(action_items),
            _unique_action_citations(action_items),
            "amber",
        ),
        AdaptiveSection(
            "risks",
            "Риски и ограничения",
            "Не потерять блокеры, допущения и слабые места.",
            "Риски и ограничения, выделенные из блоков.",
            risks,
            _unique_citations(risks),
            "red",
        ),
        AdaptiveSection(
            "questions",
            "Открытые вопросы",
            "Показать незакрытые вопросы и проверочные пункты.",
            "Вопросы, которые требуют ответа или проверки.",
            open_questions,
            _unique_citations(open_questions),
            "violet",
        ),
    ]
    coverage = build_coverage(
        transcript=base_report.transcript,
        adaptive_sections=adaptive_sections,
    )
    return MeetingReport(
        meeting_id=base_report.meeting_id,
        title=base_report.title,
        source_filename=base_report.source_filename,
        model_name=base_report.model_name,
        generated_by=f"{provider_model or report_model}/chunked",
        segment_count=base_report.segment_count,
        duration=base_report.duration,
        overview=_shorten(
            " ".join(str(note.get("summary", "")) for note in chunk_notes), limit=1200
        ),
        timeline=_chunk_timeline(base_report, chunk_notes),
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        risks=risks,
        notable_quotes=notable_quotes,
        transcript=base_report.transcript,
        profile=base_report.profile,
        adaptive_sections=adaptive_sections,
        coverage=coverage,
        warnings=[*base_report.warnings, warning],
    )


def _flatten_payload_list(
    payloads: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for payload in payloads:
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def _payload_sections_from_top_level(
    *,
    decisions: list[dict[str, Any]],
    action_items: list[dict[str, Any]],
    open_questions: list[dict[str, Any]],
    risks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _payload_section(
            kind="decisions",
            title="Решения",
            purpose="Отделить зафиксированные решения от обсуждения.",
            summary="Решения и договорённости из синтезированных блоков.",
            items=decisions,
            accent="green",
        ),
        _payload_section(
            kind="actions",
            title="Задачи и next steps",
            purpose="Зафиксировать последующие действия.",
            summary="Задачи, владельцы и дальнейшие шаги.",
            items=[
                {
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "citations": item.get("citations", []),
                }
                for item in action_items
            ],
            accent="amber",
        ),
        _payload_section(
            kind="risks",
            title="Риски и ограничения",
            purpose="Не потерять блокеры, допущения и слабые места.",
            summary="Риски и ограничения из синтезированных блоков.",
            items=risks,
            accent="red",
        ),
        _payload_section(
            kind="questions",
            title="Открытые вопросы",
            purpose="Показать незакрытые вопросы и проверочные пункты.",
            summary="Вопросы, которые требуют ответа или проверки.",
            items=open_questions,
            accent="violet",
        ),
    ]


def _payload_section(
    *,
    kind: str,
    title: str,
    purpose: str,
    summary: str,
    items: list[dict[str, Any]],
    accent: str,
) -> dict[str, Any]:
    citations: list[str] = []
    for item in items:
        citations.extend(str(value) for value in item.get("citations", []) if value)
    return {
        "kind": kind,
        "title": title,
        "purpose": purpose,
        "summary": summary,
        "items": items,
        "citations": _unique_strings(citations),
        "accent": accent,
    }


def _dedupe_payload_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for section in sections:
        items = _dedupe_payload_items(
            [item for item in section.get("items", []) if isinstance(item, dict)]
        )
        normalized = _dedupe_key(
            str(section.get("kind") or ""), str(section.get("title") or "")
        )
        existing = next(
            (
                candidate
                for candidate in merged
                if _dedupe_key(
                    str(candidate.get("kind") or ""), str(candidate.get("title") or "")
                )
                == normalized
            ),
            None,
        )
        if existing is None:
            copy = dict(section)
            copy["items"] = items
            copy["citations"] = _unique_strings(
                [str(value) for value in copy.get("citations", []) if value]
                + [citation for item in items for citation in item.get("citations", [])]
            )
            merged.append(copy)
            continue
        existing["items"] = _dedupe_payload_items(
            [
                item
                for item in [*existing.get("items", []), *items]
                if isinstance(item, dict)
            ]
        )
        existing["citations"] = _unique_strings(
            [str(value) for value in existing.get("citations", []) if value]
            + [str(value) for value in section.get("citations", []) if value]
            + [
                citation
                for item in existing["items"]
                for citation in item.get("citations", [])
            ]
        )
    return merged


def _dedupe_payload_timeline(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in items:
        citations = _unique_strings(
            str(value) for value in item.get("citations", []) if value
        )
        if not citations:
            continue
        key = _dedupe_key(str(item.get("title") or ""), str(item.get("summary") or ""))
        if any(
            _dedupe_key(
                str(candidate.get("title") or ""), str(candidate.get("summary") or "")
            )
            == key
            for candidate in merged
        ):
            continue
        copy = dict(item)
        copy["citations"] = citations
        merged.append(copy)
    return merged


def _dedupe_payload_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in items:
        citations = _unique_strings(
            str(value) for value in item.get("citations", []) if value
        )
        if not citations:
            continue
        existing = _find_duplicate_payload_item(merged, item)
        if existing is None:
            copy = dict(item)
            copy["title"] = str(copy.get("title") or "")
            copy["text"] = str(copy.get("text") or "")
            copy["citations"] = citations
            merged.append(copy)
            continue
        existing["citations"] = _unique_strings(
            [*existing.get("citations", []), *citations]
        )
        if len(str(item.get("text") or "")) > len(str(existing.get("text") or "")):
            existing["text"] = str(item.get("text") or "")
        if not existing.get("title") and item.get("title"):
            existing["title"] = str(item.get("title") or "")
    return merged


def _dedupe_payload_actions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in items:
        citations = _unique_strings(
            str(value) for value in item.get("citations", []) if value
        )
        if not citations:
            continue
        existing = _find_duplicate_payload_item(merged, item)
        if existing is None:
            copy = dict(item)
            copy["title"] = str(copy.get("title") or "")
            copy["text"] = str(copy.get("text") or "")
            copy["owner"] = str(copy.get("owner") or "")
            copy["due"] = str(copy.get("due") or "")
            copy["citations"] = citations
            merged.append(copy)
            continue
        existing["citations"] = _unique_strings(
            [*existing.get("citations", []), *citations]
        )
        if not existing.get("owner") and item.get("owner"):
            existing["owner"] = str(item.get("owner") or "")
        if not existing.get("due") and item.get("due"):
            existing["due"] = str(item.get("due") or "")
    return merged


def _dedupe_report_items(items: list[ReportItem]) -> list[ReportItem]:
    merged: list[ReportItem] = []
    for item in items:
        existing = _find_duplicate_report_item(merged, item.title, item.text)
        if existing is None:
            item.citations = _unique_strings(item.citations)
            merged.append(item)
            continue
        existing.citations = _unique_strings([*existing.citations, *item.citations])
        if len(item.text) > len(existing.text):
            existing.text = item.text
        if not existing.title and item.title:
            existing.title = item.title
    return merged


def _merge_current_report_items(
    ai_items: list[ReportItem],
    base_items: list[ReportItem],
    *,
    recover_base: bool,
) -> list[ReportItem]:
    # Без prior_context уважаем выбор AI: локальная эвристика не должна добавлять
    # шум (ложные срабатывания) или дубли к чистому ответу модели.
    if not recover_base:
        return _dedupe_report_items(ai_items)
    ai_items = [item for item in ai_items if _current_report_item_allowed(item)]
    # С prior_context страхуемся от стирания текущих пунктов: возвращаем только те
    # локальные пункты, чьи сегменты AI ещё не процитировал в этом же разделе. При
    # этом recover тащит дословный record.text, поэтому сырые реплики отсекаем строже,
    # чем пункты модели (_is_clean_recovered_statement).
    extra = [
        item
        for item in base_items
        if _current_report_item_allowed(item)
        and _is_clean_recovered_statement(item.text)
        and not _citations_covered(item.citations, ai_items)
    ]
    return _dedupe_report_items([*ai_items, *extra])


def _merge_current_action_items(
    ai_items: list[ActionItem],
    base_items: list[ActionItem],
    *,
    recover_base: bool,
) -> list[ActionItem]:
    if not recover_base:
        return _dedupe_action_items(ai_items)
    ai_items = [item for item in ai_items if _current_action_item_allowed(item)]
    extra = [
        item
        for item in base_items
        if _current_action_item_allowed(item)
        and _is_clean_recovered_statement(item.text)
        and not _citations_covered(item.citations, ai_items)
    ]
    return _dedupe_action_items([*ai_items, *extra])


def _is_clean_recovered_statement(text: str) -> bool:
    """Похоже ли это на чистое самостоятельное утверждение для recover в формальный
    раздел.

    Базовый recover тащит дословный ``record.text``. Возвращаем обратно только то, что
    не начинается с разговорного зачина и не содержит филлеров устной речи; иначе
    сырые реплики («Ну, договорились...», «Хорошо. Тогда что решим?», «Коллеги, ...»)
    утекают в Решения/Вопросы/Риски рядом с чистыми пунктами модели. На пункты самой
    модели не влияет: применяется только к base-items при recover_base.
    """
    value = " ".join(str(text).split())
    if not value:
        return False
    if RECOVERED_OPENER_RE.match(value):
        return False
    if RAW_TRANSCRIPT_FORMAL_SIGNAL_RE.search(value):
        return False
    return True


def _current_report_item_allowed(item: ReportItem) -> bool:
    text = " ".join([item.title, item.text]).strip()
    if _looks_like_raw_transcript_formal_item(text):
        return False
    if _looks_like_methodology_example_formal_item(text):
        return False
    return True


def _current_action_item_allowed(item: ActionItem) -> bool:
    text = " ".join([item.title, item.text]).strip()
    if _looks_like_raw_transcript_formal_item(text):
        return False
    if _looks_like_methodology_example_formal_item(text):
        return False
    if _looks_like_methodology_action_noise(text):
        return False
    return True


def _looks_like_raw_transcript_formal_item(text: str) -> bool:
    value = " ".join(str(text).split()).lower()
    # Высокоточные сигналы сырой речи — независимо от длины:
    if NON_DECISION_FRAGMENT_RE.search(value):  # "не решили", "тогда ты говоришь"
        return True
    if REPEATED_WORD_RE.search(value):  # "вадим, вадим", "да, да"
        return True
    if len(value) < 40:
        return False
    words = re.findall(r"[a-zа-яё]+", value)
    if (
        words
        and sum(1 for w in words if w in RAW_TRANSCRIPT_FILLER_TOKENS) / len(words)
        >= 0.35
    ):
        return True
    signals = RAW_TRANSCRIPT_FORMAL_SIGNAL_RE.findall(value)
    if re.search(r"\bо['’]?[кк]ей\b", value) and len(signals) >= 2:
        return True
    if value.startswith(("просто ", "ну ", "вот ")) and len(signals) >= 2:
        return True
    return len(signals) >= 4


def _looks_like_methodology_example_formal_item(text: str) -> bool:
    return bool(METHODOLOGY_EXAMPLE_FORMAL_RE.search(str(text)))


def _citations_covered(citations: list[str], ai_items: list[Any]) -> bool:
    """True, если все сегменты базового пункта уже процитированы пунктами AI."""
    base_segments = {citation for citation in citations if citation}
    if not base_segments:
        return False
    covered = {citation for item in ai_items for citation in item.citations if citation}
    return base_segments <= covered


def _dedupe_adaptive_sections(sections: list[AdaptiveSection]) -> list[AdaptiveSection]:
    merged: list[AdaptiveSection] = []
    for section in sections:
        existing = next(
            (
                item
                for item in merged
                if _dedupe_key(item.kind, item.title)
                == _dedupe_key(section.kind, section.title)
            ),
            None,
        )
        if existing is None:
            merged.append(section)
            continue
        existing.items = _dedupe_report_items([*existing.items, *section.items])
        existing.citations = _unique_strings([*existing.citations, *section.citations])
        if len(section.summary) > len(existing.summary):
            existing.summary = section.summary
        if not existing.purpose and section.purpose:
            existing.purpose = section.purpose
    return merged


def _dedupe_action_items(items: list[ActionItem]) -> list[ActionItem]:
    merged: list[ActionItem] = []
    for item in items:
        existing = _find_duplicate_report_item(merged, item.title, item.text)
        if existing is None:
            item.citations = _unique_strings(item.citations)
            merged.append(item)
            continue
        existing.citations = _unique_strings([*existing.citations, *item.citations])
        if not existing.owner and item.owner:
            existing.owner = item.owner
        if not existing.due and item.due:
            existing.due = item.due
    return merged


def _find_duplicate_payload_item(
    existing_items: list[dict[str, Any]],
    item: dict[str, Any],
) -> dict[str, Any] | None:
    title = str(item.get("title") or "")
    text = str(item.get("text") or "")
    for existing in existing_items:
        if _items_are_duplicates(
            str(existing.get("title") or ""),
            str(existing.get("text") or ""),
            title,
            text,
        ):
            return existing
    return None


def _find_duplicate_report_item(
    existing_items: list[ReportItem] | list[ActionItem],
    title: str,
    text: str,
) -> ReportItem | ActionItem | None:
    for existing in existing_items:
        if _items_are_duplicates(existing.title, existing.text, title, text):
            return existing
    return None


def _items_are_duplicates(
    left_title: str, left_text: str, right_title: str, right_text: str
) -> bool:
    left = _dedupe_key(left_title, left_text)
    right = _dedupe_key(right_title, right_text)
    if not left or not right:
        return False
    return left == right or difflib.SequenceMatcher(None, left, right).ratio() >= 0.88


def _dedupe_key(*values: str) -> str:
    value = " ".join(part for part in values if part).lower()
    return re.sub(r"[^0-9a-zа-яё]+", " ", value).strip()


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _chunk_report_items(
    chunk_notes: list[dict[str, Any]],
    key: str,
    *,
    title_prefix: str,
) -> list[ReportItem]:
    items: list[ReportItem] = []
    for note in chunk_notes:
        for item in note.get(key, []):
            if not isinstance(item, dict):
                continue
            citations = [
                value for value in item.get("citations", []) if isinstance(value, str)
            ]
            if not citations:
                continue
            items.append(
                ReportItem(
                    title=str(item.get("title") or f"{title_prefix} {len(items) + 1}"),
                    text=str(item.get("text") or ""),
                    citations=citations,
                )
            )
    return items


def _chunk_action_items(chunk_notes: list[dict[str, Any]]) -> list[ActionItem]:
    items: list[ActionItem] = []
    for note in chunk_notes:
        for item in note.get("action_items", []):
            if not isinstance(item, dict):
                continue
            citations = [
                value for value in item.get("citations", []) if isinstance(value, str)
            ]
            if not citations:
                continue
            items.append(
                ActionItem(
                    title=str(item.get("title") or f"Задача {len(items) + 1}"),
                    text=str(item.get("text") or ""),
                    owner=str(item.get("owner") or ""),
                    due=str(item.get("due") or ""),
                    citations=citations,
                )
            )
    return items


def _chunk_timeline(
    base_report: MeetingReport, chunk_notes: list[dict[str, Any]]
) -> list[TimelineBlock]:
    record_by_id = {record.segment_id: record for record in base_report.transcript}
    timeline: list[TimelineBlock] = []
    for index, note in enumerate(chunk_notes, start=1):
        citations: list[str] = []
        for key in [
            "key_points",
            "decisions",
            "action_items",
            "open_questions",
            "risks",
            "notable_quotes",
        ]:
            for item in note.get(key, []):
                if not isinstance(item, dict):
                    continue
                citations.extend(
                    value
                    for value in item.get("citations", [])
                    if isinstance(value, str)
                )
        citations = list(
            dict.fromkeys(
                citation for citation in citations if citation in record_by_id
            )
        )
        if citations:
            records = [record_by_id[citation] for citation in citations]
            title = f"{fmt_time(min(record.start for record in records))}-{fmt_time(max(record.end for record in records))}"
        else:
            title = f"Блок {index}"
        timeline.append(
            TimelineBlock(
                title=title,
                summary=str(note.get("summary") or ""),
                citations=citations,
            )
        )
    return timeline


def _provider_error(
    message: str, *causes: BaseException | None
) -> ReportGenerationError:
    """Сохраняет тип quota-ошибки: если хоть одна причина — нет денег, отдаём
    ReportQuotaError (чтобы сервис поставил встречу на паузу, а не в local-дамп)."""
    if any(isinstance(cause, ReportQuotaError) for cause in causes):
        return ReportQuotaError(message)
    return ReportGenerationError(message)


def _should_retry_chunked(exc: ReportGenerationError) -> bool:
    # «Нет денег» (quota) не ретраим и не пропускаем — она должна всплыть до сервиса.
    if isinstance(exc, ReportQuotaError):
        return False
    message = str(exc).lower()
    return (
        "request failed" in message
        or "remote end closed" in message
        or "timed out" in message
        or "connection reset" in message
    )


def _should_retry_direct_ai(exc: ReportGenerationError) -> bool:
    message = str(exc).lower()
    return _should_retry_chunked(exc) or "response was not valid json" in message


def _is_openrouter_grammar_error(body: str) -> bool:
    lowered = body.lower()
    return (
        "compiled grammar is too large" in lowered
        or "simplify your tool schemas" in lowered
    )


DEFAULT_OPENROUTER_JSON_OBJECT_KEYS = (
    "profile",
    "overview",
    "adaptive_sections",
    "coverage",
    "timeline",
    "decisions",
    "action_items",
    "open_questions",
    "risks",
    "notable_quotes",
)


def _openrouter_json_object_keys(schema: dict[str, Any] | None) -> list[str]:
    """Берём ключи верхнего уровня из json_schema запроса, чтобы json_object
    fallback не навязывал чужую структуру (например, схеме чанков нужны
    summary/key_points, а не profile/overview)."""
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            return list(properties.keys())
        required = schema.get("required")
        if isinstance(required, list) and required:
            return [str(key) for key in required]
    return list(DEFAULT_OPENROUTER_JSON_OBJECT_KEYS)


def _openrouter_json_object_system_message(
    schema: dict[str, Any] | None = None,
) -> dict[str, str]:
    keys = ", ".join(_openrouter_json_object_keys(schema))
    return {
        "role": "system",
        "content": (
            "Return only one valid JSON object. Do not use Markdown, code fences, tables, "
            "or prose outside JSON. The object must use these top-level keys exactly: "
            f"{keys}. Use empty arrays when a section has no items."
        ),
    }


def _filter_payload_citations(payload: Any, known_ids: set[str]) -> Any:
    if isinstance(payload, dict):
        filtered = {
            key: _filter_payload_citations(value, known_ids)
            for key, value in payload.items()
        }
        if "citations" in filtered and isinstance(filtered["citations"], list):
            filtered["citations"] = [
                citation
                for citation in filtered["citations"]
                if isinstance(citation, str) and citation in known_ids
            ]
        if "segment_id" in filtered and filtered["segment_id"] not in known_ids:
            filtered["segment_id"] = ""
        return filtered
    if isinstance(payload, list):
        return [_filter_payload_citations(item, known_ids) for item in payload]
    if isinstance(payload, str):
        return _strip_unknown_segment_refs(payload, known_ids)
    return payload


def _strip_unknown_segment_refs(value: str, known_ids: set[str]) -> str:
    return SEGMENT_REF_RE.sub(
        lambda match: match.group(0) if match.group(0) in known_ids else "", value
    )


def _profile_confidence(value: Any, fallback: float) -> float:
    # OpenRouter json_object fallback не валидирует схему: confidence может прийти
    # строкой ("high") — не роняем весь отчёт, откатываемся к базовому значению.
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _timeline_blocks(items: Any) -> list[TimelineBlock]:
    if not isinstance(items, list):
        return []
    return [
        TimelineBlock(
            title=str(item.get("title", "")),
            summary=str(item.get("summary", "")),
            citations=list(item.get("citations", [])),
        )
        for item in items
        if isinstance(item, dict)
    ]


def _report_items(items: Any) -> list[ReportItem]:
    if not isinstance(items, list):
        return []
    return [
        ReportItem(
            title=str(item.get("title", "")),
            text=str(item.get("text", "")),
            citations=list(item.get("citations", [])),
        )
        for item in items
        if isinstance(item, dict)
    ]


def _action_items_from_payload(items: Any) -> list[ActionItem]:
    if not isinstance(items, list):
        return []
    return _dedupe_action_items(
        [
            ActionItem(
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                owner=str(item.get("owner", "")),
                due=str(item.get("due", "")),
                citations=list(item.get("citations", [])),
            )
            for item in items
            if isinstance(item, dict)
        ]
    )


def _adaptive_sections_from_payload(
    items: Any,
    *,
    fallback: list[AdaptiveSection],
) -> list[AdaptiveSection]:
    if not isinstance(items, list):
        return fallback
    sections: list[AdaptiveSection] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sections.append(
            AdaptiveSection(
                kind=str(item.get("kind", "")),
                title=str(item.get("title", "")),
                purpose=str(item.get("purpose", "")),
                summary=str(item.get("summary", "")),
                items=_report_items(item.get("items", [])),
                citations=list(item.get("citations", [])),
                accent=str(item.get("accent") or "blue"),
            )
        )
    return sections or fallback


def _coverage_from_payload(
    items: Any,
    *,
    fallback: list[CoverageEntry],
) -> list[CoverageEntry]:
    if not isinstance(items, list):
        return fallback
    coverage: list[CoverageEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        coverage.append(
            CoverageEntry(
                segment_id=str(item.get("segment_id", "")),
                status=str(item.get("status", "")),
                section_titles=list(item.get("section_titles", [])),
                rationale=str(item.get("rationale", "")),
            )
        )
    return coverage or fallback
