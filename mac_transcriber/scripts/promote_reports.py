"""Финализация + промоут отчётов по дедуп-карте уникальных встреч.

Читает mapping.json (см. построение в сессии). Для каждой группы:
- regenerate=True: берёт сгенерированный суб-агентом payload (work-dir <ROOT>/<key>/ai_payload.json),
  и для КАЖДОГО dir группы строит base из его transcript.json, мёржит, валидирует и рендерит
  артефакты прямо в <dir>/artifacts.
- regenerate=False (claude уже есть): копирует report.* из claude_src в остальные dir группы.

Один payload на уникальную встречу разносится по всем её дублям (citations фильтруются под каждый dir).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mac_transcriber import reporting  # noqa: E402

MODEL = "v3_e2e_rnnt"
GENERATED_BY = "claude-opus-4-8 · mac_transcriber subagent"
ARTIFACT_FILES = [
    "report.json",
    "report.md",
    "report.html",
    "report.typ",
    "coverage.json",
    "report_health.json",
    "slack_summary.md",
    "report.pdf",
]


def _meta(meeting_dir: Path) -> dict:
    sj = meeting_dir / "status.json"
    if sj.exists():
        try:
            return json.loads(sj.read_text(encoding="utf-8")).get("metadata", {}) or {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_base(meeting_dir: Path) -> reporting.MeetingReport:
    meta = _meta(meeting_dir)
    src = str(meta.get("source_filename") or "audio.m4a")
    segments = json.loads(
        (meeting_dir / "artifacts" / "transcript.json").read_text(encoding="utf-8")
    )
    return reporting.build_report(
        meeting_id=str(meta.get("meeting_id") or meeting_dir.name),
        title=Path(src).stem or meeting_dir.name,
        source_filename=src,
        model_name=MODEL,
        segments=segments,
        use_ai=False,
    )


def _render_into(meeting_dir: Path, payload: dict, make_pdf: bool) -> dict:
    base = _build_base(meeting_dir)
    known = {r.segment_id for r in base.transcript}
    title = str(payload.get("title") or "").strip()
    filtered = reporting._filter_payload_citations(payload, known)
    report = reporting._merge_ai_payload(
        base, filtered, report_model=GENERATED_BY, recover_base_items=False
    )
    if title:
        report = dataclasses.replace(report, title=title)
    reporting.validate_report_citations(report)
    arts = reporting.write_report_artifacts_from_report(
        output_dir=meeting_dir / "artifacts",
        report=report,
        requested_ai=True,
        make_pdf=make_pdf,
    )
    return {
        "dir": meeting_dir.name,
        "title": report.title,
        "status": arts.status,
        "md_words": len(
            (meeting_dir / "artifacts" / "report.md")
            .read_text(encoding="utf-8")
            .split()
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="scratchpad reports2 dir with mapping.json + <key>/ai_payload.json",
    )
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated keys to limit")
    args = ap.parse_args()

    root = Path(args.root)
    mapping = json.loads((root / "mapping.json").read_text(encoding="utf-8"))
    only = {k for k in args.only.split(",") if k}
    make_pdf = not args.no_pdf

    out = []
    for key, v in mapping.items():
        if only and key not in only:
            continue
        all_dirs = [Path(p) for p in v["all_dirs"]]
        if v["regenerate"]:
            pf = root / key / "ai_payload.json"
            if not pf.exists():
                out.append({"key": key, "error": "no ai_payload.json"})
                continue
            payload = json.loads(pf.read_text(encoding="utf-8"))
            for d in all_dirs:
                try:
                    out.append({"key": key, **_render_into(d, payload, make_pdf)})
                except Exception as exc:  # noqa: BLE001
                    out.append({"key": key, "dir": d.name, "error": str(exc)[:160]})
        else:
            src = Path(v["claude_src"]) / "artifacts"
            for d in all_dirs:
                dst = d / "artifacts"
                if dst == src:
                    out.append(
                        {
                            "key": key,
                            "dir": d.name,
                            "status": "claude-src",
                            "md_words": len(
                                (src / "report.md").read_text(encoding="utf-8").split()
                            ),
                        }
                    )
                    continue
                copied = 0
                for f in ARTIFACT_FILES:
                    if (src / f).exists():
                        shutil.copy2(src / f, dst / f)
                        copied += 1
                out.append(
                    {"key": key, "dir": d.name, "status": f"propagated({copied})"}
                )

    print(json.dumps(out, ensure_ascii=False, indent=2))
    bad = [
        r
        for r in out
        if r.get("error")
        or (
            r.get("status") not in (None,)
            and r.get("status") not in ("ok", "degraded", "claude-src")
            and not str(r.get("status", "")).startswith("propagated")
        )
    ]
    print(f"\nTOTAL rows={len(out)} problems={len(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
