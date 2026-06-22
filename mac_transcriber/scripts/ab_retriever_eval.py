#!/usr/bin/env python3
"""A/B сравнение слоя памяти/RAG: самодел против LlamaIndex и mem0.

Каждый ретривер отдаёт один и тот же ``context_pack`` (facts/segments/
embedding_chunks), а пайплайн отчёта и судья — общие из ``ab_report_eval``. Так
изолируется ровно поиск контекста прошлых встреч.

Стадии:
  retrieve  (дёшево) — что каждый ретривер достаёт на кейс, рядом + метрики.
  report    (дорого)  — полная генерация отчёта на каждый вариант + LLM-судья.

Запуск только из изолированного venv (.local/venv-rag-ab) с PYTHONPATH=репо::

    PYTHONPATH=$(pwd) .local/venv-rag-ab/bin/python \\
        mac_transcriber/scripts/ab_retriever_eval.py retrieve --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ab_report_eval  # noqa: E402  (соседний скрипт, переиспользуем кейсы/генерацию)

from mac_transcriber import memory_db, reporting  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(".local/rag_ab_eval")
LLAMA_TABLE = (
    "abtest_meeting_chunks"  # физическая таблица станет data_abtest_meeting_chunks
)
MEM0_COLLECTION = "abtest_mem0_chunks"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = ab_report_eval.load_env_files(
        args.env_file if args.env_file is not None else [Path(".env.local")]
    )
    os.environ.update(env)
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    database_url = memory_db.database_url_from_env(env)
    if not database_url:
        raise SystemExit("MAC_TRANSCRIBER_DATABASE_URL is not configured")
    api_key = memory_db.openai_api_key_from_env(env)
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not configured")

    if args.command == "cleanup":
        return cleanup(database_url)

    retrievers = build_retrievers(database_url, api_key, names=args.retriever)
    cases = ab_report_eval.selected_cases(args.case)
    if args.limit:
        cases = cases[: args.limit]

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.command == "report":
        return run_report(retrievers, cases, output_root, database_url, api_key, args)

    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"== {case.name}: {case.title}", flush=True)
        segments = ab_report_eval.load_case_segments(database_url, case)
        query_text = memory_db.context_query_text(
            case.title, [str(row.get("text", "")) for row in segments]
        )
        packs = {
            name: retriever(case, query_text, args.top_k)
            for name, retriever in retrievers.items()
        }
        case_result = {
            "case": case.name,
            "title": case.title,
            "retrieval": {name: summarize_pack(pack) for name, pack in packs.items()},
        }
        results.append(case_result)
        write_retrieval_dump(output_root / case.name, packs)
        print_retrieval(case_result)

    ab_report_eval.write_json(
        output_root / "retrieval_summary.json", {"results": results}
    )
    write_retrieval_markdown(output_root / "retrieval_summary.md", results)
    print(f"\nsummary: {output_root / 'retrieval_summary.md'}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["retrieve", "report", "cleanup"],
        help="retrieve (дёшево) | report (отчёт+судья, платно) | cleanup",
    )
    parser.add_argument("--env-file", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--retriever",
        action="append",
        choices=["homemade", "llamaindex", "mem0"],
        help="Которые ретриверы сравнивать; по умолчанию все три.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[c.name for c in ab_report_eval.DEFAULT_CASES],
    )
    parser.add_argument("--limit", type=int, help="Только первые N кейсов.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--model", default="gpt-5-mini", help="Модель отчёта (report).")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--variant-timeout", type=int, default=300)
    parser.add_argument(
        "--judge-model",
        default="gpt-5-mini",
        help="Модель ранжирующего судьи (report).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Перегенерировать существующие варианты."
    )
    return parser.parse_args(argv)


# --- ретриверы: каждый возвращает context_pack одинаковой формы -------------


def build_retrievers(database_url: str, api_key: str, names: list[str] | None):
    wanted = names or ["homemade", "llamaindex", "mem0"]
    chunks = fetch_chunks(database_url)
    print(f"   corpus: {len(chunks)} embedding chunks", flush=True)
    registry: dict[str, Any] = {}
    if "homemade" in wanted:
        registry["homemade"] = make_homemade(database_url, api_key)
    if "llamaindex" in wanted:
        registry["llamaindex"] = make_llamaindex(database_url, chunks)
    if "mem0" in wanted:
        registry["mem0"] = make_mem0(database_url, chunks)
    return registry


def fetch_chunks(database_url: str) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    with (
        psycopg.connect(database_url) as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            """
            SELECT c.chunk_id, c.meeting_id, m.title AS meeting_title,
                   c.source_table, c.source_id, c.text, c.embedding
            FROM embedding_chunks c
            JOIN meetings m ON m.meeting_id = c.meeting_id
            WHERE c.embedding IS NOT NULL
            """
        )
        return [dict(row) for row in cur.fetchall()]


def make_homemade(database_url: str, api_key: str):
    def retrieve(case, query_text: str, top_k: int) -> dict[str, Any]:
        started = time.perf_counter()
        pack = memory_db.build_report_context_pack(
            database_url,
            meeting_id=case.meeting_id,
            title=case.title,
            source_filename=case.source_filename,
            limit=top_k,
            query_text=query_text,
        )
        pack["_elapsed"] = round(time.perf_counter() - started, 3)
        return pack

    return retrieve


def make_llamaindex(database_url: str, chunks: list[dict[str, Any]]):
    from sqlalchemy import make_url
    from llama_index.core import VectorStoreIndex
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.postgres import PGVectorStore

    url = make_url(database_url)
    embed = OpenAIEmbedding(model=EMBED_MODEL)
    # hybrid_search=True добавляет tsvector-колонку для sparse/BM25-ноги — это и есть
    # фишка OSS поверх ванильного вектора. Корпус русский → русский tsvector-конфиг.
    store = PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port or 5432,
        user=url.username,
        table_name=LLAMA_TABLE,
        embed_dim=EMBED_DIMS,
        hybrid_search=True,
        text_search_config="russian",
    )
    index = VectorStoreIndex.from_vector_store(store, embed_model=embed)
    if not _llama_already_ingested(database_url):
        nodes = [
            TextNode(
                # chunk_id не глобально уникален (диапазоны сегментов повторяются
                # в разных встречах) — иначе node_id коллизятся и дедуп LlamaIndex
                # схлопывает корпус. Префиксуем meeting_id.
                text=row["text"],
                id_=f"{row['meeting_id']}:{row['chunk_id']}",
                embedding=_as_float_list(row["embedding"]),
                metadata={
                    "meeting_id": row["meeting_id"],
                    "meeting_title": row["meeting_title"],
                    "source_table": row["source_table"],
                    "source_id": row["source_id"],
                },
            )
            for row in chunks
        ]
        index.insert_nodes(nodes)
        print(f"   llamaindex: ingested {len(nodes)} nodes", flush=True)

    def _filters(case):
        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key="meeting_id",
                    value=case.meeting_id,
                    operator=FilterOperator.NE,
                )
            ]
        )

    def retrieve(case, query_text: str, top_k: int) -> dict[str, Any]:
        started = time.perf_counter()
        k = top_k * 3
        flt = _filters(case)
        # Гибрид: dense-вектор + sparse(tsvector/BM25), слитые relative_score.
        # num_queries=1 → без LLM-расширения запроса (бесплатно). Падение sparse-ноги
        # (например, нет russian-конфига) откатывает на чистый вектор.
        try:
            fusion = QueryFusionRetriever(
                [
                    index.as_retriever(similarity_top_k=k, filters=flt),
                    index.as_retriever(
                        vector_store_query_mode="sparse",
                        similarity_top_k=k,
                        filters=flt,
                    ),
                ],
                similarity_top_k=k,
                num_queries=1,
                mode="relative_score",
                use_async=False,
            )
            hits = fusion.retrieve(query_text)
        except Exception as exc:  # noqa: BLE001
            print(
                f"   llamaindex hybrid failed ({exc}); fallback to vector", flush=True
            )
            hits = index.as_retriever(similarity_top_k=k, filters=flt).retrieve(
                query_text
            )
        retrieved = [
            {
                "meeting_id": h.node.metadata.get("meeting_id"),
                "meeting_title": h.node.metadata.get("meeting_title"),
                "source_table": h.node.metadata.get("source_table"),
                "source_id": h.node.metadata.get("source_id"),
                "text": h.node.text,
                "distance": round(1.0 - float(h.score), 4)
                if h.score is not None
                else None,
            }
            for h in hits[: top_k * 3]
        ]
        pack = to_context_pack(case, query_text, retrieved, top_k)
        pack["_elapsed"] = round(time.perf_counter() - started, 3)
        return pack

    return retrieve


def make_mem0(database_url: str, chunks: list[dict[str, Any]]):
    from sqlalchemy import make_url
    from mem0 import Memory

    url = make_url(database_url)
    config = {
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        "embedder": {
            "provider": "openai",
            "config": {"model": EMBED_MODEL, "embedding_dims": EMBED_DIMS},
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "collection_name": MEM0_COLLECTION,
                "embedding_model_dims": EMBED_DIMS,
                "dbname": url.database,
                "user": url.username,
                "password": url.password,
                "host": url.host,
                "port": url.port or 5432,
                "hnsw": True,
            },
        },
    }
    memory = Memory.from_config(config)
    if not _mem0_already_ingested(memory):
        for row in chunks:
            memory.add(
                row["text"],
                user_id="corpus",
                metadata={
                    "meeting_id": row["meeting_id"],
                    "meeting_title": row["meeting_title"],
                    "source_table": row["source_table"],
                    "source_id": row["source_id"],
                },
                infer=False,
            )
        print(f"   mem0: ingested {len(chunks)} memories (infer=False)", flush=True)

    def retrieve(case, query_text: str, top_k: int) -> dict[str, Any]:
        started = time.perf_counter()
        # mem0 2.x: user_id нельзя как top-level в search(), только через filters.
        res = memory.search(
            query=query_text, filters={"user_id": "corpus"}, limit=top_k * 4
        )
        items = res.get("results", res) if isinstance(res, dict) else res
        retrieved = []
        for item in items:
            meta = item.get("metadata") or {}
            if meta.get("meeting_id") == case.meeting_id:
                continue  # клиентский фильтр: исключаем текущую встречу
            retrieved.append(
                {
                    "meeting_id": meta.get("meeting_id"),
                    "meeting_title": meta.get("meeting_title"),
                    "source_table": meta.get("source_table"),
                    "source_id": meta.get("source_id"),
                    "text": item.get("memory") or item.get("text") or "",
                    "distance": round(1.0 - float(item["score"]), 4)
                    if item.get("score") is not None
                    else None,
                }
            )
        pack = to_context_pack(case, query_text, retrieved[: top_k * 3], top_k)
        pack["_elapsed"] = round(time.perf_counter() - started, 3)
        return pack

    return retrieve


def to_context_pack(
    case, query_text: str, retrieved: list[dict[str, Any]], top_k: int
) -> dict[str, Any]:
    """Раскладывает отобранные чанки по тем же бакетам, что у самодела."""
    facts: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for chunk in retrieved:
        if chunk["source_table"] == "meeting_facts" and len(facts) < top_k:
            facts.append(
                {
                    "meeting_id": chunk["meeting_id"],
                    "meeting_title": chunk["meeting_title"],
                    "fact_type": "fact",
                    "title": "",
                    "text": chunk["text"],
                }
            )
        elif chunk["source_table"] == "meeting_segments" and len(segments) < top_k:
            segments.append(
                {
                    "meeting_id": chunk["meeting_id"],
                    "meeting_title": chunk["meeting_title"],
                    "text": chunk["text"],
                }
            )
    return {
        "meeting_id": case.meeting_id,
        "title": case.title,
        "source_filename": case.source_filename,
        "query": case.title,
        "embedding_query": query_text[:200],
        "facts": facts,
        "segments": segments,
        "embedding_chunks": retrieved[:top_k],
    }


# --- метрики / вывод --------------------------------------------------------


def summarize_pack(pack: dict[str, Any]) -> dict[str, Any]:
    chunks = pack.get("embedding_chunks") or []
    distances = [c["distance"] for c in chunks if c.get("distance") is not None]
    meetings = {c.get("meeting_id") for c in chunks if c.get("meeting_id")}
    return {
        "facts": len(pack.get("facts") or []),
        "segments": len(pack.get("segments") or []),
        "chunks": len(chunks),
        "distinct_meetings": len(meetings),
        "min_distance": round(min(distances), 4) if distances else None,
        "mean_distance": round(sum(distances) / len(distances), 4)
        if distances
        else None,
        "elapsed": pack.get("_elapsed"),
    }


def write_retrieval_dump(case_dir: Path, packs: dict[str, dict[str, Any]]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, pack in packs.items():
        ab_report_eval.write_json(case_dir / f"{name}_pack.json", pack)


def print_retrieval(case_result: dict[str, Any]) -> None:
    for name, summary in case_result["retrieval"].items():
        print(
            f"   {name:11} chunks={summary['chunks']} "
            f"facts={summary['facts']} segs={summary['segments']} "
            f"distinct_meetings={summary['distinct_meetings']} "
            f"min_dist={summary['min_distance']} mean_dist={summary['mean_distance']} "
            f"{summary['elapsed']}s",
            flush=True,
        )


def write_retrieval_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Retriever A/B (retrieval stage)",
        "",
        "| Case | Retriever | Chunks | Facts | Segs | Distinct meetings | min dist | mean dist | s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        for name, summary in result["retrieval"].items():
            lines.append(
                f"| {result['case']} | {name} | {summary['chunks']} | "
                f"{summary['facts']} | {summary['segments']} | "
                f"{summary['distinct_meetings']} | {summary['min_distance']} | "
                f"{summary['mean_distance']} | {summary['elapsed']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- стадия report: генерация отчёта на вариант + ранжирующий судья ---------


def run_report(retrievers, cases, output_root, database_url, api_key, args) -> int:
    variant_names = ["no_context", *retrievers.keys()]
    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"== report {case.name}: {case.title}", flush=True)
        segments = ab_report_eval.load_case_segments(database_url, case)
        query_text = memory_db.context_query_text(
            case.title, [str(row.get("text", "")) for row in segments]
        )
        selected_ids = [str(row["segment_id"]) for row in segments]
        case_dir = output_root / "report" / case.name
        case_dir.mkdir(parents=True, exist_ok=True)

        packs: dict[str, dict[str, Any] | None] = {"no_context": None}
        for name, retriever in retrievers.items():
            packs[name] = retriever(case, query_text, args.top_k)
        write_retrieval_dump(case_dir, {k: v for k, v in packs.items() if v})

        variants: dict[str, dict[str, Any]] = {}
        for vname in variant_names:
            vdir = case_dir / vname
            complete = (vdir / "report.json").exists() and (
                vdir / "report_health.json"
            ).exists()
            if args.force or not complete:
                ab_report_eval.generate_variant_with_timeout(
                    variant_dir=vdir,
                    case=case,
                    segments=segments,
                    context=packs.get(vname),
                    provider=args.provider,
                    report_model=args.model,
                    api_key=api_key,
                    timeout_seconds=args.variant_timeout,
                )
            variants[vname] = ab_report_eval.analyze_variant(
                variant_dir=vdir, selected_segment_ids=selected_ids
            )
            v = variants[vname]
            print(
                f"   {vname:11} status={v.get('status')} formal={v.get('formal_total')} "
                f"sections={v.get('sections')} bytes={v.get('markdown_bytes')}",
                flush=True,
            )

        judgement = judge_retrievers(case, variants, args, api_key)
        results.append(
            {
                "case": case.name,
                "title": case.title,
                "variants": {k: _variant_brief(v) for k, v in variants.items()},
                "judge": judgement,
            }
        )
        ab_report_eval.write_json(
            output_root / "report_summary.json", {"results": results}
        )
        write_report_markdown(output_root / "report_summary.md", results)
        print(
            f"   judge best={judgement.get('best')} ranking={judgement.get('ranking')}",
            flush=True,
        )

    print(f"\nsummary: {output_root / 'report_summary.md'}", flush=True)
    return 0


def judge_retrievers(case, variants, args, api_key) -> dict[str, Any]:
    names = list(variants.keys())
    evaluation = {
        "case": {"name": case.name, "title": case.title, "notes": case.notes},
        "variants": {
            name: {
                "status": v.get("status"),
                "formal_total": v.get("formal_total"),
                "sections": v.get("sections"),
                "outline": (v.get("outline") or [])[:12],
            }
            for name, v in variants.items()
        },
    }
    payload = {
        "model": args.judge_model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You evaluate Russian meeting reports built with different "
                    "prior-context retrievers (no_context = baseline without memory). "
                    "Pick which retriever's prior context produced the most useful, "
                    "well-grounded report for the CURRENT meeting. Reward relevant prior "
                    "context; penalize invented/irrelevant context and lost current facts. "
                    "Return strict JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(evaluation, ensure_ascii=False, indent=2),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "retriever_judge",
                "strict": True,
                "schema": _judge_schema(names),
            }
        },
    }
    try:
        response = ab_report_eval.post_provider_response(
            provider=args.provider,
            model=args.judge_model,
            payload=payload,
            api_key=api_key,
        )
        parsed = json.loads(reporting._extract_output_text(response) or "{}")
        parsed["status"] = "ok"
        return parsed
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}


def _judge_schema(names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ranking": {"type": "array", "items": {"type": "string", "enum": names}},
            "best": {"type": "string", "enum": names},
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "variant": {"type": "string", "enum": names},
                        "grounding": {"type": "integer", "minimum": 1, "maximum": 5},
                        "usefulness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "prior_context_value": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": [
                        "variant",
                        "grounding",
                        "usefulness",
                        "prior_context_value",
                    ],
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["ranking", "best", "scores", "rationale"],
    }


def _variant_brief(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": variant.get("status"),
        "formal_total": variant.get("formal_total"),
        "sections": variant.get("sections"),
        "timeline": variant.get("timeline"),
        "markdown_bytes": variant.get("markdown_bytes"),
        "invalid_refs": variant.get("invalid_refs"),
        "outline": (variant.get("outline") or [])[:8],
    }


def write_report_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Retriever A/B (report + judge stage)",
        "",
        "| Case | Variant | Status | Formal | Sections | Bytes | Judge best | Ranking |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        judge = result.get("judge") or {}
        best = judge.get("best", "")
        ranking = " > ".join(judge.get("ranking") or [])
        for vname, v in result["variants"].items():
            mark = " ⭐" if vname == best else ""
            lines.append(
                f"| {result['case']} | {vname}{mark} | {v.get('status')} | "
                f"{v.get('formal_total')} | {v.get('sections')} | "
                f"{v.get('markdown_bytes')} | {best if vname == best else ''} | "
                f"{ranking if vname == best else ''} |"
            )
    lines.extend(["", "## Judge rationale", ""])
    for result in results:
        judge = result.get("judge") or {}
        lines.append(f"### {result['case']}")
        lines.append(
            f"- best: **{judge.get('best', '')}** | ranking: {' > '.join(judge.get('ranking') or [])}"
        )
        if judge.get("rationale"):
            lines.append(f"- {judge['rationale']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- утилиты ----------------------------------------------------------------


def _as_float_list(embedding: Any) -> list[float]:
    if isinstance(embedding, str):
        return [float(x) for x in embedding.strip("[]").split(",") if x.strip()]
    return [float(x) for x in embedding]


def _llama_already_ingested(database_url: str) -> bool:
    return _table_rowcount(database_url, f"data_{LLAMA_TABLE}") > 0


def _mem0_already_ingested(memory: Any) -> bool:
    try:
        existing = memory.get_all(filters={"user_id": "corpus"}, limit=1)
        items = (
            existing.get("results", existing)
            if isinstance(existing, dict)
            else existing
        )
        return bool(items)
    except Exception:
        return False


def _table_rowcount(database_url: str, table: str) -> int:
    import psycopg

    try:
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM "{table}"')
            return int(cur.fetchone()[0])
    except Exception:
        return 0


def cleanup(database_url: str) -> int:
    import psycopg

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        for table in (f"data_{LLAMA_TABLE}", MEM0_COLLECTION):
            cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            print(f"dropped {table}", flush=True)
        conn.commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
