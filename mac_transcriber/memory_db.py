import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping


OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_CHUNK_CHARS = 2000
DEFAULT_EMBEDDING_CHUNK_SEGMENTS = 8


def database_url_from_env(env: Mapping[str, str] = os.environ) -> str:
    primary = env.get("MAC_TRANSCRIBER_DATABASE_URL", "").strip()
    if primary:
        return primary
    return env.get("MAC_TRANSCRIBER_POSTGRES_DSN", "").strip()


def memory_enabled(env: Mapping[str, str] = os.environ) -> bool:
    return bool(database_url_from_env(env))


def openai_api_key_from_env(
    env: Mapping[str, str] = os.environ,
    *,
    env_path: Path | None = None,
) -> str:
    key = env.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    configured = env.get("MAC_TRANSCRIBER_OPENAI_ENV_FILE", "").strip()
    candidates = [Path(configured)] if configured else []
    if env_path is not None:
        candidates.append(env_path)
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
            if name.strip() == "OPENAI_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return ""


def build_schema_sql() -> str:
    return """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS meetings (
    id bigserial PRIMARY KEY,
    meeting_id text NOT NULL,
    title text,
    source_filename text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    result jsonb,
    status jsonb,
    manifest jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (meeting_id)
);

CREATE TABLE IF NOT EXISTS meeting_artifacts (
    id bigserial PRIMARY KEY,
    meeting_id text NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    relative_path text NOT NULL,
    kind text,
    size_bytes bigint,
    sha256 text,
    modified_at timestamptz,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (meeting_id, relative_path)
);

CREATE TABLE IF NOT EXISTS meeting_segments (
    id bigserial PRIMARY KEY,
    meeting_id text NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    segment_id text NOT NULL,
    start_seconds double precision,
    end_seconds double precision,
    speaker text,
    track text,
    text text NOT NULL DEFAULT '',
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (meeting_id, segment_id)
);

CREATE TABLE IF NOT EXISTS meeting_facts (
    id bigserial PRIMARY KEY,
    meeting_id text NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    fact_type text NOT NULL,
    title text NOT NULL DEFAULT '',
    text text NOT NULL DEFAULT '',
    owner text,
    due text,
    citations jsonb NOT NULL DEFAULT '[]'::jsonb,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (meeting_id, fact_type, title, text)
);

CREATE TABLE IF NOT EXISTS report_context_links (
    id bigserial PRIMARY KEY,
    meeting_id text NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    fact_id bigint REFERENCES meeting_facts(id) ON DELETE CASCADE,
    segment_id text,
    citation text,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (meeting_id, fact_id, citation)
);

CREATE TABLE IF NOT EXISTS embedding_chunks (
    id bigserial PRIMARY KEY,
    meeting_id text NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    chunk_id text NOT NULL,
    source_table text NOT NULL,
    source_id text,
    text text NOT NULL DEFAULT '',
    embedding vector(1536),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (meeting_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_meetings_title_trgm
    ON meetings USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_meeting_artifacts_payload
    ON meeting_artifacts USING gin (payload jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_meeting_segments_text_fts
    ON meeting_segments USING gin (to_tsvector('simple', text));
CREATE INDEX IF NOT EXISTS idx_meeting_segments_text_trgm
    ON meeting_segments USING gin (text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_meeting_facts_citations
    ON meeting_facts USING gin (citations jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_meeting_facts_payload
    ON meeting_facts USING gin (payload jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_meeting_facts_text_fts
    ON meeting_facts USING gin (to_tsvector('simple', title || ' ' || text));
CREATE INDEX IF NOT EXISTS idx_report_context_links_segment
    ON report_context_links (meeting_id, segment_id);
CREATE INDEX IF NOT EXISTS idx_embedding_chunks_metadata
    ON embedding_chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_embedding_chunks_embedding
    ON embedding_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
""".strip()


def extract_facts_from_report(report_json_path: Path) -> list[dict[str, object]]:
    report = _read_json(report_json_path)
    if not isinstance(report, dict):
        return []

    facts: list[dict[str, object]] = []
    sections = (
        ("decisions", "decision"),
        ("tasks", "action_item"),
        ("questions", "open_question"),
        ("risks", "risk"),
    )
    for section_name, fact_type in sections:
        entries = report.get(section_name)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = _first_string(entry, ("summary", "text", "description"))
            title = _first_string(entry, ("title", "id")) or fact_type
            facts.append(
                {
                    "fact_type": fact_type,
                    "title": title,
                    "text": text,
                    "owner": _string_or_empty(entry.get("owner")),
                    "due": _string_or_empty(entry.get("due")),
                    "citations": _extract_citations(entry),
                    "payload": entry,
                }
            )
    return facts


def sync_meeting_memory(meeting_dir: Path, manifest_path: Path) -> str | None:
    database_url = database_url_from_env()
    if not database_url:
        return None
    try:
        meeting_id = upsert_meeting_memory(database_url, meeting_dir, manifest_path)
    except Exception as exc:
        return f"Memory sync failed: {exc}"
    return sync_meeting_embeddings_from_env(database_url, meeting_id)


def load_report_context_pack(
    *,
    meeting_id: str,
    title: str,
    source_filename: str,
    limit: int = 12,
) -> tuple[dict[str, object] | None, str | None]:
    database_url = database_url_from_env()
    if not database_url:
        return None, None
    try:
        context_pack = build_report_context_pack(
            database_url,
            meeting_id=meeting_id,
            title=title,
            source_filename=source_filename,
            limit=limit,
        )
        if not _context_pack_has_content(context_pack):
            return None, None
        return context_pack, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Context pack failed: {exc}"


def build_report_context_pack(
    database_url: str,
    *,
    meeting_id: str,
    title: str,
    source_filename: str,
    limit: int = 12,
) -> dict[str, object]:
    import psycopg
    from psycopg.rows import dict_row

    query = _context_query(title=title, source_filename=source_filename)
    bounded_limit = max(1, min(int(limit), 50))
    semantic_chunks: list[dict[str, object]] = []
    with psycopg.connect(database_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            fact_where, fact_score, fact_params = _token_search_sql(
                "lower(f.title || ' ' || f.text)",
                query,
            )
            cur.execute(
                f"""
                SELECT
                    f.meeting_id,
                    m.title AS meeting_title,
                    m.source_filename,
                    f.fact_type,
                    f.title,
                    f.text,
                    f.owner,
                    f.due,
                    f.citations
                FROM meeting_facts f
                JOIN meetings m ON m.meeting_id = f.meeting_id
                WHERE f.meeting_id <> %s
                  AND ({fact_where})
                ORDER BY {fact_score} DESC, m.updated_at DESC, f.id DESC
                LIMIT %s
                """,
                (meeting_id, *fact_params, *fact_params, bounded_limit),
            )
            facts = [_plain_dict(row) for row in cur.fetchall()]
            if not facts:
                cur.execute(
                    """
                    SELECT
                        f.meeting_id,
                        m.title AS meeting_title,
                        m.source_filename,
                        f.fact_type,
                        f.title,
                        f.text,
                        f.owner,
                        f.due,
                        f.citations
                    FROM meeting_facts f
                    JOIN meetings m ON m.meeting_id = f.meeting_id
                    WHERE f.meeting_id <> %s
                    ORDER BY m.updated_at DESC, f.id DESC
                    LIMIT %s
                    """,
                    (meeting_id, bounded_limit),
                )
                facts = [_plain_dict(row) for row in cur.fetchall()]

            segment_where, segment_score, segment_params = _token_search_sql("lower(s.text)", query)
            cur.execute(
                f"""
                SELECT
                    s.meeting_id,
                    m.title AS meeting_title,
                    m.source_filename,
                    s.segment_id,
                    s.start_seconds,
                    s.end_seconds,
                    s.speaker,
                    s.text
                FROM meeting_segments s
                JOIN meetings m ON m.meeting_id = s.meeting_id
                WHERE s.meeting_id <> %s
                  AND ({segment_where})
                ORDER BY {segment_score} DESC, m.updated_at DESC, s.start_seconds ASC
                LIMIT %s
                """,
                (meeting_id, *segment_params, *segment_params, bounded_limit),
            )
            segments = [_plain_dict(row) for row in cur.fetchall()]
            if not segments:
                cur.execute(
                    """
                    SELECT
                        s.meeting_id,
                        m.title AS meeting_title,
                        m.source_filename,
                        s.segment_id,
                        s.start_seconds,
                        s.end_seconds,
                        s.speaker,
                        s.text
                    FROM meeting_segments s
                    JOIN meetings m ON m.meeting_id = s.meeting_id
                    WHERE s.meeting_id <> %s
                    ORDER BY m.updated_at DESC, s.start_seconds ASC
                    LIMIT %s
                    """,
                    (meeting_id, bounded_limit),
                )
                segments = [_plain_dict(row) for row in cur.fetchall()]

    api_key = openai_api_key_from_env()
    if api_key and query:
        try:
            semantic_chunks = search_embedding_chunks(
                database_url,
                query=query,
                api_key=api_key,
                limit=bounded_limit,
                exclude_meeting_id=meeting_id,
                model=os.environ.get("MAC_TRANSCRIBER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            )
        except Exception:
            semantic_chunks = []

    return {
        "meeting_id": meeting_id,
        "title": title,
        "source_filename": source_filename,
        "query": query,
        "generated_at": datetime.now(UTC).isoformat(),
        "facts": facts,
        "segments": segments,
        "embedding_chunks": semantic_chunks,
    }


def build_embedding_chunks(
    *,
    meeting: Mapping[str, object],
    segments: list[Mapping[str, object]],
    facts: list[Mapping[str, object]],
    max_segments: int = DEFAULT_EMBEDDING_CHUNK_SEGMENTS,
    max_chars: int = DEFAULT_EMBEDDING_CHUNK_CHARS,
) -> list[dict[str, object]]:
    meeting_id = _string_or_empty(meeting.get("meeting_id"))
    title = _string_or_empty(meeting.get("title"))
    source_filename = _string_or_empty(meeting.get("source_filename"))
    chunks: list[dict[str, object]] = []

    current: list[Mapping[str, object]] = []
    current_chars = 0
    bounded_max_segments = max(1, int(max_segments))
    bounded_max_chars = max(300, int(max_chars))

    def flush_segments() -> None:
        nonlocal current, current_chars
        if not current:
            return
        first_id = _segment_id_for_embedding(current[0], 1)
        last_id = _segment_id_for_embedding(current[-1], len(current))
        segment_ids = [_segment_id_for_embedding(row, index) for index, row in enumerate(current, start=1)]
        speakers = sorted(
            {
                _string_or_empty(row.get("speaker"))
                for row in current
                if _string_or_empty(row.get("speaker"))
            }
        )
        start_seconds = _number_or_none(current[0].get("start_seconds"))
        end_seconds = _number_or_none(current[-1].get("end_seconds"))
        source_id = f"{first_id}-{last_id}"
        text = "\n".join(
            [
                _embedding_header(title=title, source_filename=source_filename),
                *[_format_segment_for_embedding(row, index) for index, row in enumerate(current, start=1)],
            ]
        ).strip()
        chunks.append(
            {
                "meeting_id": meeting_id,
                "chunk_id": f"segments:{source_id}",
                "source_table": "meeting_segments",
                "source_id": source_id,
                "text": text,
                "metadata": {
                    "kind": "transcript_segments",
                    "meeting_id": meeting_id,
                    "title": title,
                    "source_filename": source_filename,
                    "segment_ids": segment_ids,
                    "speakers": speakers,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                },
            }
        )
        current = []
        current_chars = 0

    for row in segments:
        text = _string_or_empty(row.get("text"))
        if not text:
            continue
        rendered = _format_segment_for_embedding(row, len(current) + 1)
        would_exceed_chars = current and current_chars + len(rendered) > bounded_max_chars
        would_exceed_count = len(current) >= bounded_max_segments
        if would_exceed_chars or would_exceed_count:
            flush_segments()
        current.append(row)
        current_chars += len(rendered)
    flush_segments()

    for fact in facts:
        title_text = _string_or_empty(fact.get("title"))
        body_text = _string_or_empty(fact.get("text"))
        if not title_text and not body_text:
            continue
        fact_id = _string_or_empty(fact.get("id")) or _fact_identity(fact, len(chunks) + 1)
        fact_type = _string_or_empty(fact.get("fact_type"))
        owner = _string_or_empty(fact.get("owner"))
        due = _string_or_empty(fact.get("due"))
        citations = _list_of_strings(fact.get("citations"))
        text = "\n".join(
            part
            for part in (
                _embedding_header(title=title, source_filename=source_filename),
                f"Fact type: {fact_type}" if fact_type else "",
                f"Title: {title_text}" if title_text else "",
                f"Text: {body_text}" if body_text else "",
                f"Owner: {owner}" if owner else "",
                f"Due: {due}" if due else "",
                f"Citations: {', '.join(citations)}" if citations else "",
            )
            if part
        ).strip()
        chunks.append(
            {
                "meeting_id": meeting_id,
                "chunk_id": f"fact:{fact_id}",
                "source_table": "meeting_facts",
                "source_id": fact_id,
                "text": text,
                "metadata": {
                    "kind": "meeting_fact",
                    "meeting_id": meeting_id,
                    "title": title,
                    "source_filename": source_filename,
                    "fact_type": fact_type,
                    "fact_title": title_text,
                    "owner": owner,
                    "due": due,
                    "citations": citations,
                },
            }
        )

    return chunks


def upsert_meeting_embeddings(
    database_url: str,
    *,
    api_key: str,
    meeting_id: str | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> dict[str, int]:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    bounded_batch_size = max(1, int(batch_size))
    stats = {"meetings": 0, "chunks": 0}
    with psycopg.connect(database_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(build_schema_sql())
            if meeting_id:
                cur.execute(
                    """
                    SELECT meeting_id, title, source_filename
                    FROM meetings
                    WHERE meeting_id = %s
                    ORDER BY updated_at ASC, id ASC
                    """,
                    (meeting_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT meeting_id, title, source_filename
                    FROM meetings
                    ORDER BY updated_at ASC, id ASC
                    """
                )
            meetings = [_plain_dict(row) for row in cur.fetchall()]

            for meeting in meetings:
                current_meeting_id = _string_or_empty(meeting.get("meeting_id"))
                cur.execute(
                    """
                    SELECT segment_id, start_seconds, end_seconds, speaker, track, text
                    FROM meeting_segments
                    WHERE meeting_id = %s
                    ORDER BY start_seconds ASC, segment_id ASC
                    """,
                    (current_meeting_id,),
                )
                segments = [_plain_dict(row) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT id, fact_type, title, text, owner, due, citations
                    FROM meeting_facts
                    WHERE meeting_id = %s
                    ORDER BY id ASC
                    """,
                    (current_meeting_id,),
                )
                facts = [_plain_dict(row) for row in cur.fetchall()]
                chunks = build_embedding_chunks(meeting=meeting, segments=segments, facts=facts)

                cur.execute("DELETE FROM embedding_chunks WHERE meeting_id = %s", (current_meeting_id,))
                if not chunks:
                    continue

                for batch in _chunked(chunks, bounded_batch_size):
                    embeddings = _post_openai_embeddings(
                        api_key=api_key,
                        input_texts=[_string_or_empty(chunk.get("text")) for chunk in batch],
                        model=model,
                        dimensions=dimensions,
                    )
                    if len(embeddings) != len(batch):
                        raise RuntimeError("OpenAI embeddings response length did not match request")
                    for chunk, embedding in zip(batch, embeddings, strict=True):
                        cur.execute(
                            """
                            INSERT INTO embedding_chunks (
                                meeting_id, chunk_id, source_table, source_id, text, embedding, metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                            ON CONFLICT (meeting_id, chunk_id) DO UPDATE SET
                                source_table = EXCLUDED.source_table,
                                source_id = EXCLUDED.source_id,
                                text = EXCLUDED.text,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                created_at = now()
                            """,
                            (
                                current_meeting_id,
                                chunk["chunk_id"],
                                chunk["source_table"],
                                chunk["source_id"],
                                chunk["text"],
                                _vector_literal(embedding, dimensions=dimensions),
                                Jsonb(chunk["metadata"]),
                            ),
                        )
                stats["meetings"] += 1
                stats["chunks"] += len(chunks)
        conn.commit()
    return stats


def sync_meeting_embeddings_from_env(
    database_url: str,
    meeting_id: str,
    env: Mapping[str, str] = os.environ,
) -> str | None:
    api_key = openai_api_key_from_env(env)
    if not api_key:
        return None
    model = env.get("MAC_TRANSCRIBER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
    batch_size = _int_from_env(env, "MAC_TRANSCRIBER_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
    try:
        upsert_meeting_embeddings(
            database_url,
            api_key=api_key,
            meeting_id=meeting_id,
            model=model or DEFAULT_EMBEDDING_MODEL,
            batch_size=batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Embedding sync failed: {exc}"
    return None


def search_embedding_chunks(
    database_url: str,
    *,
    query: str,
    api_key: str,
    limit: int = 8,
    exclude_meeting_id: str | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[dict[str, object]]:
    import psycopg
    from psycopg.rows import dict_row

    query_text = _string_or_empty(query)
    if not query_text:
        return []
    query_embedding = _post_openai_embeddings(
        api_key=api_key,
        input_texts=[query_text],
        model=model,
        dimensions=dimensions,
    )[0]
    with psycopg.connect(database_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            query_vector = _vector_literal(query_embedding, dimensions=dimensions)
            cur.execute("SET LOCAL ivfflat.probes = 10")
            if exclude_meeting_id:
                cur.execute(
                    """
                    SELECT
                        c.meeting_id,
                        m.title AS meeting_title,
                        m.source_filename,
                        c.chunk_id,
                        c.source_table,
                        c.source_id,
                        c.text,
                        c.metadata,
                        c.embedding <=> %s::vector AS distance
                    FROM embedding_chunks c
                    JOIN meetings m ON m.meeting_id = c.meeting_id
                    WHERE c.embedding IS NOT NULL
                      AND c.meeting_id <> %s
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        exclude_meeting_id,
                        query_vector,
                        max(1, int(limit)),
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        c.meeting_id,
                        m.title AS meeting_title,
                        m.source_filename,
                        c.chunk_id,
                        c.source_table,
                        c.source_id,
                        c.text,
                        c.metadata,
                        c.embedding <=> %s::vector AS distance
                    FROM embedding_chunks c
                    JOIN meetings m ON m.meeting_id = c.meeting_id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        query_vector,
                        max(1, int(limit)),
                    ),
                )
            return [_plain_dict(row) for row in cur.fetchall()]


def upsert_meeting_memory(database_url: str, meeting_dir: Path, manifest_path: Path) -> str:
    import psycopg
    from psycopg.types.json import Jsonb

    manifest = _read_json(manifest_path)
    metadata = _read_json(meeting_dir / "input" / "metadata.json")
    status = _read_json(meeting_dir / "status.json")
    transcript = _read_json(meeting_dir / "artifacts" / "transcript.json")

    if not isinstance(manifest, dict):
        manifest = {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(status, dict):
        status = {}

    meeting_id = _meeting_id(meeting_dir, manifest, metadata)
    title = _string_or_none(manifest.get("title")) or _string_or_none(metadata.get("title"))
    source_filename = _string_or_none(manifest.get("source_filename")) or _string_or_none(
        metadata.get("source_filename")
    )
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else None

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(build_schema_sql())
            cur.execute(
                """
                INSERT INTO meetings (
                    meeting_id, title, source_filename, metadata, result, status, manifest, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (meeting_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_filename = EXCLUDED.source_filename,
                    metadata = EXCLUDED.metadata,
                    result = EXCLUDED.result,
                    status = EXCLUDED.status,
                    manifest = EXCLUDED.manifest,
                    updated_at = now()
                """,
                (
                    meeting_id,
                    title,
                    source_filename,
                    Jsonb(metadata),
                    Jsonb(result),
                    Jsonb(status),
                    Jsonb(manifest),
                ),
            )
            _delete_child_rows(cur, meeting_id)
            _upsert_artifacts(cur, Jsonb, meeting_id, manifest)
            _upsert_segments(cur, Jsonb, meeting_id, transcript)
            _upsert_facts(cur, Jsonb, meeting_id, meeting_dir / "artifacts" / "report.json")
        conn.commit()
    return meeting_id


def _delete_child_rows(cur, meeting_id: str) -> None:
    for table in (
        "report_context_links",
        "meeting_facts",
        "meeting_segments",
        "meeting_artifacts",
    ):
        cur.execute(f"DELETE FROM {table} WHERE meeting_id = %s", (meeting_id,))


def _upsert_artifacts(cur, jsonb, meeting_id: str, manifest: dict) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        return
    for entry in files:
        if not isinstance(entry, dict):
            continue
        relative_path = _string_or_none(entry.get("relative_path"))
        if not relative_path:
            continue
        cur.execute(
            """
            INSERT INTO meeting_artifacts (
                meeting_id, relative_path, kind, size_bytes, sha256, modified_at, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (meeting_id, relative_path) DO UPDATE SET
                kind = EXCLUDED.kind,
                size_bytes = EXCLUDED.size_bytes,
                sha256 = EXCLUDED.sha256,
                modified_at = EXCLUDED.modified_at,
                payload = EXCLUDED.payload
            """,
            (
                meeting_id,
                relative_path,
                _string_or_none(entry.get("kind")),
                entry.get("size_bytes") if isinstance(entry.get("size_bytes"), int) else None,
                _string_or_none(entry.get("sha256")),
                _string_or_none(entry.get("modified_at")),
                jsonb(entry),
            ),
        )


def _upsert_segments(cur, jsonb, meeting_id: str, transcript: object) -> None:
    rows = _transcript_rows(transcript)
    for index, row in enumerate(rows, start=1):
        segment_id = _string_or_none(row.get("segment_id")) or _string_or_none(row.get("id"))
        if not segment_id:
            segment_id = f"S{index:04d}"
        cur.execute(
            """
            INSERT INTO meeting_segments (
                meeting_id, segment_id, start_seconds, end_seconds, speaker, track, text, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (meeting_id, segment_id) DO UPDATE SET
                start_seconds = EXCLUDED.start_seconds,
                end_seconds = EXCLUDED.end_seconds,
                speaker = EXCLUDED.speaker,
                track = EXCLUDED.track,
                text = EXCLUDED.text,
                payload = EXCLUDED.payload
            """,
            (
                meeting_id,
                segment_id,
                _number_or_none(row.get("start")),
                _number_or_none(row.get("end")),
                _string_or_none(row.get("speaker")),
                _string_or_none(row.get("track")),
                _string_or_empty(row.get("text")),
                jsonb({**row, "segment_id": segment_id}),
            ),
        )


def _upsert_facts(cur, jsonb, meeting_id: str, report_path: Path) -> None:
    for fact in extract_facts_from_report(report_path):
        cur.execute(
            """
            INSERT INTO meeting_facts (
                meeting_id, fact_type, title, text, owner, due, citations, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (meeting_id, fact_type, title, text) DO UPDATE SET
                owner = EXCLUDED.owner,
                due = EXCLUDED.due,
                citations = EXCLUDED.citations,
                payload = EXCLUDED.payload
            RETURNING id
            """,
            (
                meeting_id,
                fact["fact_type"],
                fact["title"],
                fact["text"],
                fact["owner"] or None,
                fact["due"] or None,
                jsonb(fact["citations"]),
                jsonb(fact["payload"]),
            ),
        )
        fact_id = cur.fetchone()[0]
        for citation in fact["citations"]:
            cur.execute(
                """
                INSERT INTO report_context_links (meeting_id, fact_id, segment_id, citation, payload)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (meeting_id, fact_id, citation) DO UPDATE SET
                    segment_id = EXCLUDED.segment_id,
                    payload = EXCLUDED.payload
                """,
                (
                    meeting_id,
                    fact_id,
                    _segment_id_from_citation(citation),
                    citation,
                    jsonb({"fact_type": fact["fact_type"], "title": fact["title"]}),
                ),
            )


def _post_openai_embeddings(
    *,
    api_key: str,
    input_texts: list[str],
    model: str,
    dimensions: int,
) -> list[list[float]]:
    payload = {
        "model": model,
        "input": input_texts,
        "dimensions": dimensions,
    }
    request = urllib.request.Request(
        OPENAI_EMBEDDINGS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI embeddings returned HTTP {exc.code}: {detail}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc

    data = body.get("data")
    if not isinstance(data, list):
        raise RuntimeError("OpenAI embeddings response did not contain data")
    ordered = sorted(
        (item for item in data if isinstance(item, dict)),
        key=lambda item: int(item.get("index", 0)),
    )
    embeddings: list[list[float]] = []
    for item in ordered:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("OpenAI embeddings response item has no embedding")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def _context_query(*, title: str, source_filename: str) -> str:
    title_text = _string_or_empty(title)
    if title_text:
        return title_text
    source_text = _string_or_empty(source_filename)
    generic_names = {
        "audio.m4a",
        "audio.wav",
        "recording.m4a",
        "recording.wav",
        "zoom_timeline.json",
    }
    if source_text.lower() in generic_names:
        return ""
    return source_text


def _token_search_sql(expression: str, query: str) -> tuple[str, str, list[str]]:
    tokens = _query_tokens(query)
    if not tokens:
        return "false", "0", []
    clauses = [f"{expression} LIKE %s" for _token in tokens]
    score_parts = [f"CASE WHEN {expression} LIKE %s THEN 1 ELSE 0 END" for _token in tokens]
    params = [f"%{token}%" for token in tokens]
    return " OR ".join(clauses), " + ".join(score_parts), params


def _query_tokens(query: str) -> list[str]:
    import re

    seen: set[str] = set()
    tokens: list[str] = []
    for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", query.lower()):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _embedding_header(*, title: str, source_filename: str) -> str:
    parts = []
    if title:
        parts.append(f"Meeting: {title}")
    if source_filename:
        parts.append(f"Source: {source_filename}")
    return "\n".join(parts)


def _format_segment_for_embedding(row: Mapping[str, object], fallback_index: int) -> str:
    segment_id = _segment_id_for_embedding(row, fallback_index)
    start = _number_or_none(row.get("start_seconds"))
    end = _number_or_none(row.get("end_seconds"))
    speaker = _string_or_empty(row.get("speaker"))
    text = _string_or_empty(row.get("text"))
    time_range = ""
    if start is not None and end is not None:
        time_range = f" {_format_seconds(start)}-{_format_seconds(end)}"
    speaker_part = f" {speaker}:" if speaker else ""
    return f"{segment_id}{time_range}{speaker_part} {text}".strip()


def _segment_id_for_embedding(row: Mapping[str, object], fallback_index: int) -> str:
    return _string_or_none(row.get("segment_id")) or f"S{fallback_index:04d}"


def _fact_identity(fact: Mapping[str, object], fallback_index: int) -> str:
    fact_type = _string_or_empty(fact.get("fact_type")) or "fact"
    title = _string_or_empty(fact.get("title")) or str(fallback_index)
    normalized = "".join(char if char.isalnum() else "-" for char in title.lower()).strip("-")
    return f"{fact_type}:{normalized or fallback_index}"


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _vector_literal(values: list[float], *, dimensions: int = EMBEDDING_DIMENSIONS) -> str:
    if len(values) != dimensions:
        raise RuntimeError(f"Embedding has {len(values)} dimensions; expected {dimensions}")
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _chunked(items: list[dict[str, object]], size: int) -> list[list[dict[str, object]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _list_of_strings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _int_from_env(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, str(default)).strip())
    except ValueError:
        return default


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _plain_dict(row: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in dict(row).items()}


def _context_pack_has_content(context_pack: object) -> bool:
    if not isinstance(context_pack, dict):
        return False
    return bool(context_pack.get("facts") or context_pack.get("segments"))


def _meeting_id(meeting_dir: Path, manifest: dict, metadata: dict) -> str:
    return (
        _string_or_none(manifest.get("meeting_id"))
        or _string_or_none(metadata.get("meeting_id"))
        or meeting_dir.name
    )


def _transcript_rows(transcript: object) -> list[dict]:
    if isinstance(transcript, list):
        return [row for row in transcript if isinstance(row, dict)]
    if isinstance(transcript, dict):
        for key in ("segments", "transcript"):
            rows = transcript.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _extract_citations(entry: dict) -> list[str]:
    for key in ("citations", "refs"):
        value = entry.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item is not None and str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    ref = entry.get("ref")
    if isinstance(ref, str) and ref.strip():
        return [ref.strip()]
    return []


def _first_string(entry: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _string_or_empty(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _string_or_none(value: object) -> str | None:
    text = _string_or_empty(value)
    return text or None


def _number_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _segment_id_from_citation(citation: object) -> str | None:
    if not isinstance(citation, str):
        return None
    first = citation.strip().split(maxsplit=1)[0]
    return first if first.startswith("S") else None
