import json
import sys
import types

from mac_transcriber import memory_db


def test_database_url_from_env_uses_primary_and_fallback_keys():
    assert (
        memory_db.database_url_from_env(
            {
                "MAC_TRANSCRIBER_DATABASE_URL": " postgresql://primary/db ",
                "MAC_TRANSCRIBER_POSTGRES_DSN": " postgresql://fallback/db ",
            }
        )
        == "postgresql://primary/db"
    )
    assert (
        memory_db.database_url_from_env(
            {"MAC_TRANSCRIBER_POSTGRES_DSN": " postgresql://fallback/db "}
        )
        == "postgresql://fallback/db"
    )
    assert memory_db.database_url_from_env({}) == ""


def test_memory_enabled_requires_database_url():
    assert memory_db.memory_enabled({"MAC_TRANSCRIBER_DATABASE_URL": "postgres://db"})
    assert not memory_db.memory_enabled({"MAC_TRANSCRIBER_DATABASE_URL": "   "})
    assert not memory_db.memory_enabled({})


def test_build_schema_sql_declares_required_extensions_tables_and_indexes():
    sql = memory_db.build_schema_sql()
    normalized = " ".join(sql.lower().split())

    for extension in ("vector", "pg_trgm", "unaccent"):
        assert f"create extension if not exists {extension}" in normalized

    for table in (
        "meetings",
        "meeting_artifacts",
        "meeting_segments",
        "meeting_facts",
        "report_context_links",
        "embedding_chunks",
    ):
        assert f"create table if not exists {table}" in normalized

    for expected in (
        "unique (meeting_id)",
        "unique (meeting_id, segment_id)",
        "unique (meeting_id, relative_path)",
        "using gin",
        "gin_trgm_ops",
        "vector_cosine_ops",
        "jsonb_path_ops",
    ):
        assert expected in normalized


def test_extract_facts_from_report_reads_protocol_sections(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "id": "D1",
                        "title": "Ship MVP",
                        "summary": "Launch the first version.",
                        "citations": ["00:01:00"],
                    }
                ],
                "tasks": [
                    {
                        "id": "T1",
                        "description": "Prepare rollout checklist.",
                        "owner": "Ilya",
                        "due": "2026-06-20",
                        "ref": "S0003 · 00:03:00",
                        "note": "Needs final owner confirmation.",
                    }
                ],
                "questions": [{"id": "Q1", "text": "Who owns onboarding?"}],
                "risks": [{"id": "R1", "summary": "Budget may slip."}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    facts = memory_db.extract_facts_from_report(report_path)

    assert [fact["fact_type"] for fact in facts] == [
        "decision",
        "action_item",
        "open_question",
        "risk",
    ]
    assert facts[0]["title"] == "Ship MVP"
    assert facts[0]["text"] == "Launch the first version."
    assert facts[0]["citations"] == ["00:01:00"]
    assert facts[1]["owner"] == "Ilya"
    assert facts[1]["due"] == "2026-06-20"
    assert facts[1]["citations"] == ["S0003 · 00:03:00"]
    assert facts[1]["payload"]["note"] == "Needs final owner confirmation."
    assert facts[2]["title"] == "Q1"
    assert facts[2]["text"] == "Who owns onboarding?"
    assert facts[3]["payload"] == {"id": "R1", "summary": "Budget may slip."}


def test_extract_facts_from_report_returns_empty_for_missing_or_invalid_json(tmp_path):
    missing_path = tmp_path / "missing.json"
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not json", encoding="utf-8")

    assert memory_db.extract_facts_from_report(missing_path) == []
    assert memory_db.extract_facts_from_report(invalid_path) == []


def _write_health(tmp_path, generated_by=None, status="ok"):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    if generated_by is not None:
        (artifacts / "report_health.json").write_text(
            json.dumps({"status": status, "generated_by": generated_by}),
            encoding="utf-8",
        )
    return report_path


def test_report_facts_trusted_only_for_ai_reports(tmp_path):
    # AI-отчёт — доверяем; local-fallback и "ok"-local — нет.
    assert memory_db._report_facts_trusted(
        _write_health(tmp_path / "ai", generated_by="gpt-5.5/chunked")
    )
    assert memory_db._report_facts_trusted(
        _write_health(
            tmp_path / "ai_degraded", generated_by="gpt-5.5/chunked", status="degraded"
        )
    )
    assert not memory_db._report_facts_trusted(
        _write_health(tmp_path / "local_ok", generated_by="local", status="ok")
    )
    # Нет report_health.json вовсе — не доверяем (консервативно, не плодим мусор).
    assert not memory_db._report_facts_trusted(
        _write_health(tmp_path / "no_health", generated_by=None)
    )


def test_sync_meeting_memory_noops_without_database_url(monkeypatch, tmp_path):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.delenv("MAC_TRANSCRIBER_DATABASE_URL", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_POSTGRES_DSN", raising=False)
    monkeypatch.setattr(memory_db, "upsert_meeting_memory", fail_if_called)

    assert memory_db.sync_meeting_memory(tmp_path, tmp_path / "manifest.json") is None
    assert not called


def test_sync_meeting_memory_reports_import_or_connection_errors(monkeypatch, tmp_path):
    def raise_connection_error(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setenv("MAC_TRANSCRIBER_DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setattr(memory_db, "upsert_meeting_memory", raise_connection_error)

    error = memory_db.sync_meeting_memory(tmp_path, tmp_path / "manifest.json")

    assert error is not None
    assert "connection refused" in error


def test_load_report_context_pack_noops_without_database_url(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.delenv("MAC_TRANSCRIBER_DATABASE_URL", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_POSTGRES_DSN", raising=False)
    monkeypatch.setattr(memory_db, "build_report_context_pack", fail_if_called)

    context_pack, error = memory_db.load_report_context_pack(
        meeting_id="current",
        title="Current meeting",
        source_filename="current.m4a",
    )

    assert context_pack is None
    assert error is None
    assert not called


def test_build_report_context_pack_reads_recent_facts_and_segments(monkeypatch):
    calls = []

    class FakeCursor:
        def __init__(self):
            self.rows = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            calls.append((sql, params))
            normalized = " ".join(sql.lower().split())
            if "from meeting_facts" in normalized:
                self.rows = [
                    {
                        "meeting_id": "old-meeting",
                        "meeting_title": "Old sync",
                        "source_filename": "old.m4a",
                        "fact_type": "decision",
                        "title": "D1",
                        "text": "Ранее решили хранить транскрипты.",
                        "owner": None,
                        "due": None,
                        "citations": ["S0003 · 00:02"],
                    }
                ]
            elif "from meeting_segments" in normalized:
                self.rows = [
                    {
                        "meeting_id": "old-meeting",
                        "meeting_title": "Old sync",
                        "source_filename": "old.m4a",
                        "segment_id": "S0003",
                        "start_seconds": 12.0,
                        "end_seconds": 18.0,
                        "speaker": "Илья",
                        "text": "Нужно хранить транскрипты локально.",
                    }
                ]
            else:
                self.rows = []

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, *args, **kwargs):
            return FakeCursor()

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda database_url: FakeConnection()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    context_pack = memory_db.build_report_context_pack(
        "postgresql://example/db",
        meeting_id="current",
        title="Current memory sync",
        source_filename="current.m4a",
        limit=3,
    )

    assert context_pack["meeting_id"] == "current"
    assert context_pack["query"] == "Current memory sync"
    assert context_pack["facts"][0]["text"] == "Ранее решили хранить транскрипты."
    assert context_pack["segments"][0]["segment_id"] == "S0003"
    assert any(params and "current" in params for _sql, params in calls)


def test_build_report_context_pack_does_not_pollute_query_with_generic_audio_filename(
    monkeypatch,
):
    semantic_calls = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, _sql, _params=None):
            self.rows = []

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, *args, **kwargs):
            return FakeCursor()

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda _database_url: FakeConnection()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        memory_db,
        "search_embedding_chunks",
        lambda *args, **kwargs: semantic_calls.append(kwargs) or [],
    )

    context_pack = memory_db.build_report_context_pack(
        "postgresql://example/db",
        meeting_id="current",
        title="База данных",
        source_filename="audio.m4a",
        limit=3,
    )

    assert context_pack["query"] == "База данных"
    assert semantic_calls[0]["query"] == "База данных"


def test_build_report_context_pack_falls_back_to_recent_segments(monkeypatch):
    segment_queries = 0

    class FakeCursor:
        def __init__(self):
            self.rows = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            nonlocal segment_queries
            normalized = " ".join(sql.lower().split())
            if "from meeting_facts" in normalized:
                self.rows = []
            elif "from meeting_segments" in normalized:
                segment_queries += 1
                self.rows = (
                    []
                    if segment_queries == 1
                    else [
                        {
                            "meeting_id": "old-meeting",
                            "meeting_title": "Old sync",
                            "source_filename": "old.json",
                            "segment_id": "S0001",
                            "start_seconds": 1.0,
                            "end_seconds": 2.0,
                            "speaker": "Ilya",
                            "text": "Недавний контекст без совпадения по названию.",
                        }
                    ]
                )
            else:
                self.rows = []

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, *args, **kwargs):
            return FakeCursor()

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda database_url: FakeConnection()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    context_pack = memory_db.build_report_context_pack(
        "postgresql://example/db",
        meeting_id="current",
        title="Unrelated title",
        source_filename="current.json",
        limit=3,
    )

    assert segment_queries == 2
    assert (
        context_pack["segments"][0]["text"]
        == "Недавний контекст без совпадения по названию."
    )


def test_build_report_context_pack_adds_semantic_embedding_chunks(monkeypatch):
    class FakeCursor:
        def __init__(self):
            self.rows = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            normalized = " ".join(sql.lower().split())
            if (
                "from meeting_facts" in normalized
                or "from meeting_segments" in normalized
            ):
                self.rows = []
            else:
                self.rows = []

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, *args, **kwargs):
            return FakeCursor()

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda database_url: FakeConnection()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    semantic_calls = []

    def fake_search(*args, **kwargs):
        semantic_calls.append((args, kwargs))
        return [
            {
                "meeting_id": "old-meeting",
                "meeting_title": "Old architecture sync",
                "chunk_id": "segments:S0001-S0002",
                "text": "Раньше обсуждали семантическую память.",
                "distance": 0.12,
            }
        ]

    monkeypatch.setattr(memory_db, "search_embedding_chunks", fake_search)

    context_pack = memory_db.build_report_context_pack(
        "postgresql://example/db",
        meeting_id="current",
        title="semantic memory",
        source_filename="current.m4a",
        limit=3,
    )

    assert (
        context_pack["embedding_chunks"][0]["text"]
        == "Раньше обсуждали семантическую память."
    )
    assert semantic_calls[0][0] == ("postgresql://example/db",)
    assert semantic_calls[0][1]["query"] == "semantic memory"
    assert semantic_calls[0][1]["api_key"] == "test-key"
    assert semantic_calls[0][1]["exclude_meeting_id"] == "current"


def test_build_embedding_chunks_groups_segments_and_facts():
    chunks = memory_db.build_embedding_chunks(
        meeting={
            "meeting_id": "meeting-123",
            "title": "Architecture sync",
            "source_filename": "audio.m4a",
        },
        segments=[
            {
                "segment_id": "S0001",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "speaker": "Ilya",
                "text": "Нужно сохранить транскрипты.",
            },
            {
                "segment_id": "S0002",
                "start_seconds": 2.0,
                "end_seconds": 4.0,
                "speaker": "Aziz",
                "text": "И потом искать по ним семантически.",
            },
        ],
        facts=[
            {
                "id": 42,
                "fact_type": "action_item",
                "title": "Сделать embeddings",
                "text": "Заполнить embedding_chunks через OpenAI.",
                "owner": "Ilya",
                "due": "",
                "citations": ["S0001"],
            }
        ],
        max_segments=8,
        max_chars=2000,
    )

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "segments:S0001-S0002",
        "fact:42",
    ]
    assert chunks[0]["source_table"] == "meeting_segments"
    assert chunks[0]["source_id"] == "S0001-S0002"
    assert "Architecture sync" in chunks[0]["text"]
    assert "S0001" in chunks[0]["text"]
    assert chunks[0]["metadata"]["segment_ids"] == ["S0001", "S0002"]
    assert chunks[0]["metadata"]["speakers"] == ["Aziz", "Ilya"]
    assert chunks[1]["source_table"] == "meeting_facts"
    assert chunks[1]["source_id"] == "42"
    assert chunks[1]["metadata"]["fact_type"] == "action_item"


def test_upsert_meeting_embeddings_requests_openai_and_writes_vectors(monkeypatch):
    calls = []

    class FakeJsonb:
        def __init__(self, value):
            self.value = value

    class FakeCursor:
        def __init__(self):
            self.rows = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            calls.append((sql, params))
            normalized = " ".join(sql.lower().split())
            if "from meetings" in normalized:
                self.rows = [
                    {
                        "meeting_id": "meeting-123",
                        "title": "Architecture sync",
                        "source_filename": "audio.m4a",
                    }
                ]
            elif "from meeting_segments" in normalized:
                self.rows = [
                    {
                        "segment_id": "S0001",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "speaker": "Ilya",
                        "track": "audio.m4a",
                        "text": "Нужно сохранить транскрипты.",
                    }
                ]
            elif "from meeting_facts" in normalized:
                self.rows = [
                    {
                        "id": 42,
                        "fact_type": "action_item",
                        "title": "Сделать embeddings",
                        "text": "Заполнить embedding_chunks через OpenAI.",
                        "owner": "Ilya",
                        "due": None,
                        "citations": ["S0001"],
                    }
                ]
            else:
                self.rows = []

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()
            self.committed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, *args, **kwargs):
            return self.cursor_obj

        def commit(self):
            self.committed = True

    fake_connection = FakeConnection()
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda database_url: fake_connection
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    fake_types = types.ModuleType("psycopg.types")
    fake_json = types.ModuleType("psycopg.types.json")
    fake_json.Jsonb = FakeJsonb
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)
    monkeypatch.setitem(sys.modules, "psycopg.types", fake_types)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", fake_json)

    requested_batches = []

    def fake_embeddings(*, api_key, input_texts, model, dimensions):
        requested_batches.append(
            {
                "api_key": api_key,
                "input_texts": input_texts,
                "model": model,
                "dimensions": dimensions,
            }
        )
        return [
            [float(index)] * dimensions
            for index, _text in enumerate(input_texts, start=1)
        ]

    monkeypatch.setattr(memory_db, "_post_openai_embeddings", fake_embeddings)

    stats = memory_db.upsert_meeting_embeddings(
        "postgresql://example/db",
        api_key="test-key",
        meeting_id="meeting-123",
        model="text-embedding-3-small",
        batch_size=10,
    )

    assert stats == {"meetings": 1, "chunks": 2}
    assert requested_batches[0]["api_key"] == "test-key"
    assert requested_batches[0]["model"] == "text-embedding-3-small"
    assert requested_batches[0]["dimensions"] == 1536
    assert len(requested_batches[0]["input_texts"]) == 2
    insert_calls = [
        (sql, params)
        for sql, params in calls
        if "insert into embedding_chunks" in " ".join(sql.lower().split())
    ]
    assert len(insert_calls) == 2
    assert "segments:S0001-S0001" in insert_calls[0][1]
    assert insert_calls[0][1][5].startswith("[1.0,")
    assert insert_calls[1][1][5].startswith("[2.0,")
    assert fake_connection.committed


def test_search_embedding_chunks_sets_ivfflat_probes(monkeypatch):
    calls = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            calls.append((" ".join(sql.lower().split()), params))
            self.rows = []

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, *args, **kwargs):
            return FakeCursor()

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda _database_url: FakeConnection()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)
    monkeypatch.setattr(
        memory_db,
        "_post_openai_embeddings",
        lambda **_kwargs: [[0.1] * memory_db.EMBEDDING_DIMENSIONS],
    )

    memory_db.search_embedding_chunks(
        "postgresql://example/db",
        query="semantic query",
        api_key="test-key",
    )

    assert calls[0][0] == "set local ivfflat.probes = 10"


def test_load_report_context_pack_treats_empty_pack_as_no_context(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_DATABASE_URL", "postgresql://example/db")
    monkeypatch.setattr(
        memory_db,
        "build_report_context_pack",
        lambda *_args, **_kwargs: {
            "meeting_id": "current",
            "facts": [],
            "segments": [],
        },
    )

    context_pack, error = memory_db.load_report_context_pack(
        meeting_id="current",
        title="Current meeting",
        source_filename="current.m4a",
    )

    assert context_pack is None
    assert error is None


def test_upsert_meeting_memory_deletes_stale_children_and_inserts_current_rows(
    monkeypatch, tmp_path
):
    meeting_dir = tmp_path / "meeting-789"
    input_dir = meeting_dir / "input"
    artifacts_dir = meeting_dir / "artifacts"
    input_dir.mkdir(parents=True)
    artifacts_dir.mkdir()

    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "meeting_id": "meeting-789",
                "title": "Planning",
                "source_filename": "planning.m4a",
                "result": {"status": "done"},
                "files": [
                    {
                        "relative_path": "artifacts/transcript.json",
                        "kind": "transcript",
                        "size_bytes": 123,
                        "sha256": "abc",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "metadata.json").write_text(
        json.dumps({"meeting_id": "meeting-789", "title": "Planning"}),
        encoding="utf-8",
    )
    (meeting_dir / "status.json").write_text(
        json.dumps({"phase": "done"}),
        encoding="utf-8",
    )
    (artifacts_dir / "transcript.json").write_text(
        json.dumps(
            [
                {
                    "start": 1.0,
                    "end": 2.5,
                    "speaker": "A",
                    "track": "audio.m4a",
                    "text": "We should ship this.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "report.json").write_text(
        json.dumps(
            {
                "decisions": [{"id": "D1", "text": "Ship it.", "ref": "S0001 · 00:01"}],
                "tasks": [{"id": "T1", "text": "Write notes.", "owner": "Ilya"}],
            }
        ),
        encoding="utf-8",
    )
    # Факты в память берутся только из доверенных AI-отчётов (не local-fallback).
    (artifacts_dir / "report_health.json").write_text(
        json.dumps({"status": "ok", "generated_by": "gpt-5.5/chunked"}),
        encoding="utf-8",
    )

    calls = []

    class FakeJsonb:
        def __init__(self, value):
            self.value = value

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            calls.append((sql, params))

        def fetchone(self):
            return (42,)

    class FakeConnection:
        def __init__(self):
            self.cursor_obj = FakeCursor()
            self.committed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

    fake_connection = FakeConnection()
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda database_url: fake_connection
    fake_types = types.ModuleType("psycopg.types")
    fake_json = types.ModuleType("psycopg.types.json")
    fake_json.Jsonb = FakeJsonb
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types", fake_types)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", fake_json)

    memory_db.upsert_meeting_memory(
        "postgresql://example/db", meeting_dir, manifest_path
    )

    sql_calls = [" ".join(sql.lower().split()) for sql, _params in calls]
    assert any("create extension if not exists vector" in sql for sql in sql_calls)
    assert any("insert into meetings" in sql for sql in sql_calls)
    assert any("insert into meeting_artifacts" in sql for sql in sql_calls)
    assert any("insert into meeting_segments" in sql for sql in sql_calls)
    assert any("insert into meeting_facts" in sql for sql in sql_calls)

    delete_tables = [
        sql.split("delete from ", 1)[1].split(" ", 1)[0]
        for sql in sql_calls
        if sql.startswith("delete from ")
    ]
    assert delete_tables == [
        "report_context_links",
        "meeting_facts",
        "meeting_segments",
        "meeting_artifacts",
    ]

    flat_params = [
        item.value if isinstance(item, FakeJsonb) else item
        for _sql, params in calls
        if params
        for item in params
    ]
    assert "S0001" in flat_params
    assert any(
        isinstance(item, dict) and item.get("segment_id") == "S0001"
        for item in flat_params
    )
    assert fake_connection.committed


# --- Content-based (semantic) retrieval ---


def test_context_query_text_combines_title_and_segments():
    text = memory_db.context_query_text(
        "Архитектура памяти",
        ["Обсудили pgvector.", "", "Нужно хранить эмбеддинги."],
    )
    assert text == "Архитектура памяти Обсудили pgvector. Нужно хранить эмбеддинги."


def test_context_query_text_truncates_to_max_chars():
    text = memory_db.context_query_text("T", ["a" * 100], max_chars=10)
    assert len(text) == 10


def _install_fake_psycopg(monkeypatch, connection_factory):
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda *_args, **_kwargs: connection_factory()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)


class _EmptyCursor:
    rows: list = []

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, *_a, **_k):
        self.rows = []

    def fetchall(self):
        return self.rows


class _EmptyConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def cursor(self, *_a, **_k):
        return _EmptyCursor()


def test_build_report_context_pack_uses_query_text_for_semantic_search(monkeypatch):
    semantic_calls = []
    _install_fake_psycopg(monkeypatch, _EmptyConnection)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MAC_TRANSCRIBER_CONTEXT_MAX_DISTANCE", raising=False)
    monkeypatch.setattr(
        memory_db,
        "search_embedding_chunks",
        lambda *args, **kwargs: semantic_calls.append(kwargs) or [],
    )

    context_pack = memory_db.build_report_context_pack(
        "postgresql://example/db",
        meeting_id="current",
        title="audio.m4a",  # generic заголовок
        source_filename="audio.m4a",  # generic имя файла -> token query пустой
        query_text="Обсуждали стратегию ТОиР и анализ надёжности оборудования.",
    )

    # Семантический поиск идёт по содержанию текущей встречи, а не по generic-заголовку.
    assert (
        semantic_calls[0]["query"]
        == "Обсуждали стратегию ТОиР и анализ надёжности оборудования."
    )
    assert (
        context_pack["embedding_query"]
        == "Обсуждали стратегию ТОиР и анализ надёжности оборудования."
    )


def test_build_report_context_pack_filters_chunks_by_distance(monkeypatch):
    _install_fake_psycopg(monkeypatch, _EmptyConnection)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MAC_TRANSCRIBER_CONTEXT_MAX_DISTANCE", "0.4")
    monkeypatch.setattr(
        memory_db,
        "search_embedding_chunks",
        lambda *args, **kwargs: [
            {"chunk_id": "near", "text": "Релевантно.", "distance": 0.2},
            {"chunk_id": "far", "text": "Шум.", "distance": 0.9},
        ],
    )

    context_pack = memory_db.build_report_context_pack(
        "postgresql://example/db",
        meeting_id="current",
        title="Стратегия ТОиР",
        source_filename="audio.m4a",
        query_text="ТОиР и надёжность",
    )

    chunk_ids = [chunk["chunk_id"] for chunk in context_pack["embedding_chunks"]]
    assert chunk_ids == ["near"]


def test_build_report_context_pack_falls_back_to_title_when_no_query_text(monkeypatch):
    semantic_calls = []
    _install_fake_psycopg(monkeypatch, _EmptyConnection)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MAC_TRANSCRIBER_CONTEXT_MAX_DISTANCE", raising=False)
    monkeypatch.setattr(
        memory_db,
        "search_embedding_chunks",
        lambda *args, **kwargs: semantic_calls.append(kwargs) or [],
    )

    memory_db.build_report_context_pack(
        "postgresql://example/db",
        meeting_id="current",
        title="Стратегия ТОиР",
        source_filename="audio.m4a",
    )

    # Без query_text семантический поиск откатывается к заголовку (обратная совместимость).
    assert semantic_calls[0]["query"] == "Стратегия ТОиР"
