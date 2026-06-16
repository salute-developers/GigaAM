# Meeting Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local file-archive manifests and optional Postgres-backed meeting memory to `mac_transcriber`.

**Architecture:** Keep ASR/report generation unchanged and add a narrow finalization layer after artifacts are written. Postgres persistence is enabled only when `MAC_TRANSCRIBER_DATABASE_URL` is set and must be best-effort.

**Tech Stack:** Python, pytest, FastAPI service hooks, PostgreSQL with `psycopg[binary]`, future pgvector-compatible schema.

---

### Task 1: Archive Manifest

**Files:**
- Create: `mac_transcriber/archive.py`
- Create: `mac_transcriber/tests/test_archive.py`

- [ ] **Step 1: Write failing tests**

Create tests for `sha256_file`, `build_file_inventory`, and `write_meeting_manifest`. The tests should create `input/audio.m4a`, `artifacts/transcript.md`, and `status.json` under a temporary meeting directory, then assert that `manifest.json` includes relative paths, sizes, hashes, meeting id, title, source filename, and generated timestamp.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest mac_transcriber/tests/test_archive.py -q
```

Expected: import failure for missing `mac_transcriber.archive`.

- [ ] **Step 3: Implement archive module**

Implement:

```python
sha256_file(path: Path) -> str
build_file_inventory(meeting_dir: Path) -> list[dict[str, object]]
write_meeting_manifest(meeting_dir: Path, metadata: dict, result: dict | None = None) -> Path
```

The inventory must include files from `input/`, `artifacts/`, and `status.json`, skip `manifest.json` itself, and use POSIX relative paths.

- [ ] **Step 4: Run focused test**

Run:

```bash
.venv/bin/python -m pytest mac_transcriber/tests/test_archive.py -q
```

Expected: pass.

### Task 2: Postgres Memory Repository

**Files:**
- Create: `mac_transcriber/memory_db.py`
- Create: `mac_transcriber/tests/test_memory_db.py`
- Modify: `mac_transcriber/requirements.txt`

- [ ] **Step 1: Write failing tests**

Create tests that use a fake connection/cursor to verify:

- `database_url_from_env` reads `MAC_TRANSCRIBER_DATABASE_URL`.
- `memory_enabled` is false when no URL is set.
- `build_schema_sql` contains tables `meetings`, `meeting_artifacts`, `meeting_segments`, `meeting_facts`, `report_context_links`, `embedding_chunks` and extension setup for `vector`, `pg_trgm`, and `unaccent`.
- `extract_facts_from_report` reads `decisions`, `tasks`, `questions`, and `risks` from a protocol-shaped `report.json`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest mac_transcriber/tests/test_memory_db.py -q
```

Expected: import failure for missing `mac_transcriber.memory_db`.

- [ ] **Step 3: Implement memory module**

Implement a repository with:

```python
database_url_from_env(env: Mapping[str, str] = os.environ) -> str
memory_enabled(env: Mapping[str, str] = os.environ) -> bool
build_schema_sql() -> str
extract_facts_from_report(report_json_path: Path) -> list[dict[str, object]]
upsert_meeting_memory(database_url: str, meeting_dir: Path, manifest_path: Path) -> None
sync_meeting_memory(meeting_dir: Path, manifest_path: Path) -> str | None
```

`sync_meeting_memory` returns `None` on success/no-op and returns an error message on non-fatal failure. Import `psycopg` lazily so tests and no-DB installs still import the package.

- [ ] **Step 4: Run focused test**

Run:

```bash
.venv/bin/python -m pytest mac_transcriber/tests/test_memory_db.py -q
```

Expected: pass.

### Task 3: Pipeline Integration

**Files:**
- Modify: `mac_transcriber/asr.py`
- Modify: `mac_transcriber/service.py`
- Modify: `mac_transcriber/tests/test_service.py`
- Modify: `mac_transcriber/tests/test_archive.py`

- [ ] **Step 1: Write failing integration tests**

Add tests that monkeypatch archive/memory helpers and assert:

- `write_artifacts` writes `manifest.json` after standard artifacts.
- `transcript.json` rows include stable `segment_id` values.
- `manifest_json` is exposed by the service artifact registry.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest mac_transcriber/tests/test_archive.py mac_transcriber/tests/test_service.py -q
```

Expected: failures for missing manifest integration and missing `segment_id`.

- [ ] **Step 3: Wire finalization**

After standard artifacts are written in `write_artifacts`, call `write_meeting_manifest` and `sync_meeting_memory`. Add non-fatal memory warnings to `summary.json`. Add `manifest_json` to `ARTIFACTS`. Add `segment_id` to `transcript.json`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest mac_transcriber/tests/test_archive.py mac_transcriber/tests/test_memory_db.py mac_transcriber/tests/test_service.py -q
```

Expected: pass.

### Task 4: Verification

**Files:**
- Modify: `mac_transcriber/README.md`

- [ ] **Step 1: Document configuration**

Document `MAC_TRANSCRIBER_DATABASE_URL`, `manifest.json`, and the local-only memory responsibility split.

- [ ] **Step 2: Run full mac transcriber tests**

Run:

```bash
.venv/bin/python -m pytest mac_transcriber/tests -q
```

Expected: all tests pass.

- [ ] **Step 3: Review git status**

Run:

```bash
git status --short
```

Expected: only intentional files changed or created.
