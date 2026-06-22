# Meeting Memory Design

## Goal

Build a local, Postgres-backed meeting memory for `mac_transcriber` while keeping the Mac service the owner of raw recordings, generated transcripts, reports, and derived meeting facts.

## Architecture

The VPS/Slack/Zoom side remains an orchestrator. The Mac transcriber owns processing, archival storage, and memory indexing. The existing job API remains compatible: jobs still write artifacts under the local meeting directory and expose the same status and artifact endpoints.

The first version adds two local layers:

- A file archive/manifest layer that records every input and generated artifact with relative path, size, sha256, and timestamps.
- A Postgres memory layer, enabled by `MAC_TRANSCRIBER_DATABASE_URL`, that persists meetings, status, transcript segments, facts from report artifacts, and artifact references.

If Postgres is not configured or unavailable, transcription must still complete. Memory persistence is best-effort and reports a warning in local logs/status rather than blocking core ASR/report output.

## Data Model

Postgres owns normalized queryable state:

- `meetings`: id, title, source filename, language, status, phase, progress, duration, metadata, result, error, local paths, timestamps.
- `meeting_artifacts`: meeting id, kind, relative path, absolute path, size, sha256, mime type, producer, created timestamp.
- `meeting_segments`: meeting id, stable segment id, start/end seconds, speaker, track, text.
- `meeting_facts`: meeting id, fact type, title, text, owner, due, citations, payload.
- `report_context_links`: reserved for future retrieval links between a generated report and prior meeting segments/facts.
- `embedding_chunks`: reserved for future pgvector-backed semantic search.

The filesystem remains the audit archive. Postgres stores paths and hashes, not large raw binary content.

## Data Flow

1. `POST /meetings` writes upload inputs and metadata as it does today.
2. The job writes status transitions through the existing `_write_status` choke point.
3. `transcribe_meeting` produces transcript/report artifacts as today.
4. A finalizer builds `manifest.json` from `input/`, `artifacts/`, and `status.json`.
5. When `MAC_TRANSCRIBER_DATABASE_URL` is set, the finalizer upserts the meeting, segments, facts, and artifact inventory into Postgres.

## Error Handling

Memory/archive failures must not fail transcription unless file artifact generation itself fails. Postgres failures are captured as non-fatal memory warnings and can be retried later by re-running ingestion.

## Testing

Regular tests use pure functions and fake repositories, so CI/local test runs do not need Postgres. A future opt-in contract test can run against `MAC_TRANSCRIBER_TEST_DATABASE_URL`.
