# Mac Transcriber

Local GigaAM service that implements the `slack_zoom` `meeting_mvp` transcriber API.

The service lives on the Mac because GigaAM/RNNT is too heavy for the small VPS. The
VPS keeps Slack, Zoom, storage, Redis/RQ, and polling. This service receives the audio
bundle, runs `v3_e2e_rnnt`, applies the local glossary, and exposes transcript
artifacts back to `slack_zoom`.

The launchd agent includes `/opt/homebrew/bin` in `PATH` so GigaAM audio loading can
find Homebrew `ffmpeg`.

When Zoom provides separate participant tracks, those tracks remain the source of
speaker labels. When only one mixed audio file is available, the service first tries
local speaker diarization with `pyannote/speaker-diarization-community-1`, then sends
speaker turns to GigaAM for recognition. If pyannote is not installed, the Hugging Face
token is missing, or diarization fails, the service falls back to the original
single-speaker segmentation.

## Architecture

```text
Slack/Zoom -> VPS slack_zoom -> Docker host relay -> SSH reverse tunnel -> Mac transcriber
                                             ^                              |
                                             |                              v
                                     artifacts polling              GigaAM RNNT
```

Current ports:

- Mac local service: `127.0.0.1:18003`
- SSH reverse tunnel on VPS: `127.0.0.1:18013 -> Mac 127.0.0.1:18003`
- VPS Docker bridge relay: `172.22.0.1:18004 -> 127.0.0.1:18013`
- `slack_zoom` containers use `http://host.docker.internal:18004`

The transcriber is not exposed publicly.

## Manual Run

```bash
uv pip install --python .venv/bin/python -r mac_transcriber/requirements.txt
export HF_TOKEN="hf_..."
MAC_TRANSCRIBER_API_KEY=change-me \
MAC_TRANSCRIBER_ROOT=/Users/ilyaqa/Projects/GigaAM/.local/mac_transcriber \
.venv/bin/python -m uvicorn mac_transcriber.service:app --host 127.0.0.1 --port 18003
```

Health check:

```bash
curl -fsS http://127.0.0.1:18003/healthz
```

## Run With launchd

```bash
uv pip install --python .venv/bin/python -r mac_transcriber/requirements.txt
```

For single-file speaker diarization, accept the Hugging Face terms for
`pyannote/speaker-diarization-community-1`, create a read token, and keep it in
`HF_TOKEN` when installing/running the LaunchAgent. The default diarization device is
CPU; override with `MAC_TRANSCRIBER_DIARIZATION_DEVICE` if needed.

Install and start the user LaunchAgents:

```bash
export HF_TOKEN="hf_..."
export MAC_TRANSCRIBER_API_KEY="<same-shared-secret-as-vps>"
mac_transcriber/scripts/install_launchd.sh
```

You can also pass the key as the first argument:

```bash
mac_transcriber/scripts/install_launchd.sh "$MAC_TRANSCRIBER_API_KEY"
```

The installer copies the plist examples into `~/Library/LaunchAgents`, writes logs under
`~/Library/Logs/slack_zoom`, replaces the placeholder API key in the generated plist, then
bootstraps and kickstarts both agents:

- `com.slack-zoom.gigaam-transcriber`
- `com.slack-zoom.gigaam-tunnel`

Do not put real secrets in `mac_transcriber/launchd/*.example`, this README, or git. The
generated plist in `~/Library/LaunchAgents` contains the local shared secret.

Check service state:

```bash
launchctl print "gui/$(id -u)/com.slack-zoom.gigaam-transcriber"
launchctl print "gui/$(id -u)/com.slack-zoom.gigaam-tunnel"
curl -fsS http://127.0.0.1:18003/healthz
```

Check logs:

```bash
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-transcriber.err.log
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-tunnel.err.log
```

Uninstall:

```bash
mac_transcriber/scripts/uninstall_launchd.sh
```

## VPS Settings

`/root/projects/slack_zoom/.env` on the VPS should contain:

```env
TRANSCRIPTION_SERVICE_ADAPTER=meeting_mvp
TRANSCRIPTION_SERVICE_URL=http://host.docker.internal:18004
TRANSCRIPTION_SERVICE_API_KEY=<same-key>
```

After changing `.env`, recreate the containers:

```bash
cd /root/projects/slack_zoom
docker compose up -d --force-recreate app worker
```

The VPS relay is a systemd service:

```bash
systemctl status slack-zoom-mac-transcriber-relay.service
```

The relay binds only to the Docker bridge (`172.22.0.1:18004`) and forwards to the
loopback SSH reverse tunnel (`127.0.0.1:18013`).

## Endpoints

- `GET /healthz`
- `POST /meetings`
- `GET /meetings/{id}/status`
- `GET /meetings/{id}/artifacts/{kind}`

## External Service Contract

External systems should treat this service as an async transcription/report job API.
The service stores generated files locally and exposes them through artifact endpoints.

Authentication, when `MAC_TRANSCRIBER_API_KEY` is set:

```http
Authorization: Bearer <api-key>
```

Recommended flow:

1. Create a meeting job with `POST /meetings`.
2. Poll `GET /meetings/{id}/status` until `status == "completed"` or `status == "failed"`.
3. Download `summary_json`.
4. Send `summary_json.slack.text` as the Slack message body.
5. Upload only files listed in `summary_json.slack.files` by downloading each `kind` from
   `GET /meetings/{id}/artifacts/{kind}`.

Create a job:

```bash
curl -fsS -X POST "$TRANSCRIPTION_SERVICE_URL/meetings" \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  -F "file=@/absolute/path/to/recording.m4a" \
  -F "title=Meeting title" \
  -F "language_code=ru" \
  -F "processing_mode=simple"
```

`title` is optional but recommended. If it is omitted, the service derives a display
title from the uploaded filename and strips common Zoom date/time suffixes instead of
using the internal UUID.

Response:

```json
{"id":"<meeting-id>","status":"uploaded"}
```

Poll status:

```bash
curl -fsS \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  "$TRANSCRIPTION_SERVICE_URL/meetings/<meeting-id>/status"
```

Important status fields:

- `status`: `uploaded`, `processing`, `completed`, or `failed`
- `phase`: current pipeline phase
- `progress`: `0..1`
- `artifacts`: available artifact kinds
- `report_health_status`: `ok`, `degraded`, or `failed`
- `report_alerts`: report generation warnings that should be visible to operators

Download summary for Slack:

```bash
curl -fsS \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  "$TRANSCRIPTION_SERVICE_URL/meetings/<meeting-id>/artifacts/summary_json"
```

`summary_json.slack` is the Slack handoff contract:

```json
{
  "slack": {
    "text": "*Meeting title*\nDate · duration\n...",
    "files": [
      {"kind":"report_pdf","filename":"report.pdf","mime_type":"application/pdf"},
      {"kind":"report_md","filename":"report.md","mime_type":"text/markdown"},
      {"kind":"transcript_md","filename":"transcript.md","mime_type":"text/markdown"}
    ]
  }
}
```

Do not upload JSON files to Slack. `coverage_json`, `report_health`, and `report_json`
are machine/debug artifacts only.

Download each Slack attachment:

```bash
curl -fsS \
  -H "Authorization: Bearer $TRANSCRIPTION_SERVICE_API_KEY" \
  -o report.pdf \
  "$TRANSCRIPTION_SERVICE_URL/meetings/<meeting-id>/artifacts/report_pdf"
```

For Zoom participant tracks, send `zoom_participant_files` as repeated multipart fields
and optionally send `participants` as comma-separated display names.

## Rich Reports

The transcriber writes a cited protocol report next to the raw transcript:

- `report.json`: structured source of truth for report sections and segment citations
- `report.md`: human-readable Markdown protocol report
- `report.html`: bundled Jinja protocol template used for PDF rendering
- `report.typ`: legacy Typst source kept as a diagnostic/export artifact
- `report.pdf`: generated from `report.html` with WeasyPrint when PDF generation is enabled

The full transcript stays in `transcript.md`. The PDF intentionally omits the full
transcript and coverage audit; it includes protocol sections, citations, and
artifact/source notes. Coverage details stay in `coverage.json`.

Manual run with a local deterministic report:

```bash
.venv/bin/python mac_transcriber/scripts/transcribe_md.py "/absolute/path/to/recording.webm" \
  --report
```

Manual run with OpenAI report enrichment and PDF compilation:

```bash
.venv/bin/python mac_transcriber/scripts/transcribe_md.py "/absolute/path/to/recording.webm" \
  --ai-report \
  --pdf
```

Service mode enables OpenAI report enrichment and PDF generation by default in the
provided local/launchd starters. Provide `OPENAI_API_KEY` or `.env.local`, and install
Playwright browsers. If AI or PDF generation fails, the job still writes fallback
artifacts and marks the report as `degraded` in `report_health.json`.

```env
MAC_TRANSCRIBER_REPORT_MODE=ai
MAC_TRANSCRIBER_REPORT_MODEL=gpt-5.5
MAC_TRANSCRIBER_REPORT_PDF=1
MAC_TRANSCRIBER_AI_CHUNK_SIZE=80
MAC_TRANSCRIBER_AI_SYNTHESIS_CHUNK_LIMIT=24
MAC_TRANSCRIBER_AI_SYNTHESIS_BATCH_SIZE=4
# Optional: retry thin current-report extraction with a stronger model before memory enrichment.
# MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=gpt-5.5
# OpenRouter can be used as a primary model by prefixing the model id:
# MAC_TRANSCRIBER_REPORT_MODEL=openrouter:google/gemini-3.1-pro-preview
# MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=openrouter:openai/gpt-5.5
# Optional OpenRouter reserve when the OpenAI report request fails:
# OPENROUTER_API_KEY=sk-or-v1-...
# MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL=openai/gpt-4o-mini
# MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS=12000
# Optional when playwright is not on PATH:
# MAC_TRANSCRIBER_PLAYWRIGHT=/opt/homebrew/bin/playwright
```

Supported artifacts:

- `manifest_json`
- `context_pack_json`
- `transcript_md`
- `transcript_json`
- `summary_json`
- `report_md`
- `report_json`
- `report_html`
- `report_health`
- `report_typ`
- `report_pdf`
- `segments_tsv`
- `speaker_track_map`

Slack message text is stored in `summary_json.slack.text`. Upload only the files
listed in `summary_json.slack.files`.

## Local Meeting Memory

The Mac transcriber is the owner of meeting memory. Slack/Zoom on the VPS remains an
orchestrator: it sends recordings, polls status, downloads final artifacts, and posts
Slack messages. It does not need direct database access.

Every completed job writes `artifacts/manifest.json`. The manifest records meeting
metadata, result summary, and an inventory of input/generated files with relative path,
size, sha256, and timestamps. Keep it with the meeting artifacts; it is the audit index
for later reprocessing.

Optional Postgres memory is enabled only when a local database URL is configured:

```env
MAC_TRANSCRIBER_DATABASE_URL=postgresql://gigaam:gigaam@127.0.0.1:5432/gigaam_memory
```

`MAC_TRANSCRIBER_POSTGRES_DSN` is accepted as a fallback variable. When configured, the
service fetches a small `context_pack.json` from prior meetings before report
generation, then upserts the current meeting, artifacts, transcript segments, report
facts, and citation links into Postgres after artifact generation. The schema reserves
tables for future `pgvector` embedding chunks and report context links. If Postgres is
unavailable, the job still completes; `summary.json` gets `context_pack_error` or
`memory_sync_error`, and the final manifest is rewritten so recorded hashes match the
files on disk.

The database should be local to the Mac for this architecture. Back up both Postgres
and `MAC_TRANSCRIBER_ROOT`; Postgres is the queryable memory, while the artifact folder
is the durable file archive.

OpenAI embeddings can be enabled with the existing `OPENAI_API_KEY`. The default model
is `text-embedding-3-small`, stored in the reserved `embedding_chunks.embedding
vector(1536)` column:

```bash
.venv/bin/python mac_transcriber/scripts/backfill_embeddings.py --env-file .env.local
```

Useful options:

```bash
.venv/bin/python mac_transcriber/scripts/backfill_embeddings.py \
  --env-file .env.local \
  --meeting-id <meeting-id> \
  --model text-embedding-3-small \
  --batch-size 64
```

The embedding backfill rebuilds chunks for selected meetings from transcript segments
and extracted report facts. New memory syncs also attempt a best-effort embedding sync
when both `MAC_TRANSCRIBER_DATABASE_URL` and `OPENAI_API_KEY` are configured.

### Semantic prior-context retrieval

Prior-meeting context is retrieved by the **meaning of the current meeting**, not just
its title: the semantic (embedding) search query is built from the current transcript,
so context is still found when the title/source filename is generic (e.g. `audio.m4a`).
Token search and the recency fallback continue to use the title for cheap exact matches.

Optionally drop weakly-related embedding hits by setting a maximum cosine distance
(`0`–`2`; lower is stricter). Unset means no threshold. A good starting point:

```env
# Discard prior-context chunks whose cosine distance to the current meeting exceeds this.
MAC_TRANSCRIBER_CONTEXT_MAX_DISTANCE=0.6
```

### Reasoning effort

For gpt-5 / o-series report models the reasoning depth of the report-shaping passes
(direct report, chunked synthesis, memory enrichment) can be raised for higher quality
at the cost of latency/tokens. Levels: `minimal` < `low` < `medium` < `high`. Defaults
are `medium` (report/synthesis) and `low` (enrichment); per-chunk notes stay `low`.

```env
# Override reasoning depth of the report-shaping passes (omit to keep defaults).
MAC_TRANSCRIBER_REASONING_EFFORT=high
```

### Out-of-funds handling (AI quota)

If the AI provider rejects a request for billing/quota reasons (HTTP 429
`insufficient_quota` / "check your plan and billing"), the report is **not** allowed to
fall back to the raw local keyword dump. Instead the meeting is parked with status
`blocked_on_quota`: the transcript is kept, no (garbage) report is written, and the job
is not marked `completed` or `failed`. After you top up the balance, drain the backlog
**sequentially** (report-only — the saved transcript is reused, ASR is NOT re-run) — it
stops at the first meeting that hits the quota again:

```bash
.venv/bin/python mac_transcriber/scripts/reprocess_blocked.py            # drain backlog
.venv/bin/python mac_transcriber/scripts/reprocess_blocked.py --dry-run  # just list the queue
```

## Verification

Local Mac:

```bash
curl -fsS http://127.0.0.1:18003/healthz
launchctl list | rg 'com\.slack-zoom\.gigaam'
```

VPS to Mac tunnel:

```bash
ssh root@193.233.87.211 'curl -fsS http://127.0.0.1:18013/healthz'
```

`slack_zoom-worker` to Mac through the relay:

```bash
ssh root@193.233.87.211 'cd /root/projects/slack_zoom && docker compose exec -T worker python - <<'"'"'PY'"'"'
import os
import urllib.error
import urllib.request

base = os.environ["TRANSCRIPTION_SERVICE_URL"]
key = os.environ["TRANSCRIPTION_SERVICE_API_KEY"]
print(urllib.request.urlopen(base + "/healthz", timeout=5).read().decode())
req = urllib.request.Request(
    base + "/meetings/no-such-meeting/status",
    headers={"Authorization": "Bearer " + key},
)
try:
    urllib.request.urlopen(req, timeout=5)
except urllib.error.HTTPError as exc:
    print("AUTH_STATUS_CHECK=" + str(exc.code))
PY'
```

Expected result: `{"status":"ok"}` and `AUTH_STATUS_CHECK=404`.

## Security Notes

- Keep `MAC_TRANSCRIBER_API_KEY` out of git and chat logs.
- Do not bind the Mac service or VPS relay to public `0.0.0.0`.
- `launchctl print` can reveal environment variables, including the API key.
- Raw audio and generated artifacts are stored under `MAC_TRANSCRIBER_ROOT`; keep that
  path out of git.
- The current VPS SSH configuration is key-only; password login is disabled.

## Troubleshooting

Check Mac logs:

```bash
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-transcriber.err.log
tail -n 100 ~/Library/Logs/slack_zoom/gigaam-tunnel.err.log
```

Check VPS relay and ports:

```bash
ssh root@193.233.87.211 'systemctl status slack-zoom-mac-transcriber-relay.service --no-pager'
ssh root@193.233.87.211 'ss -ltnp | grep -E ":18013|:18004"'
```

If the VPS reboots, `slack_zoom` containers should restart automatically with
`restart=unless-stopped`, and the relay is enabled through systemd.
