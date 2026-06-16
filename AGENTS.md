# AGENTS.md

Guidelines for agents working in this repository.

## Ad Hoc Markdown Transcripts

When the user asks to transcribe an audio/video recording to Markdown, for example
"transcript MD", "make an MD transcript", "transcribe this recording", or the same
request in Russian, do the work through the local GigaAM pipeline by default.

Use GigaAM `v3_e2e_rnnt` from the project virtualenv. Do not start with Whisper,
cloud ASR, or generic speech-to-text tools unless the user explicitly asks for them
or GigaAM fails after a real attempt.

Preferred command:

```bash
.venv/bin/python mac_transcriber/scripts/transcribe_md.py "/absolute/path/to/recording.webm"
```

Useful options:

```bash
.venv/bin/python mac_transcriber/scripts/transcribe_md.py "/absolute/path/to/recording.webm" \
  --title "Meeting title" \
  --output transcripts/meeting_transcript.md
```

Operational defaults:

- Model: `v3_e2e_rnnt`
- Cache: `/tmp/gigaam-cache`
- Device: `cpu`
- Batch size: `4`
- Output directory: `transcripts/`
- Working artifacts: `.local/manual_transcripts/`
- Diarization: off by default for one mixed recording; enable only when the user asks
  for speaker separation or participant tracks are available.

Expected behavior:

- Produce a `.md` file, not just a plan or command suggestion.
- Include source filename, model, duration when available, and timestamped transcript
  blocks.
- Tell the user the absolute Markdown path when finished.
- Keep generated transcripts, raw audio links/copies, and `.local/` artifacts out of
  git unless the user explicitly asks to version them.

If the model cache is missing and the environment cannot download dependencies because
network access is restricted, ask for permission to download or report the blocker
clearly.
