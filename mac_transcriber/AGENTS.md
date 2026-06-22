# AGENTS.md

Guidelines for agents working on the local Mac transcriber.

## Intent

`mac_transcriber` is a local FastAPI service that implements the `slack_zoom`
`meeting_mvp` transcription contract. It runs GigaAM `v3_e2e_rnnt` on the Mac and
returns artifacts to the VPS orchestrator.

The VPS remains the Slack/Zoom/RQ/storage orchestrator. Do not move ASR, model loading,
or long audio processing back to the VPS.

## Boundaries

- Own files under `mac_transcriber/` unless the user explicitly asks for broader GigaAM
  changes.
- Keep the service compatible with:
  - `POST /meetings`
  - `GET /meetings/{id}/status`
  - `GET /meetings/{id}/artifacts/{kind}`
- Keep the Mac service bound to `127.0.0.1`.
- Keep the VPS-facing path private through SSH tunnel and Docker bridge relay.
- Keep Homebrew `ffmpeg` visible to launchd. The transcriber LaunchAgent should include
  `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin` in `PATH`.

## Security

- Never print or commit `MAC_TRANSCRIBER_API_KEY`.
- Do not paste generated LaunchAgent plists into chat because they contain secrets.
- Do not bind the transcriber or relay to public `0.0.0.0`.
- Keep `.local/`, `transcripts/`, raw audio, and generated artifacts out of git.
- Be careful with `launchctl print`; it can show environment variables.

## Verification

Use these checks after edits:

```bash
bash -n mac_transcriber/scripts/install_launchd.sh
bash -n mac_transcriber/scripts/uninstall_launchd.sh
.venv/bin/python -m compileall mac_transcriber
curl -fsS http://127.0.0.1:18003/healthz
```

For end-to-end VPS checks:

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

Expected: local health is `200`, worker health prints `{"status":"ok"}`, and the
authenticated fake meeting check returns `AUTH_STATUS_CHECK=404`.

## Operational Notes

- LaunchAgents:
  - `com.slack-zoom.gigaam-transcriber`
  - `com.slack-zoom.gigaam-tunnel`
- VPS systemd relay:
  - `slack-zoom-mac-transcriber-relay.service`
- VPS `slack_zoom` transcription URL:
  - `http://host.docker.internal:18004`

If the transcriber fails after a reboot, check in this order:

1. `curl -fsS http://127.0.0.1:18003/healthz` on the Mac.
2. `ssh root@193.233.87.211 'curl -fsS http://127.0.0.1:18013/healthz'`.
3. `systemctl status slack-zoom-mac-transcriber-relay.service` on the VPS.
4. Worker container access to `http://host.docker.internal:18004/healthz`.
5. `tail -n 100 ~/Library/Logs/slack_zoom/gigaam-transcriber.err.log`; if the error is
   `No such file or directory: 'ffmpeg'`, verify the LaunchAgent `PATH`.

## Ad Hoc Markdown Transcripts

For direct user requests to transcribe an audio/video file into Markdown, use the
repository helper and GigaAM `v3_e2e_rnnt`:

```bash
.venv/bin/python mac_transcriber/scripts/transcribe_md.py "/absolute/path/to/recording.webm"
```

This writes the final `.md` under `transcripts/` and keeps intermediate JSON/TSV
artifacts under `.local/manual_transcripts/`. Do not use Whisper for these requests
unless the user explicitly asks for it or GigaAM fails after a real attempt.
