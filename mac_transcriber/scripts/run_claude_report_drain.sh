#!/bin/bash
# Обёртка launchd: слив очереди отчётов через claude (drain_reports_via_claude.py).
# Логи: ~/Library/Logs/slack_zoom/gigaam-claude-reports.log
# claude берёт креды из ~/.claude (нужен залогиненный Claude Code), ffmpeg/typst — из PATH.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"
LOG_DIR="$HOME/Library/Logs/slack_zoom"
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') drain start ==="
  echo "claude: $(command -v claude || echo MISSING) $(claude --version 2>/dev/null || echo '(no --version)')"
} >>"$LOG_DIR/gigaam-claude-reports.log"
exec "$REPO/.venv/bin/python" mac_transcriber/scripts/drain_reports_via_claude.py "$@" \
  >>"$LOG_DIR/gigaam-claude-reports.log" 2>&1
