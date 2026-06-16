#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  mac_transcriber/scripts/uninstall_launchd.sh

Stops and removes the GigaAM transcriber and tunnel launchd agents for the current macOS user.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  usage >&2
  exit 64
fi

launch_agents_dir="$HOME/Library/LaunchAgents"
log_dir="$HOME/Library/Logs/slack_zoom"
domain="gui/$(id -u)"
launchctl_bin="${MAC_TRANSCRIBER_LAUNCHCTL:-/bin/launchctl}"

labels=(
  "com.slack-zoom.gigaam-transcriber"
  "com.slack-zoom.gigaam-tunnel"
)
plists=(
  "com.slack-zoom.gigaam-transcriber.plist"
  "com.slack-zoom.gigaam-tunnel.plist"
)

if [[ ! -x "$launchctl_bin" ]]; then
  echo "Required command not found or not executable: launchctl ($launchctl_bin)" >&2
  exit 69
fi

for index in "${!plists[@]}"; do
  plist_path="$launch_agents_dir/${plists[$index]}"
  service="$domain/${labels[$index]}"

  "$launchctl_bin" bootout "$service" >/dev/null 2>&1 || true
  if [[ -f "$plist_path" ]]; then
    "$launchctl_bin" bootout "$domain" "$plist_path" >/dev/null 2>&1 || true
    rm -f "$plist_path"
    echo "Removed ${labels[$index]}"
  else
    echo "No installed plist for ${labels[$index]}"
  fi
done

echo "Logs kept at $log_dir"
