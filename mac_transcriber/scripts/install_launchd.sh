#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  MAC_TRANSCRIBER_API_KEY=<shared-secret> mac_transcriber/scripts/install_launchd.sh
  HF_TOKEN=<hugging-face-token> MAC_TRANSCRIBER_API_KEY=<shared-secret> mac_transcriber/scripts/install_launchd.sh
  mac_transcriber/scripts/install_launchd.sh <shared-secret>

Installs the GigaAM transcriber and tunnel launchd agents for the current macOS user.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mac_transcriber_dir="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$mac_transcriber_dir/.." && pwd)"
launchd_source_dir="$mac_transcriber_dir/launchd"
launch_agents_dir="$HOME/Library/LaunchAgents"
log_dir="$HOME/Library/Logs/slack_zoom"
domain="gui/$(id -u)"
launchctl_bin="${MAC_TRANSCRIBER_LAUNCHCTL:-/bin/launchctl}"
plutil_bin="${MAC_TRANSCRIBER_PLUTIL:-/usr/bin/plutil}"
plistbuddy_bin="${MAC_TRANSCRIBER_PLISTBUDDY:-/usr/libexec/PlistBuddy}"

labels=(
  "com.slack-zoom.gigaam-transcriber"
  "com.slack-zoom.gigaam-tunnel"
)
plists=(
  "com.slack-zoom.gigaam-transcriber.plist"
  "com.slack-zoom.gigaam-tunnel.plist"
)

require_command() {
  local command_path="$1"
  local name="$2"

  if [[ ! -x "$command_path" ]]; then
    echo "Required command not found or not executable: $name ($command_path)" >&2
    exit 69
  fi
}

write_plist() {
  local plist_name="$1"
  local label="$2"
  local source_path="$launchd_source_dir/$plist_name.example"
  local target_path="$launch_agents_dir/$plist_name"

  if [[ ! -f "$source_path" ]]; then
    echo "Missing launchd example: $source_path" >&2
    exit 66
  fi

  cp "$source_path" "$target_path"
  chmod 600 "$target_path"

  case "$label" in
    "com.slack-zoom.gigaam-transcriber")
      "$plutil_bin" -replace WorkingDirectory -string "$repo_root" "$target_path"
      "$plistbuddy_bin" -c "Delete :ProgramArguments" "$target_path" >/dev/null 2>&1 || true
      "$plistbuddy_bin" -c "Add :ProgramArguments array" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:0 string $repo_root/.venv/bin/python" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:1 string -m" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:2 string uvicorn" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:3 string mac_transcriber.service:app" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:4 string --host" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:5 string 127.0.0.1" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:6 string --port" "$target_path"
      "$plistbuddy_bin" -c "Add :ProgramArguments:7 string 18003" "$target_path"
      "$plutil_bin" -replace EnvironmentVariables.MAC_TRANSCRIBER_API_KEY -string "$shared_secret" "$target_path"
      "$plutil_bin" -replace EnvironmentVariables.MAC_TRANSCRIBER_ROOT -string "$repo_root/.local/mac_transcriber" "$target_path"
      "$plutil_bin" -replace EnvironmentVariables.MAC_TRANSCRIBER_REPORT_MODE -string "${MAC_TRANSCRIBER_REPORT_MODE:-ai}" "$target_path"
      "$plutil_bin" -replace EnvironmentVariables.MAC_TRANSCRIBER_REPORT_MODEL -string "${MAC_TRANSCRIBER_REPORT_MODEL:-gpt-5.5}" "$target_path"
      "$plutil_bin" -replace EnvironmentVariables.MAC_TRANSCRIBER_REPORT_PDF -string "${MAC_TRANSCRIBER_REPORT_PDF:-1}" "$target_path"
      if [[ -n "${HF_TOKEN:-}" ]]; then
        "$plutil_bin" -replace EnvironmentVariables.HF_TOKEN -string "$HF_TOKEN" "$target_path"
      fi
      if [[ -n "${MAC_TRANSCRIBER_DIARIZATION_DEVICE:-}" ]]; then
        "$plutil_bin" -replace EnvironmentVariables.MAC_TRANSCRIBER_DIARIZATION_DEVICE -string "$MAC_TRANSCRIBER_DIARIZATION_DEVICE" "$target_path"
      fi
      "$plutil_bin" -replace EnvironmentVariables.MPLCONFIGDIR -string "/tmp/slack-zoom-matplotlib" "$target_path"
      "$plutil_bin" -replace EnvironmentVariables.PATH -string "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" "$target_path"
      "$plutil_bin" -replace StandardOutPath -string "$log_dir/gigaam-transcriber.out.log" "$target_path"
      "$plutil_bin" -replace StandardErrorPath -string "$log_dir/gigaam-transcriber.err.log" "$target_path"
      ;;
    "com.slack-zoom.gigaam-tunnel")
      "$plistbuddy_bin" -c "Delete :ProgramArguments" "$target_path" >/dev/null 2>&1 || true
      "$plistbuddy_bin" -c "Add :ProgramArguments array" "$target_path"
      local tunnel_identity_file="${MAC_TRANSCRIBER_TUNNEL_IDENTITY_FILE:-$HOME/.ssh/valamis_vps_ed25519}"
      local tunnel_remote_forward="${MAC_TRANSCRIBER_TUNNEL_REMOTE_FORWARD:-127.0.0.1:18013:127.0.0.1:18003}"
      local tunnel_target="${MAC_TRANSCRIBER_TUNNEL_TARGET:-root@193.233.87.211}"
      local tunnel_args=(
        "/usr/bin/ssh"
        "-N"
        "-o" "BatchMode=yes"
        "-o" "ExitOnForwardFailure=yes"
        "-o" "ServerAliveInterval=30"
        "-o" "ServerAliveCountMax=3"
        "-o" "StrictHostKeyChecking=accept-new"
      )
      if [[ -f "$tunnel_identity_file" ]]; then
        tunnel_args+=("-i" "$tunnel_identity_file" "-o" "IdentitiesOnly=yes")
      fi
      tunnel_args+=("-R" "$tunnel_remote_forward" "$tunnel_target")
      for arg_index in "${!tunnel_args[@]}"; do
        "$plistbuddy_bin" -c "Add :ProgramArguments:$arg_index string ${tunnel_args[$arg_index]}" "$target_path"
      done
      "$plutil_bin" -replace StandardOutPath -string "$log_dir/gigaam-tunnel.out.log" "$target_path"
      "$plutil_bin" -replace StandardErrorPath -string "$log_dir/gigaam-tunnel.err.log" "$target_path"
      ;;
    *)
      echo "Unknown launchd label: $label" >&2
      exit 70
      ;;
  esac

  "$plutil_bin" -lint "$target_path" >/dev/null
}

reload_agent() {
  local plist_name="$1"
  local label="$2"
  local target_path="$launch_agents_dir/$plist_name"
  local service="$domain/$label"

  "$launchctl_bin" bootout "$service" >/dev/null 2>&1 || true
  "$launchctl_bin" bootout "$domain" "$target_path" >/dev/null 2>&1 || true
  "$launchctl_bin" bootstrap "$domain" "$target_path"
  "$launchctl_bin" kickstart -k "$service"
  echo "Installed and started $label"
}

require_command "$launchctl_bin" launchctl
require_command "$plutil_bin" plutil
require_command "$plistbuddy_bin" PlistBuddy

shared_secret="${1:-${MAC_TRANSCRIBER_API_KEY:-}}"
existing_transcriber_plist="$launch_agents_dir/com.slack-zoom.gigaam-transcriber.plist"
if [[ -z "$shared_secret" && -f "$existing_transcriber_plist" ]]; then
  shared_secret="$("$plistbuddy_bin" -c "Print :EnvironmentVariables:MAC_TRANSCRIBER_API_KEY" "$existing_transcriber_plist" 2>/dev/null || true)"
fi
if [[ -z "$shared_secret" || "$shared_secret" == "REPLACE_WITH_SHARED_SECRET" ]]; then
  echo "MAC_TRANSCRIBER_API_KEY, a shared-secret argument, or an existing installed secret is required." >&2
  usage >&2
  exit 64
fi

mkdir -p "$launch_agents_dir" "$log_dir"

for index in "${!plists[@]}"; do
  write_plist "${plists[$index]}" "${labels[$index]}"
done

for index in "${!plists[@]}"; do
  reload_agent "${plists[$index]}" "${labels[$index]}"
done

echo "LaunchAgents: $launch_agents_dir"
echo "Logs: $log_dir"
