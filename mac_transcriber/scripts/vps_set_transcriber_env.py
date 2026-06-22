#!/usr/bin/env python3
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: vps_set_transcriber_env.py ENV_PATH SECRET_PATH TRANSCRIBER_URL",
            file=sys.stderr,
        )
        return 2

    env_path = Path(sys.argv[1])
    secret_path = Path(sys.argv[2])
    transcriber_url = sys.argv[3].rstrip("/")
    secret = secret_path.read_text(encoding="utf-8").strip()
    if not secret:
        raise SystemExit("secret file is empty")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = env_path.with_name(f"{env_path.name}.bak-mac-transcriber-{stamp}")
    backup.write_bytes(env_path.read_bytes())

    updates = {
        "TRANSCRIPTION_SERVICE_ADAPTER": "meeting_mvp",
        "TRANSCRIPTION_SERVICE_URL": transcriber_url,
        "TRANSCRIPTION_SERVICE_API_KEY": secret,
    }

    lines = env_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0]
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    secret_path.unlink(missing_ok=True)
    print("updated transcription env; backup created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
