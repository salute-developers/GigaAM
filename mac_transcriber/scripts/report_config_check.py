#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


CONFIG_KEYS = [
    "MAC_TRANSCRIBER_REPORT_MODE",
    "MAC_TRANSCRIBER_REPORT_MODEL",
    "MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL",
    "MAC_TRANSCRIBER_REPORT_CRITIC_MODEL",
    "MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL",
    "MAC_TRANSCRIBER_OPENROUTER_MAX_TOKENS",
]
SECRET_KEYS = ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]


def main() -> int:
    env = load_env(REPO_ROOT / ".env.local")
    print("Report AI config check")
    print(f"env_file: {REPO_ROOT / '.env.local'}")
    for key in CONFIG_KEYS:
        value = env.get(key, "").strip()
        print(f"{key}: {value or 'missing'}")
    for key in SECRET_KEYS:
        value = env.get(key, "").strip()
        print(f"{key}: {'present' if value else 'missing'}")
    blocking = print_recommendations(env)
    # Ненулевой код выхода нужен, чтобы CI/скрипты ловили жёсткую misconfig.
    return 1 if blocking else 0


def load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def print_recommendations(env: dict[str, str]) -> int:
    """Печатает рекомендации и возвращает число блокирующих проблем."""
    issues: list[str] = []
    blocking = 0
    if not env.get("OPENAI_API_KEY") and not env.get("OPENROUTER_API_KEY"):
        issues.append("configure OPENAI_API_KEY or OPENROUTER_API_KEY")
        blocking += 1
    model = env.get("MAC_TRANSCRIBER_REPORT_MODEL", "").strip()
    if model:
        # Сверяем выбранную модель с ключом нужного провайдера: openrouter:* требует
        # OPENROUTER_API_KEY, обычная модель — OPENAI_API_KEY.
        if model.lower().startswith("openrouter:"):
            if not env.get("OPENROUTER_API_KEY"):
                issues.append(
                    "MAC_TRANSCRIBER_REPORT_MODEL uses the 'openrouter:' prefix "
                    "but OPENROUTER_API_KEY is missing; set OPENROUTER_API_KEY"
                )
                blocking += 1
        elif not env.get("OPENAI_API_KEY"):
            issues.append(
                "MAC_TRANSCRIBER_REPORT_MODEL is a direct OpenAI model "
                "but OPENAI_API_KEY is missing; set OPENAI_API_KEY"
            )
            blocking += 1
    else:
        issues.append("set MAC_TRANSCRIBER_REPORT_MODEL for predictable reports")
    if not env.get("MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL"):
        issues.append(
            "optional: set MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=openrouter:openai/gpt-5.5 "
            "to retry thin current extraction via OpenRouter"
        )
    # MAC_TRANSCRIBER_REPORT_CRITIC_MODEL печатается в дампе (CONFIG_KEYS), но не
    # рекомендуется отдельно: критик экспериментальный и выключен по умолчанию, нагать
    # о нём преждевременно (включать осознанно после A/B).
    if not issues:
        print("recommendations: ok")
        return 0
    print("recommendations:")
    for issue in issues:
        print(f"- {issue}")
    return blocking


if __name__ == "__main__":
    raise SystemExit(main())
