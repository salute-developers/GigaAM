import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_config_check.py"
SPEC = importlib.util.spec_from_file_location("report_config_check", SCRIPT_PATH)
report_config_check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["report_config_check"] = report_config_check
SPEC.loader.exec_module(report_config_check)


def _write_env(tmp_path, lines):
    env_file = tmp_path / ".env.local"
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file


def test_main_masks_secret_and_exits_zero(tmp_path, monkeypatch, capsys):
    # Гоняем main(), чтобы реально пройти путь маскировки секрета (present/missing),
    # а не тривиально проверять, что print_recommendations его не печатает.
    _write_env(
        tmp_path,
        [
            "OPENAI_API_KEY=sk-secret",
            "MAC_TRANSCRIBER_REPORT_MODEL=google/gemini-3.1-pro-preview",
            "MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=openai/gpt-5.5",
        ],
    )
    monkeypatch.setattr(report_config_check, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    exit_code = report_config_check.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "sk-secret" not in output
    assert "OPENAI_API_KEY: present" in output
    assert "recommendations: ok" in output


def test_main_exits_nonzero_without_any_provider_key(tmp_path, monkeypatch, capsys):
    _write_env(tmp_path, ["MAC_TRANSCRIBER_REPORT_MODEL=google/gemini-3.1-pro-preview"])
    monkeypatch.setattr(report_config_check, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    exit_code = report_config_check.main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "configure OPENAI_API_KEY or OPENROUTER_API_KEY" in output


def test_recommendations_flag_openrouter_model_without_openrouter_key(capsys):
    blocking = report_config_check.print_recommendations(
        {
            "OPENAI_API_KEY": "sk-secret",
            "MAC_TRANSCRIBER_REPORT_MODEL": "openrouter:google/gemini-3.1-pro-preview",
            "MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL": "openrouter:openai/gpt-5.5",
        }
    )
    output = capsys.readouterr().out

    assert blocking == 1
    assert "OPENROUTER_API_KEY is missing" in output


def test_recommendations_flag_openai_model_without_openai_key(capsys):
    blocking = report_config_check.print_recommendations(
        {
            "OPENROUTER_API_KEY": "sk-or",
            "MAC_TRANSCRIBER_REPORT_MODEL": "gpt-5.5",
            "MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL": "openrouter:openai/gpt-5.5",
        }
    )
    output = capsys.readouterr().out

    assert blocking == 1
    assert "OPENAI_API_KEY is missing" in output


def test_print_recommendations_mentions_optional_upgrade(capsys):
    report_config_check.print_recommendations(
        {
            "OPENAI_API_KEY": "sk-secret",
            "MAC_TRANSCRIBER_REPORT_MODEL": "google/gemini-3.1-pro-preview",
        }
    )
    output = capsys.readouterr().out

    assert "MAC_TRANSCRIBER_BASELINE_UPGRADE_MODEL=openrouter:openai/gpt-5.5" in output
