"""Тесты kill-switch бэкенда отчётов (MAC_TRANSCRIBER_REPORT_BACKEND=claude).

Проверяем, что флаг в самом начале build_ai_report() отключает синхронный API-путь
ДО любых сетевых вызовов и чтения ключей, бросая ReportUnavailableError с особым
сообщением. Без флага срабатывает уже другая ветка (нет ключа -> другое сообщение),
что доказывает: именно флаг управляет этой веткой.
"""

import pytest

from mac_transcriber.asr import Segment
from mac_transcriber.reporting import (
    ReportUnavailableError,
    build_ai_report,
    build_local_report,
)


def _segment(speaker: str, start: float, end: float, text: str) -> Segment:
    return Segment(
        speaker=speaker,
        track="audio.m4a",
        start=start,
        end=end,
        text=text,
    )


def _base_report():
    return build_local_report(
        meeting_id="planning-call",
        title="Planning Call",
        source_filename="planning.m4a",
        model_name="v3_e2e_rnnt",
        segments=[
            _segment("Илья", 0.0, 12.0, "Договорились сделать PDF отчеты."),
            _segment("Анна", 12.5, 25.0, "Нужно сохранить ссылки на таймкоды."),
        ],
    )


@pytest.fixture(autouse=True)
def _isolate_keys(tmp_path, monkeypatch):
    """Глушим источники ключей: пустой cwd без .env.local и снятые env-переменные.

    Без этого ветка "нет флага" могла бы подхватить реальный ключ из окружения
    разработчика и уйти в сеть. С чистым окружением она детерминированно падает
    на "OPENAI_API_KEY is not configured" — без единого сетевого запроса.
    """
    monkeypatch.chdir(tmp_path)
    for name in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "MAC_TRANSCRIBER_OPENAI_ENV_FILE",
        "MAC_TRANSCRIBER_OPENROUTER_ENV_FILE",
        "MAC_TRANSCRIBER_OPENROUTER_FALLBACK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_report_backend_claude_defers_before_network(monkeypatch):
    """С флагом claude build_ai_report падает сразу с deferred-сообщением.

    Любой реальный POST к провайдеру в этом тесте — баг: подменяем сетевые
    функции на взрывающиеся, чтобы это поймать.
    """

    def _boom(*args, **kwargs):  # pragma: no cover - не должно вызываться
        raise AssertionError("network call must not happen when backend=claude")

    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", _boom)
    monkeypatch.setattr("mac_transcriber.reporting._post_openrouter_response", _boom)
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_BACKEND", "claude")

    with pytest.raises(ReportUnavailableError) as excinfo:
        build_ai_report(base_report=_base_report(), report_model="gpt-5.5")

    assert "deferred to scheduled claude job" in str(excinfo.value)


@pytest.mark.parametrize("flag_value", ["claude", "  Claude  ", "CLAUDE", "ClAuDe"])
def test_report_backend_flag_is_case_and_whitespace_insensitive(
    monkeypatch, flag_value
):
    """Флаг распознаётся без учёта регистра и обрамляющих пробелов."""

    def _boom(*args, **kwargs):  # pragma: no cover - не должно вызываться
        raise AssertionError("network call must not happen when backend=claude")

    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", _boom)
    monkeypatch.setattr("mac_transcriber.reporting._post_openrouter_response", _boom)
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_BACKEND", flag_value)

    with pytest.raises(ReportUnavailableError) as excinfo:
        build_ai_report(base_report=_base_report(), report_model="gpt-5.5")

    assert "deferred to scheduled claude job" in str(excinfo.value)


def test_without_flag_kill_switch_does_not_fire(monkeypatch):
    """Без флага kill-switch не срабатывает: падаем на другой ошибке (нет ключа).

    Это и отличает, что именно флаг рулит deferred-веткой: тот же вызов без флага
    доходит до проверки ключей и бросает другое сообщение.
    """
    monkeypatch.delenv("MAC_TRANSCRIBER_REPORT_BACKEND", raising=False)

    def _boom(*args, **kwargs):  # pragma: no cover - не должно вызываться
        raise AssertionError("no network call expected without an API key")

    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", _boom)
    monkeypatch.setattr("mac_transcriber.reporting._post_openrouter_response", _boom)

    with pytest.raises(ReportUnavailableError) as excinfo:
        build_ai_report(base_report=_base_report(), report_model="gpt-5.5")

    message = str(excinfo.value)
    assert "deferred to scheduled claude job" not in message
    assert "OPENAI_API_KEY is not configured" in message


def test_empty_backend_flag_does_not_fire(monkeypatch):
    """Пустая/пробельная строка флага не считается claude-режимом."""
    monkeypatch.setenv("MAC_TRANSCRIBER_REPORT_BACKEND", "   ")

    def _boom(*args, **kwargs):  # pragma: no cover - не должно вызываться
        raise AssertionError("no network call expected without an API key")

    monkeypatch.setattr("mac_transcriber.reporting._post_openai_response", _boom)
    monkeypatch.setattr("mac_transcriber.reporting._post_openrouter_response", _boom)

    with pytest.raises(ReportUnavailableError) as excinfo:
        build_ai_report(base_report=_base_report(), report_model="gpt-5.5")

    message = str(excinfo.value)
    assert "deferred to scheduled claude job" not in message
    assert "OPENAI_API_KEY is not configured" in message
