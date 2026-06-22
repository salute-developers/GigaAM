"""Изоляция окружения для тестов.

В рабочем `.env.local` стоит прод-флаг `MAC_TRANSCRIBER_REPORT_BACKEND=claude`: он
уводит генерацию отчётов в очередь (kill-switch в `reporting.build_ai_report`).
`service.py` грузит `.env.local` на импорте, а `test_service.py` импортит `service`
на уровне модуля — без изоляции флаг протекает в тест-процесс и валит тесты AI-отчётов.

Снимаем флаг перед каждым тестом. Тест, которому он реально нужен (kill-switch),
ставит его сам через `monkeypatch.setenv`, и это срабатывает уже после фикстуры.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_report_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAC_TRANSCRIBER_REPORT_BACKEND", raising=False)
