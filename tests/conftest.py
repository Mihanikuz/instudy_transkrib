from __future__ import annotations

import pytest

from transkrib import log

# Переменные окружения, которые читает Settings: тесты не должны зависеть от окружения машины.
_ENV_VARS = (
    "INSTUDY_API_URL",
    "SERVICE_TOKEN",
    "HF_TOKEN",
    "POLL_INTERVAL_SEC",
    "HEARTBEAT_SEC",
    "MAX_AUDIO_BYTES",
    "MAX_AUDIO_SEC",
    "MAX_SEGMENT_SEC",
    "KEEP_MODELS_LOADED",
    "RESULTS_KEEP_DAYS",
    "MIN_FREE_GB",
    "MODELS_DIR",
    "DATA_DIR",
    "LOG_LEVEL",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--gpu", action="store_true", default=False, help="запускать тесты с маркером gpu")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="нужен GPU: запуск с --gpu на сервере")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _clean_secrets() -> None:
    log.clear_secrets()
    yield
    log.clear_secrets()
