from __future__ import annotations

import pytest

from transkrib.config import ConfigError, load_daemon_settings, load_settings

TOKEN = "svc_tok_9f8e7d6c5b4a3210-very-secret"


def test_defaults_without_env() -> None:
    s = load_settings()
    assert s.instudy_api_url is None
    assert s.service_token is None
    assert s.poll_interval_sec == 60
    assert s.heartbeat_sec == 300
    assert s.max_segment_sec == 22.0
    assert s.keep_models_loaded is False


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSTUDY_API_URL", "https://api.example.ru/")
    monkeypatch.setenv("SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("POLL_INTERVAL_SEC", "15")
    monkeypatch.setenv("KEEP_MODELS_LOADED", "true")
    s = load_daemon_settings()
    assert s.instudy_api_url == "https://api.example.ru"
    assert s.service_token is not None and s.service_token.get_secret_value() == TOKEN
    assert s.poll_interval_sec == 15
    assert s.keep_models_loaded is True


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.ru",
        "ftp://api.example.ru",
        "api.example.ru",
        "https://",
        "https://u:p@api.example.ru",
        "https://api.example.ru/?x=1",
    ],
)
def test_only_plain_https_url(url: str) -> None:
    with pytest.raises(ConfigError, match="INSTUDY_API_URL"):
        load_settings(instudy_api_url=url)


def test_daemon_requires_url_and_token() -> None:
    with pytest.raises(ConfigError) as exc:
        load_daemon_settings()
    assert "INSTUDY_API_URL" in str(exc.value)
    assert "SERVICE_TOKEN" in str(exc.value)


def test_daemon_requires_token_even_with_url() -> None:
    with pytest.raises(ConfigError, match="SERVICE_TOKEN"):
        load_daemon_settings(instudy_api_url="https://api.example.ru")


def test_cli_settings_do_not_require_token() -> None:
    assert load_settings(instudy_api_url="https://api.example.ru").service_token is None


def test_blank_token_is_missing() -> None:
    with pytest.raises(ConfigError, match="SERVICE_TOKEN"):
        load_daemon_settings(instudy_api_url="https://api.example.ru", service_token="   ")


def test_token_not_in_repr_str_or_dump() -> None:
    s = load_daemon_settings(instudy_api_url="https://api.example.ru", service_token=TOKEN, hf_token="hf_" + TOKEN)
    for text in (repr(s), str(s), str(s.model_dump()), s.model_dump_json()):
        assert TOKEN not in text
    assert set(s.secret_values()) == {TOKEN, "hf_" + TOKEN}


@pytest.mark.parametrize(
    ("field", "value"),
    [("poll_interval_sec", 0), ("heartbeat_sec", -1), ("max_segment_sec", 25), ("results_keep_days", -1)],
)
def test_rejects_bad_numbers(field: str, value: int) -> None:
    with pytest.raises(ConfigError, match=field.upper()):
        load_settings(**{field: value})


def test_validation_error_does_not_echo_secret_input() -> None:
    # Неверное значение рядом с токеном: текст ошибки не должен содержать сам токен.
    with pytest.raises(ConfigError) as exc:
        load_settings(service_token=TOKEN, poll_interval_sec=TOKEN)
    assert TOKEN not in str(exc.value)
