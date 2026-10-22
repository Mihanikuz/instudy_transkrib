from __future__ import annotations

import io
import json
import logging

import pytest

from transkrib import log

TOKEN = "svc_tok_9f8e7d6c5b4a3210-very-secret"
SIGNED = "https://api.example.ru/api/service/v1/audio/5?exp=1761140400&sig=abcDEF123"


@pytest.fixture
def capture() -> io.StringIO:
    stream = io.StringIO()
    handler = log.setup_logging("DEBUG", secrets=[TOKEN], stream=stream)
    yield stream
    logging.getLogger().removeHandler(handler)


def records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_masks_token_in_message(capture: io.StringIO) -> None:
    logging.getLogger("t").info(f"token is {TOKEN}")
    out = capture.getvalue()
    assert TOKEN not in out
    assert records(capture)[0]["msg"] == "token is ***"


def test_masks_token_in_args(capture: io.StringIO) -> None:
    logging.getLogger("t").warning("auth %s failed for %s", TOKEN, "home-gpu")
    assert TOKEN not in capture.getvalue()
    assert "home-gpu" in capture.getvalue()


def test_masks_token_in_extra(capture: io.StringIO) -> None:
    logging.getLogger("t").info("request", extra={"headers": {"Authorization": f"Bearer {TOKEN}"}, "url": SIGNED})
    out = capture.getvalue()
    assert TOKEN not in out
    assert "sig=abc" not in out and "1761140400" not in out
    rec = records(capture)[0]
    assert rec["headers"]["Authorization"] == "Bearer ***"


def test_masks_token_in_exception_text(capture: io.StringIO) -> None:
    try:
        raise RuntimeError(f"bad header Bearer {TOKEN} for {SIGNED}")
    except RuntimeError:
        logging.getLogger("t").exception("download failed")
    out = capture.getvalue()
    assert TOKEN not in out
    assert "sig=abc" not in out
    assert "RuntimeError" in records(capture)[0]["exc"]


def test_masks_service_url_query() -> None:
    assert log.redact(f"GET {SIGNED} 200") == "GET https://api.example.ru/api/service/v1/audio/5?*** 200"


def test_masks_unversioned_service_url_query() -> None:
    text = "https://api.example/api/service/audio/5?exp=1&sig=abc"
    assert log.redact(text) == "https://api.example/api/service/audio/5?***"


def test_masks_sig_and_exp_anywhere() -> None:
    assert log.redact("x?a=1&sig=Zz9&exp=99 y") == "x?a=1&sig=***&exp=*** y"


def test_bearer_without_registered_secret() -> None:
    assert log.redact("Authorization: Bearer abc.def-ghi") == "Authorization: Bearer ***"


def test_does_not_touch_ordinary_text() -> None:
    text = "Диаризация: 4 спикера, 1873 сегмента, 912 с"
    assert log.redact(text) == text


def test_short_values_are_not_registered() -> None:
    log.register_secret("abc")
    assert log.redact("abcdef") == "abcdef"


def test_json_lines_have_required_fields(capture: io.StringIO) -> None:
    logging.getLogger("transkrib.test").error("boom", extra={"job_id": 731})
    rec = records(capture)[0]
    assert rec["level"] == "ERROR"
    assert rec["logger"] == "transkrib.test"
    assert rec["job_id"] == 731
    assert rec["ts"].endswith("Z")


def test_broken_format_does_not_raise(capture: io.StringIO) -> None:
    logging.getLogger("t").info("value %d", "not-a-number")
    assert records(capture)


def test_httpx_info_is_silenced(capture: io.StringIO) -> None:
    logging.getLogger("httpx").info(f"HTTP Request: GET {SIGNED}")
    assert capture.getvalue() == ""
