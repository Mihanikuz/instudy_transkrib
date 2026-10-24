"""Контракт с instudy, зафиксированный тестом: пути, поля, коды и лимиты из instudy/docs/transcription-api.md.

Сверено с instudy main 4e55604 (23.10.2026), перепроверено на c5ad409 (24.10.2026) — без изменений.
Если instudy поменяет договор, должен упасть этот тест, а не прод:
тогда обновить договорные константы ниже вместе с клиентом, фейком (tests/fake_instudy.py), эталоном
(tests/instudy_reference.py) и docs/contract.md.

Каждый код ответа проверяется на настоящем клиенте с подставленным ответом (httpx.MockTransport), а не на фейке:
так тест фиксирует, как клиент понимает код, независимо от того, как его отдаёт фейк.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from tests.instudy_reference import MAX_HOURS, MetaIn, parse_multipart
from transkrib import client as client_mod
from transkrib.client import (
    API_PREFIX,
    RETRYABLE,
    FailReason,
    InstudyClient,
    InvalidJob,
    Job,
    LeaseLost,
    LinkExpired,
    RateLimited,
    Rejected,
    ResultOutcome,
    Transient,
    Unauthorized,
)
from transkrib.render import MAX_SPEAKERS, MAX_TIMECODE_MS, build_meta

BASE = "https://api.example.ru"
TOKEN = "svc_tok_contract_1234567890"
MiB = 1024 * 1024

# ------------------------------------------------------------------ договор (transcription-api.md)

PREFIX = "/api/service/v1"
ENDPOINTS = {
    "health": ("GET", "/health"),
    "next": ("GET", "/jobs/next"),
    "heartbeat": ("POST", "/jobs/{id}/heartbeat"),
    "result": ("POST", "/jobs/{id}/result"),
    "fail": ("POST", "/jobs/{id}/fail"),
}
JOB_FIELDS = {
    "id",
    "dayId",
    "date",
    "discipline",
    "audioUrl",
    "audioSizeBytes",
    "audioDurationSec",
    "audioSha256",
    "leaseUntil",
}
HEALTH_FIELDS = {"ok", "serverTime", "client", "paused"}
RESULT_FIELDS = {"vtt", "txt", "meta"}
META_REQUIRED = {"server", "asr", "diarization"}
META_OPTIONAL = {"audioDurationSec", "processingSec", "speakers", "segments"}
FAIL_REASONS = {
    "download_failed": True,
    "decode_failed": False,
    "no_speech": False,
    "oom": True,
    "model_error": True,
    "invalid_job": False,
    "result_too_large": False,
    "result_rejected": False,
    "shutdown": True,
}
LIMITS = {"file": 10 * MiB, "meta": 4096, "body": 50 * MiB, "speakers": 99, "hours": 12, "message": 2000}

JOB = {
    "id": 731,
    "dayId": 412,
    "date": "2026-09-29",
    "discipline": "Общая психология",
    "audioUrl": f"{BASE}{PREFIX}/audio/412?job=731&exp=1761141600&sig=Q2hhbmdlTWU",
    "audioSizeBytes": 108234567,
    "audioDurationSec": 17940,
    "audioSha256": "9f2c" + "0" * 60,
    "leaseUntil": "2026-10-22T13:30:00Z",
}


def make(status: int, body=None, headers=None, log: list[httpx.Request] | None = None) -> InstudyClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if log is not None:
            log.append(request)
        if body is None:
            return httpx.Response(status, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    return InstudyClient(
        BASE,
        TOKEN,
        max_audio_bytes=300 * MiB,
        max_audio_sec=21600,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


def call(c: InstudyClient, name: str, tmp_path: Path | None = None):
    if name == "health":
        return c.health()
    if name == "next":
        return c.next_job()
    if name == "heartbeat":
        return c.heartbeat(731)
    if name == "result":
        return c.submit_result(731, b"WEBVTT\n", b"txt", {"server": "s", "asr": "a", "diarization": "d"})
    if name == "fail":
        return c.fail(731, FailReason.OOM, "oom")
    raise AssertionError(name)


# ------------------------------------------------------------------ пути, авторизация


def test_prefix() -> None:
    assert API_PREFIX == PREFIX


@pytest.mark.parametrize("name", sorted(ENDPOINTS))
def test_endpoint_paths_and_bearer(name: str) -> None:
    log: list[httpx.Request] = []
    body = {"health": {"ok": True}, "heartbeat": {"leaseUntil": "2026-10-22T13:35:00Z"}}.get(name)
    status = {"next": 204, "result": 202, "fail": 204}.get(name, 200)
    call(make(status, body, log=log), name)
    method, path = ENDPOINTS[name]
    assert [(r.method, r.url.path) for r in log] == [(method, PREFIX + path.replace("{id}", "731"))]
    assert log[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert "cookie" not in log[0].headers


def test_audio_download_without_authorization(tmp_path: Path) -> None:
    import hashlib

    data = b"m4a" * 100
    log: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        log.append(request)
        return httpx.Response(200, content=data, headers={"Content-Type": "audio/mp4"})

    c = InstudyClient(BASE, TOKEN, max_audio_bytes=MiB, max_audio_sec=60, transport=httpx.MockTransport(handler))
    job = Job.model_validate({**JOB, "audioSizeBytes": len(data), "audioSha256": hashlib.sha256(data).hexdigest()})
    c.download(job, tmp_path / "a.m4a")
    assert str(log[0].url) == job.audio_url  # ссылка берётся целиком, как есть
    assert "authorization" not in log[0].headers


# ------------------------------------------------------------------ поля


def test_job_fields() -> None:
    aliases = {f.alias or name for name, f in Job.model_fields.items()}
    assert aliases == JOB_FIELDS
    job = Job.model_validate(JOB)
    assert job.audio_sha256 == JOB["audioSha256"] and job.audio_duration_sec == 17940


def test_job_duration_may_be_null() -> None:
    assert Job.model_validate({**JOB, "audioDurationSec": None}).audio_duration_sec is None


@pytest.mark.parametrize("field", sorted(JOB_FIELDS - {"discipline", "audioDurationSec"}))
def test_job_required_fields(field: str) -> None:
    job = {k: v for k, v in JOB.items() if k != field}
    with pytest.raises(InvalidJob):
        make(200, {"job": job}).next_job()


def test_health_fields_and_paused() -> None:
    body = {"ok": True, "serverTime": "2026-10-22T13:00:00Z", "client": "home-gpu", "paused": True}
    data = make(200, body).health()
    assert set(data) == HEALTH_FIELDS
    assert InstudyClient.health_paused(data) is True
    assert InstudyClient.health_paused({"ok": True}) is False


def test_result_multipart_fields() -> None:
    log: list[httpx.Request] = []
    meta = {"server": "s", "asr": "a", "diarization": "d", "speakers": 2}
    make(202, {"transcriptId": 1}, log=log).submit_result(731, b"WEBVTT\n", b"txt", meta)
    request = log[0]
    parts = parse_multipart(request.headers["Content-Type"], request.read())
    assert parts is not None and set(parts) == RESULT_FIELDS
    assert parts["vtt"][0].startswith("text/vtt") and parts["txt"][0].startswith("text/plain")
    assert json.loads(parts["meta"][1]) == meta


def test_meta_matches_instudy_schema() -> None:
    meta = build_meta(
        server="instudy_transkrib 0.1.0",
        asr="gigaam v3_e2e_rnnt @7447938d79",
        diarization="pyannote/speaker-diarization-community-1",
        audio_duration_sec=17940.4,
        processing_sec=912.2,
        speakers=150,
        segments=1873,
    )
    assert set(meta) == META_REQUIRED | META_OPTIONAL
    assert set(MetaIn.model_fields) == META_REQUIRED | META_OPTIONAL
    assert MetaIn.model_validate(meta).speakers == 99
    assert all(isinstance(meta[k], int) for k in META_OPTIONAL)


# ------------------------------------------------------------------ коды ответов


@pytest.mark.parametrize(
    ("name", "status", "expected"),
    [
        ("health", 401, Unauthorized),
        ("health", 429, RateLimited),
        ("next", 204, None),
        ("next", 401, Unauthorized),
        ("next", 429, RateLimited),
        ("next", 503, Transient),
        ("heartbeat", 401, Unauthorized),
        ("heartbeat", 410, LeaseLost),
        ("heartbeat", 409, LeaseLost),  # совместимость: договор отдаёт только 410
        ("heartbeat", 422, Transient),
        ("heartbeat", 429, RateLimited),
        ("result", 202, ResultOutcome.ACCEPTED),
        ("result", 400, Transient),  # форма оборвалась — повторять, пока жива аренда
        ("result", 401, Unauthorized),
        ("result", 409, ResultOutcome.ALREADY_HAS_TRANSCRIPT),
        ("result", 410, ResultOutcome.GONE),
        ("result", 413, FailReason.RESULT_TOO_LARGE),
        ("result", 415, FailReason.RESULT_REJECTED),
        ("result", 422, FailReason.RESULT_REJECTED),
        ("result", 429, RateLimited),
        ("result", 500, Transient),
        ("fail", 204, True),
        ("fail", 401, Unauthorized),
        ("fail", 410, False),
        ("fail", 409, False),  # совместимость
        ("fail", 429, RateLimited),
    ],
)
def test_status_codes(name: str, status: int, expected) -> None:
    body = {"detail": "…"} if status >= 400 else ({"transcriptId": 1534} if status == 202 else None)
    c = make(status, body, headers={"Retry-After": "17"} if status == 429 else None)
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            call(c, name)
    elif isinstance(expected, FailReason):
        with pytest.raises(Rejected) as err:
            call(c, name)
        assert err.value.reason is expected and err.value.status == status
    else:
        assert call(c, name) == expected


def test_caddy_413_without_body_is_result_too_large() -> None:
    """Тело результата сверх лимита Caddy instudy (OPS-2: max_size 51MiB) режет ещё до API: 413 с пустым телом,
    не JSON. Клиент смотрит только на код — тот же fail result_too_large, что и на 413 от API."""
    c = make(413, headers={"Content-Type": "text/plain; charset=utf-8"})  # body=None — пустое тело, как у Caddy
    with pytest.raises(Rejected) as err:
        call(c, "result")
    assert err.value.reason is FailReason.RESULT_TOO_LARGE and err.value.status == 413


def test_retry_after_seconds() -> None:
    with pytest.raises(RateLimited) as err:
        make(429, {"detail": "…"}, headers={"Retry-After": "17"}).health()
    assert err.value.retry_after == 17.0


@pytest.mark.parametrize(("status", "expected"), [(403, LinkExpired), (416, None)])
def test_audio_status_codes(tmp_path: Path, status: int, expected) -> None:
    c = make(status, {"detail": "Ссылка недействительна или устарела"})
    job = Job.model_validate({**JOB, "audioSizeBytes": 10})
    if expected is None:
        with pytest.raises(client_mod.DownloadFailed):  # 416 на пустой файл — начать заново, попытки кончатся
            c.download(job, tmp_path / "a.m4a", backoff=0)
    else:
        with pytest.raises(expected):
            c.download(job, tmp_path / "a.m4a")


# ------------------------------------------------------------------ коды reason и лимиты


def test_fail_reasons_and_retryable() -> None:
    assert {r.value: RETRYABLE[r] for r in FailReason} == FAIL_REASONS
    assert all(re.fullmatch(r"[a-z_]{1,32}", r.value) for r in FailReason)


def test_fail_body() -> None:
    log: list[httpx.Request] = []
    make(204, log=log).fail(731, FailReason.SHUTDOWN, "Демон остановлен /home/user/x " + "я" * 3000)
    body = json.loads(log[0].read())
    assert set(body) == {"reason", "retryable", "message"}
    assert body["reason"] == "shutdown" and body["retryable"] is True
    assert len(body["message"]) <= min(300, LIMITS["message"]) and "/home" not in body["message"]
    assert log[0].headers["Content-Type"].startswith("application/json")


def test_result_limits() -> None:
    assert client_mod.RESULT_FILE_MAX == LIMITS["file"]
    assert client_mod.RESULT_META_MAX == LIMITS["meta"]
    assert client_mod.RESULT_BODY_MAX == LIMITS["body"]
    assert MAX_SPEAKERS == LIMITS["speakers"]
    assert MAX_TIMECODE_MS == LIMITS["hours"] * 3600 * 1000 == MAX_HOURS * 3600 * 1000
