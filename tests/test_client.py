from __future__ import annotations

import io
import json
import logging
import socket
import threading
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path

import pytest

from tests.fake_instudy import BASE_URL, TOKEN, FakeInstudy
from transkrib import log
from transkrib.client import (
    Cancelled,
    DownloadFailed,
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
    sanitize_message,
)

MiB = 1024 * 1024


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args, **kwargs):
        raise AssertionError("Тесты не ходят в сеть")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)


@pytest.fixture
def fake() -> FakeInstudy:
    return FakeInstudy()


@pytest.fixture
def client(fake: FakeInstudy) -> InstudyClient:
    sleeps: list[float] = []
    c = InstudyClient(
        BASE_URL,
        TOKEN,
        max_audio_bytes=300 * MiB,
        max_audio_sec=6 * 3600,
        transport=fake.transport,
        sleep=sleeps.append,
    )
    c.sleeps = sleeps  # type: ignore[attr-defined]
    yield c
    c.close()


@pytest.fixture
def logs() -> io.StringIO:
    stream = io.StringIO()
    handler = log.setup_logging("DEBUG", secrets=[TOKEN], stream=stream)
    yield stream
    logging.getLogger().removeHandler(handler)


def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str, bytes]]:
    msg = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    parts = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        parts[name] = (
            part.get_content_type() + (f"; charset={part.get_content_charset()}" if part.get_content_charset() else ""),
            part.get_payload(decode=True),
        )
    return parts


# --------------------------------------------------------------------------- работа


def test_health(client: InstudyClient) -> None:
    assert client.health()["ok"] is True


def test_empty_queue_returns_none(client: InstudyClient) -> None:
    assert client.next_job() is None


def test_next_job_returns_job(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(731, day_id=412)
    job = client.next_job()
    assert isinstance(job, Job)
    assert (job.id, job.day_id, job.audio_size_bytes) == (731, 412, 3_000_000)
    assert job.lease_until.tzinfo is not None
    assert "sig=" not in repr(job) and "audio_url" not in repr(job)


def test_oldest_job_first(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(2, date="2026-10-01")
    fake.add_job(1, date="2025-09-01")
    assert client.next_job().id == 1


def test_job_over_size_limit_is_invalid(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(5).size_override = 301 * MiB
    with pytest.raises(InvalidJob) as exc:
        client.next_job()
    assert exc.value.job_id == 5


def test_job_over_duration_limit_is_invalid(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(6, duration_sec=7 * 3600)
    with pytest.raises(InvalidJob):
        client.next_job()


@pytest.mark.parametrize(
    "patch",
    [{"id": -1}, {"dayId": 0}, {"dayId": "412"}, {"audioSizeBytes": 0}, {"date": "вчера"}, {"leaseUntil": None}],
)
def test_malformed_job_is_invalid(client: InstudyClient, fake: FakeInstudy, patch: dict) -> None:
    import httpx

    good = {
        "id": 7,
        "dayId": 9,
        "date": "2026-09-29",
        "discipline": "x",
        "leaseUntil": "2026-10-22T13:00:00Z",
        "audioUrl": f"{BASE_URL}/api/service/v1/audio/9?exp=1&sig=a",
        "audioSizeBytes": 10,
    }
    fake.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"job": {**good, **patch}}))
    c = InstudyClient(BASE_URL, TOKEN, max_audio_bytes=MiB, max_audio_sec=60, transport=fake.transport)
    with pytest.raises(InvalidJob):
        c.next_job()


def test_audio_url_on_foreign_host_is_invalid(client: InstudyClient, fake: FakeInstudy) -> None:
    import httpx

    job = {
        "id": 7,
        "dayId": 9,
        "date": "2026-09-29",
        "leaseUntil": "2026-10-22T13:00:00Z",
        "audioUrl": "https://evil.example/a.m4a",
        "audioSizeBytes": 10,
    }
    t = httpx.MockTransport(lambda r: httpx.Response(200, json={"job": job}))
    c = InstudyClient(BASE_URL, TOKEN, max_audio_bytes=MiB, max_audio_sec=60, transport=t)
    with pytest.raises(InvalidJob, match="не на сервер instudy") as exc:
        c.next_job()
    assert exc.value.job_id == 7


# --------------------------------------------------------------------------- скачивание


def test_download_whole_file(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1)
    job = client.next_job()
    out = client.download(job, tmp_path / "a.m4a")
    assert out.read_bytes() == fake.jobs[1].audio


def test_download_resumes_with_range_after_break(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1)
    fake.break_download_at = [1_000_000, 500_000]
    job = client.next_job()
    out = client.download(job, tmp_path / "a.m4a")
    assert out.read_bytes() == fake.jobs[1].audio
    ranges = [r.headers.get("range") for r in fake.requests("GET", "/audio/")]
    assert ranges == [None, "bytes=1000000-", "bytes=1500000-"]
    assert client.sleeps == [5.0, 10.0]


def test_download_restarts_when_range_ignored(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1)
    fake.break_download_at = [1_000_000]
    fake.ignore_range = True
    job = client.next_job()
    assert client.download(job, tmp_path / "a.m4a").read_bytes() == fake.jobs[1].audio


def test_download_size_mismatch_fails(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1).size_override = 3_000_001  # instudy обещает больше, чем отдаёт
    job = client.next_job()
    with pytest.raises(DownloadFailed):
        client.download(job, tmp_path / "a.m4a")


def test_download_too_many_breaks_fails(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1)
    fake.break_download_at = [10] * 5
    job = client.next_job()
    with pytest.raises(DownloadFailed):
        client.download(job, tmp_path / "a.m4a", max_attempts=5)


def test_expired_link_is_link_expired(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.advance(minutes=31)
    with pytest.raises(LinkExpired):
        client.download(job, tmp_path / "a.m4a")


def test_download_cancel(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1)
    job = client.next_job()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        client.download(job, tmp_path / "a.m4a", cancel=cancel)


def test_download_has_no_authorization(client: InstudyClient, fake: FakeInstudy, tmp_path: Path) -> None:
    fake.add_job(1)
    client.download(client.next_job(), tmp_path / "a.m4a")
    for r in fake.log:
        if "/audio/" in r.path:
            assert "authorization" not in {k.lower() for k in r.headers}
        else:
            assert r.headers.get("authorization") == f"Bearer {TOKEN}"


# --------------------------------------------------------------------------- коды ответов


def test_unauthorized(fake: FakeInstudy) -> None:
    c = InstudyClient(BASE_URL, "wrong-token-123", max_audio_bytes=MiB, max_audio_sec=60, transport=fake.transport)
    with pytest.raises(Unauthorized):
        c.next_job()


def test_rate_limited_has_retry_after(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.rate_limited = 1
    with pytest.raises(RateLimited) as exc:
        client.next_job()
    assert exc.value.retry_after == 17


def test_server_error_is_transient(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.fail_next("GET jobs/next", 503)
    with pytest.raises(Transient):
        client.next_job()


def test_network_error_is_transient() -> None:
    import httpx

    def boom(request):
        raise httpx.ConnectError("refused")

    c = InstudyClient(BASE_URL, TOKEN, max_audio_bytes=MiB, max_audio_sec=60, transport=httpx.MockTransport(boom))
    with pytest.raises(Transient):
        c.health()


@pytest.mark.parametrize("status", [409, 410])
def test_heartbeat_lease_lost(client: InstudyClient, fake: FakeInstudy, status: int) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.fail_next("POST heartbeat", status)
    with pytest.raises(LeaseLost):
        client.heartbeat(job.id)


def test_heartbeat_extends_lease(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.advance(minutes=10)
    assert client.heartbeat(job.id) > job.lease_until


def test_heartbeat_after_expiry_is_lease_lost(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.advance(minutes=31)
    with pytest.raises(LeaseLost):
        client.heartbeat(job.id)


# --------------------------------------------------------------------------- результат


def test_submit_result_accepted_and_multipart(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(1)
    job = client.next_job()
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\n<v Спикер 1>Привет.\n".encode()
    txt = "Общая психология — 29.09.2026\n".encode()
    meta = {"speakers": 1, "server": "instudy_transkrib 0.1.0"}
    assert client.submit_result(job.id, vtt, txt, meta) is ResultOutcome.ACCEPTED
    parts = parse_multipart(fake.jobs[1].result["content_type"], fake.jobs[1].result["raw"])
    assert set(parts) == {"vtt", "txt", "meta"}
    assert parts["vtt"] == ("text/vtt; charset=utf-8", vtt)
    assert parts["txt"] == ("text/plain; charset=utf-8", txt)
    assert json.loads(parts["meta"][1]) == meta


def test_submit_result_manual_transcript_409(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.jobs[1].manual_transcript = True
    assert client.submit_result(job.id, b"v", b"t", {}) is ResultOutcome.ALREADY_HAS_TRANSCRIPT


def test_submit_result_gone_410(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.jobs[1].gone = True
    assert client.submit_result(job.id, b"v", b"t", {}) is ResultOutcome.GONE


@pytest.mark.parametrize(
    ("status", "reason"),
    [(413, FailReason.RESULT_TOO_LARGE), (415, FailReason.RESULT_REJECTED), (422, FailReason.RESULT_REJECTED)],
)
def test_submit_result_rejected(client: InstudyClient, fake: FakeInstudy, status: int, reason: FailReason) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.fail_next("POST result", status)
    with pytest.raises(Rejected) as exc:
        client.submit_result(job.id, b"v", b"t", {})
    assert exc.value.reason is reason


# --------------------------------------------------------------------------- отказ


def test_fail_sends_masked_short_message(client: InstudyClient, fake: FakeInstudy) -> None:
    log.register_secret(TOKEN)
    fake.add_job(1)
    job = client.next_job()
    message = f"ошибка с токеном {TOKEN} при скачивании {job.audio_url} файла /data/work/1/audio.m4a " + "x" * 500
    assert client.fail(job.id, FailReason.DOWNLOAD_FAILED, message) is True
    sent = fake.jobs[1].fails[0]
    assert sent["reason"] == "download_failed" and sent["retryable"] is True
    assert TOKEN not in sent["message"]
    assert "sig=" not in sent["message"] and "exp=" not in sent["message"]
    assert "/data/work" not in sent["message"]
    assert len(sent["message"]) <= 300


def test_fail_retryable_from_table_and_override(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(1)
    job = client.next_job()
    client.fail(job.id, FailReason.NO_SPEECH, "тишина")
    assert fake.jobs[1].fails[0]["retryable"] is False
    job = client.next_job()
    client.fail(job.id, FailReason.NO_SPEECH, "тишина", retryable=True)
    assert fake.jobs[1].fails[1]["retryable"] is True


def test_fail_on_lost_lease_returns_false(client: InstudyClient, fake: FakeInstudy) -> None:
    fake.add_job(1)
    job = client.next_job()
    fake.advance(hours=1)
    assert client.fail(job.id, FailReason.OOM, "oom") is False


def test_fail_reason_must_be_known(client: InstudyClient) -> None:
    with pytest.raises(ValueError):
        client.fail(1, "something_else", "x")  # type: ignore[arg-type]


def test_sanitize_message_collapses_lines() -> None:
    assert sanitize_message('Traceback\n  File "/app/x.py", line 3\nValueError') == (
        'Traceback File "<path>", line 3 ValueError'
    )


# --------------------------------------------------------------------------- утечки


def test_no_secrets_in_logs_or_exceptions(
    client: InstudyClient, fake: FakeInstudy, tmp_path: Path, logs: io.StringIO
) -> None:
    fake.add_job(1)
    fake.break_download_at = [100]
    job = client.next_job()
    client.download(job, tmp_path / "a.m4a")
    errors: list[str] = []
    fake.advance(minutes=31)
    for call in (lambda: client.download(job, tmp_path / "b.m4a"), lambda: client.heartbeat(job.id)):
        try:
            call()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{exc!r} {exc}")
    logging.getLogger("transkrib.test").exception("ошибка", extra={"job": repr(job)})
    out = logs.getvalue() + " ".join(errors)
    assert errors
    assert TOKEN not in out
    assert "sig=" not in out and job.audio_url not in out


def test_rejects_plain_http() -> None:
    with pytest.raises(ValueError):
        InstudyClient("http://api.example.ru", TOKEN, max_audio_bytes=1, max_audio_sec=1)
