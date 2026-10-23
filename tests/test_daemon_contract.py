"""T-9: поведение демона по итоговому договору instudy (transcription-api.md)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fakes import FakeRecognizer
from tests.instudy_reference import parse_multipart
from tests.test_daemon import POLL, Env
from transkrib import cli
from transkrib.daemon import HEALTH_REFRESH_SEC
from transkrib.state import read_state


def _fail_reasons(env: Env) -> list[tuple[str, bool]]:
    return [(json.loads(r.body)["reason"], json.loads(r.body)["retryable"]) for r in env.fake.requests("POST", "/fail")]


def test_sha256_mismatch_sends_download_failed(tmp_path: Path) -> None:
    env = Env(tmp_path)
    job = env.add_job()
    job.sha_override = "f" * 64
    env.run()
    assert _fail_reasons(env)[0] == ("download_failed", True)
    assert "SHA-256" in job.fails[0]["message"]
    assert env.fake.requests("POST", "/result") == []
    assert env.work_items() == []
    env.assert_state_clean()


class HugeText(FakeRecognizer):
    def transcribe_batch(self, chunks):
        super().transcribe_batch(chunks)
        return ["очень длинная реплика " * 300_000 for _ in chunks]  # ≈ 12 МБ в .txt


def test_huge_result_is_not_sent_but_failed(tmp_path: Path) -> None:
    env = Env(tmp_path, recognizer=HugeText())
    job = env.add_job()
    env.run()
    assert env.fake.requests("POST", "/result") == []
    assert job.fails[0]["reason"] == "result_too_large" and job.fails[0]["retryable"] is False
    assert job.status == "failed"
    assert read_state(env.data)["last_error"]["code"] == "result_too_large"
    assert not (env.data / "results" / "1").exists() and env.work_items() == []


def test_result_400_retried_while_lease_alive(tmp_path: Path) -> None:
    env = Env(tmp_path)
    job = env.add_job()
    env.fake.fail_next("POST result", 400, 400)
    env.run()
    assert len(env.fake.requests("POST", "/result")) == 3
    assert job.status == "done" and job.fails == []
    assert env.clock.waits[:2] == [5.0, 10.0]


def test_result_202_with_transcript_id_body(tmp_path: Path) -> None:
    env = Env(tmp_path)
    job = env.add_job()
    env.run()
    assert job.transcript_id == 1534
    parts = parse_multipart(job.result["content_type"], job.result["raw"])
    meta = json.loads(parts["meta"][1])
    assert {"server", "asr", "diarization"} <= set(meta) and 1 <= meta["speakers"] <= 99
    assert read_state(env.data)["counters"]["done"] == 1


def test_instudy_pause_keeps_polling_and_is_shown(tmp_path: Path, capsys) -> None:
    env = Env(tmp_path)
    env.fake.paused = True
    job = env.add_job()
    seen: list[tuple[int, bool | None]] = []

    def on_wait(seconds: float) -> None:
        state = read_state(env.data)
        seen.append((len(env.fake.requests("GET", "/jobs/next")), state["instudy_paused"]))
        if len(seen) == 3:
            cli.cmd_status(env.data, now=env.clock.now())
            assert "выдача работ на паузе" in capsys.readouterr().out
            env.fake.paused = False  # админ снял паузу
        if len(seen) >= 3 and env.fake.jobs[1].status == "done" and seconds == POLL:
            env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    # Опрос не прекращался: jobs/next раз в POLL_INTERVAL_SEC, статус демона — idle, не ошибка.
    assert [n for n, _ in seen[:3]] == [1, 2, 3]
    assert all(paused is True for _, paused in seen[:3])
    assert job.status == "done"
    assert all(s == POLL for s in env.clock.waits[:3])
    assert read_state(env.data)["status"] == "idle"


def test_health_refreshed_on_empty_queue_not_more_often_than_10_minutes(tmp_path: Path) -> None:
    env = Env(tmp_path)

    def on_wait(seconds: float) -> None:
        if len(env.clock.waits) >= 30:
            env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    polls = len(env.clock.waits)
    health = len(env.fake.requests("GET", "/health"))
    # старт + обновления не чаще раза в 600 с при опросе раз в 77 с
    assert health == 1 + int(polls * POLL // HEALTH_REFRESH_SEC)


def test_cli_check_shows_instudy_pause(tmp_path: Path, monkeypatch, capsys) -> None:
    import httpx

    body = {"ok": True, "serverTime": "2026-10-22T13:00:00Z", "client": "home-gpu", "paused": True}
    monkeypatch.setenv("INSTUDY_API_URL", "https://api.example.ru")
    monkeypatch.setenv("SERVICE_TOKEN", "svc_tok_fake_1234567890abcdef")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)
    cli.cmd_check()
    out = capsys.readouterr().out
    assert "instudy: ok" in out and "выдача работ на паузе" in out
    assert "svc_tok_fake" not in out
