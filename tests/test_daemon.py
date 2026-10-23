from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.audio_gen import tone, write_wav
from tests.fake_instudy import BASE_URL, TOKEN, FakeInstudy
from tests.fakes import FakeDiarizer, FakeRecognizer
from transkrib import cli
from transkrib import daemon as daemon_mod
from transkrib.client import InstudyClient
from transkrib.config import load_settings
from transkrib.daemon import Daemon
from transkrib.diarize import Turn
from transkrib.state import StateStore, read_state

POLL = 77.0  # отличается от задержек повторов (5, 10, …) — по нему тесты узнают сон пустой очереди
ROOT = Path(__file__).resolve().parent.parent


class FakeClock:
    """Подменяемые часы. Сон главного потока мгновенно сдвигает время и зовёт on_wait(seconds);
    фоновые потоки (heartbeat, тикер) тикают каждые 5 мс реального времени, не двигая часы."""

    def __init__(self) -> None:
        self.t = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
        self.owner = threading.current_thread()
        self.waits: list[float] = []
        self.limit = 200
        self.on_wait: Callable[[float], None] = lambda s: None

    def now(self) -> datetime:
        return self.t

    def wait(self, event: threading.Event, seconds: float) -> bool:
        if threading.current_thread() is not self.owner:
            return event.wait(0.005)
        if event.is_set():
            return True
        self.waits.append(seconds)
        self.t += timedelta(seconds=seconds)
        self.on_wait(seconds)
        if len(self.waits) > self.limit:
            raise AssertionError("демон зациклился")
        return event.is_set()


class Env:
    def __init__(self, tmp_path: Path, *, diarizer=None, recognizer=None, free_gb: float = 100.0, **settings) -> None:
        self.fake = FakeInstudy()
        self.clock = FakeClock()
        self.data = tmp_path / "data"
        self.free_gb = free_gb
        self.states: list[str] = []
        self.diarizer = diarizer or FakeDiarizer()
        self.recognizer = recognizer or FakeRecognizer()
        self.settings = load_settings(
            data_dir=self.data, models_dir=tmp_path / "models", poll_interval_sec=POLL, **settings
        )
        self.client = InstudyClient(
            BASE_URL,
            TOKEN,
            max_audio_bytes=self.settings.max_audio_bytes,
            max_audio_sec=self.settings.max_audio_sec,
            transport=self.fake.transport,
            sleep=lambda s: None,
        )
        self.daemon = Daemon(
            self.settings,
            self.client,
            self.diarizer,
            self.recognizer,
            clock=self.clock,
            disk_usage=lambda p: SimpleNamespace(free=self.free_gb * 1000**3),
            models_check=lambda d: [],
            state_observer=self._observe,
        )
        # По умолчанию: первый сон пустой очереди — остановка.
        self.clock.on_wait = lambda s: self.daemon.request_stop() if s == POLL else None

    def _observe(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        json.loads(text)  # валидный JSON после каждой записи
        self.states.append(text)

    def add_job(self, job_id: int = 1, *, seconds: float = 20.0, audio: bytes | None = None):
        job = self.fake.add_job(job_id)
        job.audio = (
            audio if audio is not None else write_wav(self.data.parent / f"src{job_id}.wav", tone(seconds)).read_bytes()
        )
        job.duration_sec = int(seconds)
        return job

    def run(self) -> int:
        return self.daemon.run()

    def work_items(self) -> list[str]:
        return sorted(p.name for p in (self.data / "work").iterdir())

    def assert_state_clean(self) -> None:
        assert self.states
        for text in self.states:
            assert TOKEN not in text
            assert "sig=" not in text and "exp=" not in text and "audio/" not in text
            assert "Общая психология" not in text and "фраза" not in text
        for thread in threading.enumerate():
            assert not thread.name.startswith(("heartbeat-", "job-")), thread.name


def spaced_turns(n: int) -> list[Turn]:
    return [Turn(2.0 * i, 2.0 * i + 1.0, "SPEAKER_00") for i in range(n)]


def test_full_cycle(tmp_path: Path) -> None:
    env = Env(tmp_path)
    job = env.add_job(731)
    assert env.run() == 0
    assert job.status == "done" and job.result is not None and job.fails == []
    assert env.work_items() == []
    result_dir = env.data / "results" / "731"
    assert sorted(p.name for p in result_dir.iterdir()) == ["meta.json", "transcript.txt", "transcript.vtt"]
    assert (result_dir / "transcript.vtt").read_text(encoding="utf-8").startswith("WEBVTT\n")
    meta = json.loads((result_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["asr"] == "fake-asr" and meta["segments"] == 1
    state = read_state(env.data)
    assert state["status"] == "idle" and state["counters"]["done"] == 1 and state["job"] is None
    # шапка .txt — из работы (дисциплина и дата), meta — JSON-строкой в multipart
    assert "Общая психология — 29.09.2026".encode() in job.result["raw"]
    env.assert_state_clean()


@pytest.mark.parametrize(("status", "counter"), [(409, "skipped"), (410, "cancelled")])
def test_result_409_410_without_fail(tmp_path: Path, status: int, counter: str) -> None:
    env = Env(tmp_path)
    job = env.add_job()
    env.fake.fail_next("POST result", status)
    env.run()
    assert job.fails == []
    assert env.fake.requests("POST", "/fail") == []
    assert env.work_items() == []
    assert not (env.data / "results" / "1").exists()
    assert read_state(env.data)["counters"][counter] == 1
    env.assert_state_clean()


class LeaseLossRecognizer(FakeRecognizer):
    """На третьем куске день «исчезает» в instudy; ждём, пока heartbeat заметит 410 и выставит отмену."""

    def __init__(self, env_ref: list[Env]) -> None:
        super().__init__()
        self.env_ref = env_ref

    def transcribe_batch(self, chunks):
        out = super().transcribe_batch(chunks)
        if self.calls == 3:
            env = self.env_ref[0]
            env.fake.jobs[1].gone = True
            ctx = env.daemon._current
            assert ctx is not None and ctx.cancel.wait(5), "heartbeat не выставил отмену"
        return out


def test_lease_lost_mid_asr(tmp_path: Path) -> None:
    ref: list[Env] = []
    recognizer = LeaseLossRecognizer(ref)
    env = Env(tmp_path, diarizer=FakeDiarizer(spaced_turns(20)), recognizer=recognizer)
    ref.append(env)
    job = env.add_job(seconds=40.0)
    env.run()
    assert recognizer.calls == 3  # после отмены распознавание не продолжилось
    assert job.fails == [] and env.fake.requests("POST", "/fail") == []
    assert env.fake.requests("POST", "/result") == []
    assert env.work_items() == []
    assert read_state(env.data)["counters"]["cancelled"] == 1
    env.assert_state_clean()


@pytest.mark.parametrize(
    ("case", "reason", "retryable"),
    [("decode", "decode_failed", False), ("no_speech", "no_speech", False), ("oom", "oom", True)],
)
def test_pipeline_errors_send_fail(tmp_path: Path, case: str, reason: str, retryable: bool) -> None:
    diarizer = FakeDiarizer([]) if case == "no_speech" else FakeDiarizer()
    recognizer = FakeRecognizer(oom_on=[0, 1]) if case == "oom" else FakeRecognizer()
    env = Env(tmp_path, diarizer=diarizer, recognizer=recognizer)
    job = env.add_job(audio=b"\x00not audio" * 500 if case == "decode" else None)
    env.run()
    # фейковый instudy возвращает работу в очередь до 3 неудач — каждая попытка шлёт тот же fail
    assert job.fails
    assert {(f["reason"], f["retryable"]) for f in job.fails} == {(reason, retryable)}
    assert all(str(tmp_path) not in f["message"] for f in job.fails)
    assert env.work_items() == []
    state = read_state(env.data)
    assert state["counters"]["failed"] == len(job.fails) and state["last_error"]["code"] == reason
    env.assert_state_clean()


def test_result_5xx_twice_then_202(tmp_path: Path) -> None:
    env = Env(tmp_path)
    job = env.add_job()
    env.fake.fail_next("POST result", 500, 503)
    env.run()
    assert len(env.fake.requests("POST", "/result")) == 3
    assert job.status == "done" and job.fails == []
    assert env.clock.waits[:2] == [5.0, 10.0]  # экспоненциальная задержка
    assert [p.name for p in (env.data / "results").iterdir()] == ["1"]
    assert sorted(p.name for p in (env.data / "results" / "1").iterdir()) == [
        "meta.json",
        "transcript.txt",
        "transcript.vtt",
    ]
    assert env.work_items() == []


def test_rate_limited_next_respects_retry_after(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.fake.rate_limited = 1
    env.run()
    assert env.clock.waits[0] == 17.0  # Retry-After фейка (429 пришёл на health при старте)
    assert len(env.fake.requests("GET", "/health")) == 2
    assert len(env.fake.requests("GET", "/jobs/next")) == 1


def test_pause_and_resume(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.data.mkdir(parents=True)
    assert cli.cmd_pause(env.data) == 0
    seen: list[tuple[int, str | None]] = []

    def on_wait(seconds: float) -> None:
        seen.append((len(env.fake.requests("GET", "/jobs/next")), read_state(env.data)["reason"]))
        if len(seen) == 1:
            cli.cmd_resume(env.data)
        else:
            env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    assert seen[0] == (0, "pause")
    assert seen[1][0] == 1


def test_low_disk_pauses(tmp_path: Path) -> None:
    env = Env(tmp_path, free_gb=1.0)
    env.add_job()
    snapshots: list[dict] = []

    def on_wait(seconds: float) -> None:
        snapshots.append(read_state(env.data))
        env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    assert env.fake.requests("GET", "/jobs/next") == []
    assert snapshots[0]["status"] == "paused" and snapshots[0]["reason"] == "low_disk"


def test_unauthorized_on_start_waits_10_minutes(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.fake.token = "другой-токен-instudy"
    env.add_job()
    seen: list[tuple[float, int, str]] = []

    def on_wait(seconds: float) -> None:
        state = read_state(env.data)
        seen.append((seconds, len(env.fake.requests("GET", "/jobs/next")), state["status"]))
        if len(seen) == 2:
            env.fake.token = TOKEN  # токен перевыпустили
        if seconds == POLL:
            env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    assert seen[0] == (600.0, 0, "error") and seen[1] == (600.0, 0, "error")
    assert len(env.fake.requests("GET", "/health")) == 3
    assert env.fake.jobs[1].status == "done"  # после восстановления токена выдача пошла
    assert read_state(env.data)["last_error"]["code"] == "unauthorized"
    env.assert_state_clean()


def test_invalid_job_sends_fail(tmp_path: Path) -> None:
    env = Env(tmp_path, max_audio_bytes=1000)
    job = env.add_job()
    env.run()
    assert job.fails and {(f["reason"], f["retryable"]) for f in job.fails} == {("invalid_job", False)}
    assert env.fake.requests("GET", "/audio/") == []  # невалидная работа не скачивается


def test_startup_cleans_work_and_old_results(tmp_path: Path) -> None:
    env = Env(tmp_path)
    leftover = env.data / "work" / "99"
    leftover.mkdir(parents=True)
    (leftover / "audio.m4a").write_bytes(b"x")
    old, fresh = env.data / "results" / "5", env.data / "results" / "6"
    for d in (old, fresh):
        d.mkdir(parents=True)
    old_ts = (env.clock.now() - timedelta(days=20)).timestamp()
    os.utime(old, (old_ts, old_ts))
    fresh_ts = (env.clock.now() - timedelta(days=2)).timestamp()
    os.utime(fresh, (fresh_ts, fresh_ts))
    env.run()
    assert env.work_items() == []
    assert not old.exists() and fresh.exists()


def test_results_cleanup_runs_daily(tmp_path: Path) -> None:
    env = Env(tmp_path)
    calls: list[datetime] = []
    original = env.daemon.cleanup_results

    def counting() -> int:
        calls.append(env.clock.now())
        return original()

    env.daemon.cleanup_results = counting  # type: ignore[method-assign]
    env.clock.limit = 10_000
    env.clock.on_wait = lambda s: (
        env.daemon.request_stop() if env.clock.now() - calls[0] > timedelta(days=2, hours=1) else None
    )
    env.run()
    # опрос раз в 77 с, двое суток — а чистка была 3 раза (старт, +1 сутки, +2 суток)
    assert len(calls) == 3


def test_shutdown_mid_asr_in_process(tmp_path: Path) -> None:
    class Killer(FakeRecognizer):
        def transcribe_batch(self, chunks):
            out = super().transcribe_batch(chunks)
            if self.calls == 2:
                os.kill(os.getpid(), signal.SIGTERM)
                ctx = env.daemon._current
                assert ctx is not None and ctx.cancel.wait(5), "сигнал не выставил отмену"
            return out

    recognizer = Killer()
    env = Env(tmp_path, diarizer=FakeDiarizer(spaced_turns(20)), recognizer=recognizer)
    job = env.add_job(seconds=40.0)
    previous = env.daemon.install_signal_handlers()
    try:
        started = time.monotonic()
        assert env.run() == 0
        assert time.monotonic() - started < 30
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert [(f["reason"], f["retryable"]) for f in job.fails] == [("shutdown", True)]
    assert recognizer.calls < 20
    assert env.work_items() == []
    assert read_state(env.data)["counters"]["failed"] == 1
    env.assert_state_clean()


def test_stuck_stage_is_abandoned_after_grace(tmp_path: Path, monkeypatch) -> None:
    release = threading.Event()

    class Stuck(FakeDiarizer):
        def diarize(self, wav_path):
            threading.Timer(0.1, env.daemon.request_stop).start()
            release.wait(10)
            return super().diarize(wav_path)

    monkeypatch.setattr(daemon_mod, "SHUTDOWN_GRACE_SEC", 0.3)
    env = Env(tmp_path, diarizer=Stuck())
    job = env.add_job()
    started = time.monotonic()
    env.run()
    assert time.monotonic() - started < 5
    release.set()
    assert env.daemon.abandoned
    assert [f["reason"] for f in job.fails] == ["shutdown"]
    assert env.work_items() == []


def _spawn(tmp_path: Path, mode: str) -> subprocess.Popen[str]:
    data = tmp_path / "data"
    data.mkdir()
    return subprocess.Popen(
        [sys.executable, "-m", "tests.daemon_harness", str(data), mode],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _wait_ready(proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip() == "READY":
            return
    raise AssertionError("демон не дошёл до нужного этапа")


def _result(proc: subprocess.Popen[str]) -> dict:
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError("нет строки RESULT")


def test_sigterm_process_exits_with_fail_shutdown(tmp_path: Path) -> None:
    proc = _spawn(tmp_path, "asr")
    try:
        _wait_ready(proc)
        started = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        result = _result(proc)
        assert proc.wait(timeout=30) == 0
        assert time.monotonic() - started < 30
    finally:
        proc.kill()
    assert [(f["reason"], f["retryable"]) for f in result["fails"]] == [("shutdown", True)]
    assert result["work"] == []


def test_sigterm_stuck_stage_exits_within_grace(tmp_path: Path) -> None:
    proc = _spawn(tmp_path, "stuck")
    try:
        _wait_ready(proc)
        started = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        result = _result(proc)
        assert proc.wait(timeout=30) == 0
        assert time.monotonic() - started < 30
    finally:
        proc.kill()
    assert result["abandoned"] is True
    assert [f["reason"] for f in result["fails"]] == ["shutdown"]
    assert result["work"] == []


def test_second_signal_exits_immediately(tmp_path: Path) -> None:
    proc = _spawn(tmp_path, "stuck")
    try:
        _wait_ready(proc)
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=5) == 128 + signal.SIGTERM
    finally:
        proc.kill()


# ------------------------------------------------------------------------------ state и CLI


def test_state_store_atomic_and_fields(tmp_path: Path) -> None:
    now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    store = StateStore(tmp_path, now=lambda: now)
    store.start_job(5, 412, datetime(2026, 9, 29).date())
    store.progress("asr", 0.5)
    state = read_state(tmp_path)
    assert state["status"] == "working"
    assert state["job"] == {
        "id": 5,
        "dayId": 412,
        "date": "2026-09-29",
        "stage": "asr",
        "progress": 0.5,
        "started_at": "2026-10-22T12:00:00Z",
    }
    assert set(state) == {
        "pid",
        "version",
        "started_at",
        "heartbeat_at",
        "status",
        "reason",
        "next_check_at",
        "instudy_paused",
        "job",
        "counters",
        "last_error",
    }
    store.finish_job("done")
    assert read_state(tmp_path)["counters"] == {"done": 1, "failed": 0, "skipped": 0, "cancelled": 0}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # временных файлов не осталось


def test_cli_health(tmp_path: Path) -> None:
    written = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    StateStore(tmp_path, now=lambda: written).touch()
    assert cli.cmd_health(tmp_path, now=written + timedelta(seconds=60)) == 0
    assert cli.cmd_health(tmp_path, now=written + timedelta(seconds=400)) == 1
    assert cli.cmd_health(tmp_path / "nope", now=written) == 1


def test_cli_health_entrypoint(tmp_path: Path, monkeypatch) -> None:
    StateStore(tmp_path).touch()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert cli.main(["health"]) == 0


def test_cli_status(tmp_path: Path, capsys) -> None:
    now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    store = StateStore(tmp_path, now=lambda: now)
    store.start_job(731, 412, datetime(2026, 9, 29).date())
    store.progress("asr", 0.42)
    (tmp_path / "PAUSE").touch()
    assert cli.cmd_status(tmp_path, now=now + timedelta(seconds=10)) == 0
    out = capsys.readouterr().out
    assert "жив" in out and "working" in out and "#731" in out and "42 %" in out and "включена" in out
    assert cli.cmd_status(tmp_path / "nope") == 1


def test_cli_pause_resume(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert cli.main(["pause"]) == 0 and (tmp_path / "PAUSE").exists()
    assert cli.main(["resume"]) == 0 and not (tmp_path / "PAUSE").exists()


def test_cli_check_without_config(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "INSTUDY_API_URL" in out and "nvidia-smi не найден" in out


def test_daemon_without_config_exits_2() -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("INSTUDY_API_URL", "SERVICE_TOKEN")}
    res = subprocess.run(
        [sys.executable, "-m", "transkrib"], capture_output=True, text=True, timeout=60, check=False, env=env, cwd=ROOT
    )
    assert res.returncode == 2
    assert "INSTUDY_API_URL" in res.stdout + res.stderr


def test_daemon_modules_do_not_import_torch() -> None:
    code = (
        "import sys, transkrib.daemon, transkrib.state, transkrib.__main__;"
        "heavy = [m for m in ('torch', 'gigaam', 'pyannote') if m in sys.modules];"
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stdout + res.stderr
