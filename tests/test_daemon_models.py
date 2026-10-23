"""T-6a: неисправность сервера не сжигает попытки дней — проверка загрузки моделей и предохранитель."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fakes import FakeDiarizer, FakeRecognizer
from tests.test_daemon import POLL, Env, spaced_turns
from transkrib import cli
from transkrib.asr import ModelError
from transkrib.daemon import MODELS_RETRY_SEC
from transkrib.state import read_state


class FlakyLoadRecognizer(FakeRecognizer):
    """load() падает, пока broken=True (как CUDA с «Driver/library version mismatch»)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.broken = False
        self.load_attempts = 0

    def load(self) -> None:
        self.load_attempts += 1
        if self.broken:
            raise ModelError("CUDA недоступна: GigaAM работает только на GPU")
        super().load()


class AlwaysFail(FlakyLoadRecognizer):
    """Грузится, но падает на каждом куске — любая работа заканчивается model_error."""

    def transcribe_batch(self, chunks):
        self.calls += len(chunks)
        raise RuntimeError("сбой модели")


def _events(env: Env) -> list[str]:
    """Что демон сообщил instudy, по порядку: код fail или «result» (фейк выдаёт работу снова после fail)."""
    out = []
    for r in env.fake.log:
        if r.method == "POST" and r.path.endswith("/fail"):
            out.append(json.loads(r.body)["reason"])
        elif r.method == "POST" and r.path.endswith("/result"):
            out.append("result")
    return out


def _next_calls(env: Env) -> int:
    return len(env.fake.requests("GET", "/jobs/next"))


def test_models_not_loading_on_start_blocks_until_reload(tmp_path: Path) -> None:
    recognizer = FlakyLoadRecognizer()
    recognizer.broken = True
    env = Env(tmp_path, recognizer=recognizer)
    job = env.add_job()
    seen: list[tuple[float, int, dict]] = []

    def on_wait(seconds: float) -> None:
        seen.append((seconds, _next_calls(env), read_state(env.data)))
        if len(seen) == 2:
            recognizer.broken = False  # драйвер починили (перезагрузка)
        if seconds == POLL:
            env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    assert [s for s, _, _ in seen[:2]] == [MODELS_RETRY_SEC, MODELS_RETRY_SEC]
    for _, next_calls, state in seen[:2]:
        assert next_calls == 0
        assert state["status"] == "error" and state["reason"] == "models_load"
        assert state["next_check_at"] is not None
    assert env.fake.requests("GET", "/health") != []  # health — только после моделей
    assert job.status == "done" and job.fails == []
    assert recognizer.load_attempts == 4  # 3 проверки на старте и повторах + 1 загрузка в работе
    assert not recognizer.loaded and not env.diarizer.loaded  # KEEP_MODELS_LOADED=false: проверка не держит VRAM
    env.assert_state_clean()


def test_models_check_keeps_models_loaded_when_configured(tmp_path: Path) -> None:
    diarizer, recognizer = FakeDiarizer(), FakeRecognizer()
    env = Env(tmp_path, diarizer=diarizer, recognizer=recognizer, keep_models_loaded=True)
    env.daemon.clock.on_wait = lambda s: env.daemon.request_stop()
    env.daemon._startup()
    assert diarizer.loaded and recognizer.loaded


def test_model_error_on_load_mid_job_sends_shutdown(tmp_path: Path) -> None:
    recognizer = FlakyLoadRecognizer()
    env = Env(tmp_path, recognizer=recognizer)
    first, second = env.add_job(1), env.add_job(2)
    # Проверка на старте прошла; затем драйвер «отвалился» — диаризация уже была, ASR не грузится.
    original_diarize = env.diarizer.diarize

    def diarize_and_break(path):
        recognizer.broken = True
        return original_diarize(path)

    env.diarizer.diarize = diarize_and_break  # type: ignore[method-assign]
    seen: list[tuple[float, int, dict]] = []

    def on_wait(seconds: float) -> None:
        seen.append((seconds, _next_calls(env), read_state(env.data)))
        if seconds == MODELS_RETRY_SEC and len(seen) == 2:
            recognizer.broken = False
            env.diarizer.diarize = original_diarize  # type: ignore[method-assign]
        if seconds == POLL:
            env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    assert [(f["reason"], f["retryable"]) for f in first.fails] == [("shutdown", True)]
    assert "/" not in first.fails[0]["message"]
    # После сбоя — две проверки по 600 с, next_job за это время не вызывался.
    assert seen[0][0] == MODELS_RETRY_SEC and seen[1][0] == MODELS_RETRY_SEC
    assert seen[0][1] == seen[1][1] == 1
    assert seen[0][2]["status"] == "error" and seen[0][2]["reason"] == "models_load"
    assert seen[0][2]["last_error"]["code"] == "models_load"
    # Работа 1 вернулась в очередь без засчитанной попытки (у фейка — снова queued) и сделана вместе с работой 2.
    assert first.status == "done" and second.status == "done"
    assert second.fails == []
    env.assert_state_clean()


def test_breaker_after_three_model_errors(tmp_path: Path) -> None:
    recognizer = AlwaysFail()
    env = Env(tmp_path, diarizer=FakeDiarizer(spaced_turns(4)), recognizer=recognizer)
    jobs = [env.add_job(i) for i in range(1, 5)]
    seen: list[tuple[float, int, dict]] = []

    def on_wait(seconds: float) -> None:
        seen.append((seconds, _next_calls(env), read_state(env.data)))
        if seconds == MODELS_RETRY_SEC and len([s for s in seen if s[0] == MODELS_RETRY_SEC]) == 1:
            env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    assert _events(env) == ["model_error"] * 3
    assert _next_calls(env) == 3  # четвёртая работа не взята
    assert any(j.status == "queued" for j in jobs)
    seconds, _, state = seen[-1]
    assert seconds == MODELS_RETRY_SEC
    assert state["status"] == "error" and state["reason"] == "model_breaker" and state["next_check_at"]
    env.assert_state_clean()


def test_breaker_released_by_successful_reload(tmp_path: Path) -> None:
    class FailsThreeJobs(FlakyLoadRecognizer):
        def __init__(self) -> None:
            super().__init__()
            self.jobs = 0

        def transcribe_batch(self, chunks):
            if self.jobs < 3:
                self.calls += len(chunks)
                raise RuntimeError("сбой модели")
            return super().transcribe_batch(chunks)

    recognizer = FailsThreeJobs()
    env = Env(tmp_path, diarizer=FakeDiarizer(spaced_turns(4)), recognizer=recognizer)
    jobs = [env.add_job(i) for i in range(1, 5)]
    original_process = env.daemon.process

    def counting_process(job):
        try:
            return original_process(job)
        finally:
            recognizer.jobs += 1

    env.daemon.process = counting_process  # type: ignore[method-assign]
    env.run()
    assert _events(env)[:4] == ["model_error"] * 3 + ["result"]  # после успешной повторной загрузки выдача пошла
    assert any(j.status == "done" for j in jobs)
    assert env.daemon._model_failures == 0
    assert MODELS_RETRY_SEC in env.clock.waits


@pytest.mark.parametrize("reason_case", ["model_error", "oom"])
def test_breaker_reset_by_202(tmp_path: Path, reason_case: str) -> None:
    class Pattern(FakeRecognizer):
        """Работы: сбой, сбой, успех, сбой — по одной на каждую загрузку."""

        def __init__(self) -> None:
            super().__init__()
            self.pattern = [True, True, False, True]

        def load(self) -> None:
            super().load()
            self.fail_now = self.pattern.pop(0) if self.pattern else False

        def transcribe_batch(self, chunks):
            if getattr(self, "fail_now", False):
                self.calls += len(chunks)
                if reason_case == "oom":
                    raise RuntimeError("CUDA out of memory")
                raise RuntimeError("сбой модели")
            return super().transcribe_batch(chunks)

    recognizer = Pattern()
    env = Env(tmp_path, diarizer=FakeDiarizer(spaced_turns(2)), recognizer=recognizer)
    # первая загрузка — проверка на старте: она не работа, сценарий начинаем с неё «успехом»
    recognizer.pattern.insert(0, False)
    jobs = [env.add_job(i) for i in range(1, 5)]
    env.run()
    events = _events(env)
    assert events[:4] == [reason_case, reason_case, "result", reason_case]
    assert set(events[4:]) <= {"result"} and len(jobs) == 4
    assert env.daemon._model_block is None
    assert MODELS_RETRY_SEC not in env.clock.waits  # третья ошибка после 202 — не третья подряд


def test_decode_and_no_speech_do_not_trip_breaker(tmp_path: Path) -> None:
    class NoSpeechForSome(FakeDiarizer):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def diarize(self, wav_path):
            self.n += 1
            return [] if self.n % 2 else super().diarize(wav_path)

    env = Env(tmp_path, diarizer=NoSpeechForSome())
    bad = b"\x00not audio" * 500
    # Чередуем: no_speech, decode_failed, no_speech, decode_failed, затем нормальная работа.
    jobs = []
    for i in range(1, 5):
        jobs.append(env.add_job(i, audio=bad if i % 2 == 0 else None))
    last = env.add_job(5)
    env.diarizer.n = 0
    env.run()
    fails = [e for e in _events(env) if e != "result"]
    assert len(fails) >= 3 and set(fails) == {"no_speech", "decode_failed"}
    assert MODELS_RETRY_SEC not in env.clock.waits
    assert env.daemon._model_block is None and env.daemon._model_failures == 0
    assert len(jobs) == 4 and last.status in ("done", "failed")


def test_cli_status_shows_block_and_next_check(tmp_path: Path, capsys) -> None:
    recognizer = FlakyLoadRecognizer()
    recognizer.broken = True
    env = Env(tmp_path, recognizer=recognizer)
    outputs: list[str] = []

    def on_wait(seconds: float) -> None:
        cli.cmd_status(env.data, now=env.clock.now())
        outputs.append(capsys.readouterr().out)
        env.daemon.request_stop()

    env.clock.on_wait = on_wait
    env.run()
    assert "models_load" in outputs[0] and "Следующая проверка моделей" in outputs[0]
