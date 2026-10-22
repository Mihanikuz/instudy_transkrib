from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from tests.audio_gen import tone, write_wav
from tests.fakes import FakeDiarizer, FakeRecognizer
from transkrib import cli, pipeline
from transkrib.config import load_settings
from transkrib.diarize import Turn
from transkrib.render import render_txt, render_vtt, validate_vtt
from transkrib.types import PipelineCancelled, PipelineError

ENGINE = "GigaAM v3 e2e RNNT, pyannote community-1"


def spaced_turns(n: int, speakers: tuple[str, ...] = ("SPEAKER_00",)) -> list[Turn]:
    """n отрезков по 1 с через 1 с паузы — не склеиваются, каждый — отдельный кусок."""
    return [Turn(2.0 * i, 2.0 * i + 1.0, speakers[i % len(speakers)]) for i in range(n)]


def make_wav(tmp_path: Path, seconds: float) -> Path:
    job = tmp_path / "7"
    job.mkdir(exist_ok=True)
    return write_wav(job / "audio.wav", tone(seconds))


class Progress:
    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []

    def __call__(self, stage: str, fraction: float) -> None:
        self.events.append((stage, fraction))


def run(path: Path, diarizer, recognizer, *, cancel=None, progress=None, **settings):
    return pipeline.run(
        path,
        diarizer=diarizer,
        recognizer=recognizer,
        cancel=cancel or threading.Event(),
        progress=progress or Progress(),
        settings=load_settings(**settings),
    )


def test_pipeline_on_fakes_renders(tmp_path: Path) -> None:
    wav = make_wav(tmp_path, 80.0)
    turns = [
        Turn(1.0, 9.0, "SPEAKER_02"),
        Turn(9.3, 15.0, "SPEAKER_02"),  # склеится с предыдущим
        Turn(15.5, 16.0, "SPEAKER_00"),
        Turn(16.5, 16.6, "SPEAKER_01"),  # короче 0,3 с — отброшен
        Turn(17.0, 75.0, "SPEAKER_00"),  # 58 с — разрежется на 3 куска
    ]
    diarizer, recognizer, progress = FakeDiarizer(turns), FakeRecognizer(), Progress()
    result = run(wav, diarizer, recognizer, progress=progress)

    assert result.skipped_chunks == 0
    assert result.speakers == 2
    assert [s.speaker for s in result.segments] == [1, 2, 2, 2, 2]
    assert result.segments[0].start == 1.0 and result.segments[0].end == 15.0
    assert all(s.end - s.start <= 22.0 for s in result.segments)
    assert abs(result.duration_sec - 80.0) < 0.01
    assert result.chunks == recognizer.calls == 5
    assert diarizer.unload_count == 1 and recognizer.unload_count == 1  # KEEP_MODELS_LOADED=false

    vtt = render_vtt(result.segments)
    validate_vtt(vtt, result.segments)
    assert "<v Спикер 1>" in vtt and "<v Спикер 2>" in vtt
    txt = render_txt(result.segments, discipline="Тест", date=date(2026, 9, 29), engine_line=ENGINE)
    assert "[00:00:01] Спикер 1: фраза 1" in txt

    stages = [s for s, _ in progress.events]
    assert stages[0] == "prepare" and stages[-1] == "done"
    assert stages.index("diarize") < stages.index("segment") < stages.index("asr")
    asr = [f for s, f in progress.events if s == "asr"]
    assert asr[-1] == 1.0 and asr == sorted(asr)
    assert set(result.stage_sec) == {"prepare", "diarize", "segment", "asr"}


def test_pipeline_converts_non_16k_input(tmp_path: Path) -> None:
    job = tmp_path / "8"
    job.mkdir()
    src = write_wav(job / "audio.m4a.wav", tone(10.0, 44_100), rate=44_100, channels=2)
    result = run(src, FakeDiarizer(), FakeRecognizer())
    assert (job / "audio.wav").is_file()
    assert abs(result.duration_sec - 10.0) <= 0.05
    assert len(result.segments) == 1 and result.segments[0].speaker == 1


def test_keep_models_loaded(tmp_path: Path) -> None:
    diarizer, recognizer = FakeDiarizer(), FakeRecognizer()
    run(make_wav(tmp_path, 5.0), diarizer, recognizer, keep_models_loaded=True)
    assert diarizer.loaded and recognizer.loaded
    assert diarizer.unload_count == recognizer.unload_count == 0


def test_cancel_during_asr(tmp_path: Path) -> None:
    wav = make_wav(tmp_path, 40.0)
    recognizer, cancel = FakeRecognizer(), threading.Event()

    def progress(stage: str, fraction: float) -> None:
        if stage == "asr" and fraction >= 0.25:
            cancel.set()

    with pytest.raises(PipelineCancelled):
        run(wav, FakeDiarizer(spaced_turns(20)), recognizer, cancel=cancel, progress=progress)
    assert recognizer.calls == 5
    assert not recognizer.loaded  # выгружен и при отмене


def test_cancel_before_start(tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()
    diarizer = FakeDiarizer()
    with pytest.raises(PipelineCancelled):
        run(make_wav(tmp_path, 5.0), diarizer, FakeRecognizer(), cancel=cancel)
    assert diarizer.calls == []


def test_empty_diarization_is_no_speech(tmp_path: Path) -> None:
    recognizer = FakeRecognizer()
    with pytest.raises(PipelineError) as err:
        run(make_wav(tmp_path, 5.0), FakeDiarizer([]), recognizer)
    assert err.value.reason == "no_speech"
    assert recognizer.calls == 0


def test_only_short_turns_is_no_speech(tmp_path: Path) -> None:
    with pytest.raises(PipelineError) as err:
        run(make_wav(tmp_path, 5.0), FakeDiarizer([Turn(1.0, 1.1, "SPEAKER_00")]), FakeRecognizer())
    assert err.value.reason == "no_speech"


def test_ten_percent_failures_is_model_error(tmp_path: Path) -> None:
    recognizer = FakeRecognizer(fail_on=[3, 11])
    with pytest.raises(PipelineError) as err:
        run(make_wav(tmp_path, 40.0), FakeDiarizer(spaced_turns(20)), recognizer)
    assert err.value.reason == "model_error"
    assert not recognizer.loaded


def test_two_percent_failures_succeeds(tmp_path: Path) -> None:
    recognizer = FakeRecognizer(fail_on=[17])
    result = run(make_wav(tmp_path, 100.0), FakeDiarizer(spaced_turns(50)), recognizer)
    assert result.skipped_chunks == 1
    assert len(result.segments) == 49
    assert recognizer.calls == 50


def test_oom_once_retries(tmp_path: Path) -> None:
    recognizer = FakeRecognizer(oom_on=[3])
    result = run(make_wav(tmp_path, 20.0), FakeDiarizer(spaced_turns(10)), recognizer)
    assert result.skipped_chunks == 0
    assert len(result.segments) == 10
    assert recognizer.calls == 11


def test_oom_twice_is_oom(tmp_path: Path) -> None:
    recognizer = FakeRecognizer(oom_on=[3, 4])
    with pytest.raises(PipelineError) as err:
        run(make_wav(tmp_path, 20.0), FakeDiarizer(spaced_turns(10)), recognizer)
    assert err.value.reason == "oom"
    assert recognizer.calls == 5


def test_broken_input_is_decode_failed(tmp_path: Path) -> None:
    job = tmp_path / "9"
    job.mkdir()
    bad = job / "audio.m4a"
    bad.write_bytes(b"\x00garbage" * 200)
    diarizer = FakeDiarizer()
    with pytest.raises(PipelineError) as err:
        run(bad, diarizer, FakeRecognizer())
    assert err.value.reason == "decode_failed"
    assert diarizer.calls == []


class _FailingDiarizer(FakeDiarizer):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc

    def diarize(self, wav_path: Path) -> list[Turn]:
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "reason"),
    [(RuntimeError("CUDA out of memory"), "oom"), (ValueError("boom /home/user/secret"), "model_error")],
)
def test_diarization_errors(tmp_path: Path, exc: Exception, reason: str) -> None:
    diarizer = _FailingDiarizer(exc)
    with pytest.raises(PipelineError) as err:
        run(make_wav(tmp_path, 5.0), diarizer, FakeRecognizer())
    assert err.value.reason == reason
    assert "/home" not in err.value.message
    assert diarizer.unload_count == 1


class _SilentSpeakerRecognizer(FakeRecognizer):
    """Второй кусок (единственный у SPEAKER_05) распознаётся пустым."""

    def transcribe_batch(self, chunks):
        out = super().transcribe_batch(chunks)
        return ["" if self.calls == 2 else t for t in out]


def test_speaker_without_text_leaves_no_gap(tmp_path: Path) -> None:
    turns = [Turn(0.0, 1.0, "SPEAKER_01"), Turn(2.0, 3.0, "SPEAKER_05"), Turn(4.0, 5.0, "SPEAKER_00")]
    result = run(make_wav(tmp_path, 6.0), FakeDiarizer(turns), _SilentSpeakerRecognizer())
    assert [s.speaker for s in result.segments] == [1, 2]
    assert result.speakers == 2
    render_vtt(result.segments)


def test_all_empty_text_is_no_speech(tmp_path: Path) -> None:
    class Mute(FakeRecognizer):
        def transcribe_batch(self, chunks):
            super().transcribe_batch(chunks)
            return [""] * len(chunks)

    with pytest.raises(PipelineError) as err:
        run(make_wav(tmp_path, 5.0), FakeDiarizer(), Mute())
    assert err.value.reason == "no_speech"


def test_logs_have_no_transcript_text(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        run(make_wav(tmp_path, 20.0), FakeDiarizer(spaced_turns(10)), FakeRecognizer(fail_on=[], oom_on=[2]))
    dumped = " ".join(repr(vars(r)) for r in caplog.records)
    assert "фраза" not in dumped
    assert str(tmp_path) not in dumped


def test_pipeline_modules_do_not_import_torch() -> None:
    code = (
        "import sys, transkrib.audio, transkrib.segment, transkrib.pipeline;"
        "heavy = [m for m in ('torch', 'gigaam', 'pyannote') if m in sys.modules];"
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stdout + res.stderr


def test_empty_cuda_cache_without_torch(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    pipeline.empty_cuda_cache()


def test_cli_run_file_with_fakes(tmp_path: Path, monkeypatch, capsys) -> None:
    src = write_wav(tmp_path / "lecture.wav", tone(30.0, 44_100), rate=44_100)
    turns = [Turn(0.5, 10.0, "SPEAKER_01"), Turn(11.0, 29.0, "SPEAKER_00")]
    monkeypatch.setattr(cli, "make_models", lambda settings: (FakeDiarizer(turns), FakeRecognizer()))
    out = tmp_path / "out"
    assert cli.main(["run-file", str(src), "--out", str(out)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["speakers"] == 2 and summary["segments"] == 2 and summary["skippedChunks"] == 0
    assert set(summary["stageSec"]) == {"prepare", "diarize", "segment", "asr"}
    vtt = (out / "lecture.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT\n")
    assert "Спикер 2: фраза 2" in (out / "lecture.txt").read_text(encoding="utf-8")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["asr"] == "fake-asr" and meta["audioDurationSec"] == 30
    assert sorted(p.name for p in out.iterdir()) == ["lecture.txt", "lecture.vtt", "meta.json"]  # WAV убран


def test_cli_run_file_reports_pipeline_error(tmp_path: Path, monkeypatch, capsys) -> None:
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(np.zeros(100, np.uint8).tobytes())
    monkeypatch.setattr(cli, "make_models", lambda settings: (FakeDiarizer(), FakeRecognizer()))
    assert cli.main(["run-file", str(bad), "--out", str(tmp_path / "out")]) == 1
    assert "decode_failed" in capsys.readouterr().err
