from __future__ import annotations

import logging
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.audio_gen import silence, tone, write_wav
from transkrib import audio
from transkrib.types import PipelineCancelled, PipelineError


def lecture_like(seconds: float, rate: int) -> np.ndarray:
    """Чередование тона и тишины по 5 с."""
    parts, total = [], 0.0
    while total < seconds:
        step = min(5.0, seconds - total)
        parts.append(tone(step, rate) if int(total / 5) % 2 == 0 else silence(step, rate))
        total += step
    return np.concatenate(parts)


def wav_info(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as wf:
        return wf.getnchannels(), wf.getframerate(), wf.getsampwidth(), wf.getnframes() / wf.getframerate()


def test_to_wav16k_mono_from_stereo_44k(tmp_path: Path) -> None:
    job = tmp_path / "17"
    job.mkdir()
    src = write_wav(job / "source.wav", lecture_like(90.0, 44_100), rate=44_100, channels=2)
    dst = audio.to_wav16k_mono(src, job / "audio.wav")
    channels, rate, width, duration = wav_info(dst)
    assert (channels, rate, width) == (1, 16_000, 2)
    assert abs(duration - 90.0) <= 0.05
    assert not (job / "audio.wav.part").exists()
    assert abs(audio.probe_duration(dst) - 90.0) <= 0.05


def test_to_wav16k_mono_from_m4a(tmp_path: Path) -> None:
    raw = write_wav(tmp_path / "raw.wav", lecture_like(90.0, 44_100), rate=44_100, channels=2)
    m4a = tmp_path / "audio.m4a"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw), "-c:a", "aac", "-b:a", "64k", str(m4a)]
    subprocess.run(
        cmd,
        check=True,
        timeout=120,
    )
    dst = audio.to_wav16k_mono(m4a, tmp_path / "audio.wav")
    channels, rate, _, duration = wav_info(dst)
    assert (channels, rate) == (1, 16_000)
    assert abs(duration - 90.0) <= 0.05
    assert abs(audio.probe_duration(m4a) - 90.0) <= 0.05


def test_broken_file_is_decode_failed_and_log_has_no_user_paths(tmp_path: Path, caplog) -> None:
    job = tmp_path / "secret-home" / "42"
    job.mkdir(parents=True)
    src = job / "audio.m4a"
    src.write_bytes(b"\x00\x01not an audio file" * 100)
    with caplog.at_level(logging.WARNING), pytest.raises(PipelineError) as err:
        audio.to_wav16k_mono(src, job / "audio.wav")
    assert err.value.reason == "decode_failed"
    assert str(tmp_path) not in err.value.message
    assert not (job / "audio.wav").exists() and not (job / "audio.wav.part").exists()
    records = [r for r in caplog.records if r.name == "transkrib.audio"]
    assert records
    for record in records:
        dumped = repr(vars(record))
        assert "secret-home" not in dumped and str(tmp_path) not in dumped
        assert record.job == "42"
    assert any("42/audio.m4a" in str(getattr(r, "stderr", "")) for r in records)


def test_missing_file_is_decode_failed(tmp_path: Path) -> None:
    with pytest.raises(PipelineError) as err:
        audio.to_wav16k_mono(tmp_path / "nope.m4a", tmp_path / "audio.wav")
    assert err.value.reason == "decode_failed"


def test_probe_duration_broken_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"garbage" * 10)
    with pytest.raises(PipelineError) as err:
        audio.probe_duration(bad)
    assert err.value.reason == "decode_failed"


def _slow_cmd(pid_file: Path) -> list[str]:
    # Процесс с потомком: проверяем, что убивается вся группа, а не только родитель.
    return ["sh", "-c", f"sleep 30 & echo $! > {pid_file}; wait"]


def _gone(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")


def _wait_for(path: Path) -> int:
    for _ in range(100):
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError("потомок не стартовал")


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="нужен /proc")
def test_cancel_kills_process_group(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "child.pid"
    monkeypatch.setattr(audio, "_ffmpeg_cmd", lambda src, dst: _slow_cmd(pid_file))
    src = write_wav(tmp_path / "in.wav", silence(1.0))
    cancel = threading.Event()
    threading.Thread(target=lambda: (_wait_for(pid_file), cancel.set()), daemon=True).start()
    started = time.monotonic()
    with pytest.raises(PipelineCancelled):
        audio.to_wav16k_mono(src, tmp_path / "audio.wav", cancel=cancel)
    assert time.monotonic() - started < 10
    child = _wait_for(pid_file)
    for _ in range(50):
        if _gone(child):
            break
        time.sleep(0.05)
    assert _gone(child)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="нужен /proc")
def test_timeout_kills_process_group(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "child.pid"
    monkeypatch.setattr(audio, "_ffmpeg_cmd", lambda src, dst: _slow_cmd(pid_file))
    src = write_wav(tmp_path / "in.wav", silence(1.0))
    with pytest.raises(PipelineError) as err:
        audio.to_wav16k_mono(src, tmp_path / "audio.wav", timeout=1.0)
    assert err.value.reason == "decode_failed"
    child = _wait_for(pid_file)
    for _ in range(50):
        if _gone(child):
            break
        time.sleep(0.05)
    assert _gone(child)


def test_load_wav_roundtrip(tmp_path: Path) -> None:
    signal = tone(2.0, amp=0.25)
    samples = audio.load_wav(write_wav(tmp_path / "a.wav", signal))
    assert samples.dtype == np.float32
    assert samples.size == signal.size
    assert np.max(np.abs(samples - signal)) < 1e-3


def test_load_wav_rejects_other_format(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", tone(1.0, 44_100), rate=44_100)
    with pytest.raises(PipelineError) as err:
        audio.load_wav(path)
    assert err.value.reason == "decode_failed"
    assert not audio.is_wav16k_mono(path)


def test_rms_frames() -> None:
    signal = np.concatenate([np.full(480, 0.5, np.float32), np.zeros(480, np.float32), np.full(100, 1.0, np.float32)])
    rms = audio.rms_frames(signal, frame_ms=30)
    assert rms.shape == (3,)
    np.testing.assert_allclose(rms, [0.5, 0.0, 1.0], atol=1e-6)
    assert audio.rms_frames(np.zeros(0, np.float32)).size == 0
