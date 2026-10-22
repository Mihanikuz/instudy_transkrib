"""Генерация коротких тестовых сигналов (тон/тишина) — бинарники в репозиторий не коммитятся."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def tone(seconds: float, rate: int = 16_000, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, rate: int = 16_000) -> np.ndarray:
    return np.zeros(round(seconds * rate), dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int = 16_000, channels: int = 1) -> Path:
    data = np.clip(samples, -1.0, 1.0)
    pcm = (data * 32767).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return path
