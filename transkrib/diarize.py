"""Диаризация: интерфейс Diarizer и реализация на pyannote speaker-diarization-community-1 (только CUDA, офлайн).

Берётся exclusive_speaker_diarization (без перекрытий — проще резать и подписывать реплики); запасной путь —
speaker_diarization. Аудио передаётся в pyannote уже декодированным массивом (WAV 16 кГц моно, который готовит
конвейер), поэтому декодер torchcodec в работе не участвует.

torch и pyannote импортируются только в load(): модуль можно импортировать без GPU-стека.
"""

from __future__ import annotations

import gc
import logging
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from transkrib import models_store
from transkrib.asr import SAMPLE_RATE, ModelError

logger = logging.getLogger(__name__)
_NPP_WARNING = r"(?s).*libnpp"  # предупреждение torchcodec о CUDA NPP (libnppicc.so.12 и др.)


@dataclass(frozen=True, slots=True)
class Turn:
    start: float
    end: float
    speaker: str  # метка pyannote (SPEAKER_00, …); номера «Спикер N» назначает segment.py


@runtime_checkable
class Diarizer(Protocol):
    def load(self) -> None: ...

    def diarize(self, wav_path: Path) -> list[Turn]: ...

    def unload(self) -> None: ...

    def info(self) -> str: ...


def normalize_turns(output: Any) -> list[Turn]:
    """Результат pyannote → упорядоченный список отрезков. Пустой результат — пустой список (решает конвейер)."""
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)
    if not hasattr(annotation, "itertracks"):
        raise ModelError("pyannote вернул результат неизвестного формата")
    turns: list[Turn] = []
    dropped = 0
    for segment, _track, label in annotation.itertracks(yield_label=True):
        start, end = max(0.0, float(segment.start)), float(segment.end)
        if end <= start:
            dropped += 1
            continue
        turns.append(Turn(start, end, str(label)))
    if dropped:
        logger.warning("Отброшены отрезки диаризации нулевой длины", extra={"dropped": dropped})
    turns.sort(key=lambda t: (t.start, t.end, t.speaker))
    return turns


def read_wav_mono16k(path: Path) -> np.ndarray:
    """WAV PCM 16 бит, моно, 16 кГц → float32 в [-1, 1]. Другой формат — ошибка (WAV готовит конвейер)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != 2:
            raise ValueError("Ожидается WAV PCM 16 бит, моно, 16 кГц")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


class PyannoteDiarizer:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self.pipeline: Any = None
        self._torch: Any = None

    def info(self) -> str:
        return models_store.PYANNOTE_REPO

    def load(self) -> None:
        if self.pipeline is not None:
            return
        problems = models_store.check_pyannote(self.models_dir)
        if problems:
            raise ModelError("; ".join(str(p) for p in problems))
        models_store.offline_env()
        try:
            import torch

            with warnings.catch_warnings():
                # torchcodec при импорте ищет CUDA NPP (libnppicc.so.12) и предупреждает, если её нет. Безвредно:
                # аудио идёт в pyannote массивом, декодер torchcodec не используется — NPP в образ не ставим.
                warnings.filterwarnings("ignore", message=_NPP_WARNING)
                from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ModelError(f"Не установлен GPU-стек: {exc.name}") from None
        if not torch.cuda.is_available():
            raise ModelError("CUDA недоступна: диаризация работает только на GPU")
        try:
            pipeline = Pipeline.from_pretrained(str(models_store.pyannote_dir(self.models_dir)))
        except Exception as exc:
            raise ModelError(f"pyannote не загрузился: {type(exc).__name__}") from exc
        if pipeline is None:
            raise ModelError("pyannote не загрузился: снимок модели неполный")
        moved = pipeline.to(torch.device("cuda"))
        self.pipeline = moved if moved is not None else pipeline
        device = getattr(self.pipeline, "device", None)
        if device is not None and getattr(device, "type", str(device)) != "cuda":
            raise ModelError("pyannote загрузился не на CUDA")
        self._torch = torch
        logger.info("pyannote загружен", extra={"model": self.info()})

    def diarize(self, wav_path: Path) -> list[Turn]:
        if self.pipeline is None:
            raise ModelError("pyannote не загружен")
        torch = self._torch
        waveform = torch.from_numpy(read_wav_mono16k(Path(wav_path))).unsqueeze(0)
        with torch.inference_mode():
            output = self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        return normalize_turns(output)

    def unload(self) -> None:
        self.pipeline = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
