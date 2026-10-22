"""Распознавание речи: интерфейс Recognizer и реализация на GigaAM v3_e2e_rnnt (только CUDA, офлайн).

Куски передаются массивами float32 16 кГц моно (не файлами): обычный GigaAM transcribe() на каждый вызов запускает
ffmpeg, а на 5-часовой лекции это тысячи процессов. Пакетная обработка повторяет то, что делает сам GigaAM внутри
transcribe_longform (forward + _decode на дополненном нулями пакете), — без его VAD.

torch и gigaam импортируются только в load(): модуль можно импортировать без GPU-стека.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from transkrib import models_store

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAX_CHUNK_SEC = 25.0  # предел GigaAM transcribe (LONGFORM_THRESHOLD)


class ModelError(RuntimeError):
    """Модель не загрузилась или не работает. Текст — без путей пользователя."""


@runtime_checkable
class Recognizer(Protocol):
    def load(self) -> None: ...

    def transcribe(self, samples: np.ndarray) -> str: ...

    def transcribe_batch(self, chunks: Sequence[np.ndarray]) -> list[str]: ...

    def unload(self) -> None: ...

    def info(self) -> str: ...


def is_cuda_oom(exc: BaseException) -> bool:
    """Нехватка памяти GPU — без импорта torch (тип torch.cuda.OutOfMemoryError или текст ошибки)."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def result_text(value: Any) -> str:
    """Нормализовать ответ GigaAM в строку: str, dict, (text, words), объект с .text или список таких."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text", value.get("transcript", ""))).strip()
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return value[0].strip()
    if isinstance(value, list):
        return " ".join(t for t in (result_text(v) for v in value) if t)
    text = getattr(value, "text", None)
    if text is None:
        text = getattr(value, "transcript", None)
    return str(text).strip() if text is not None else ""


def check_chunk(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim != 1:
        raise ValueError("Кусок аудио должен быть одномерным (моно)")
    if arr.size == 0:
        raise ValueError("Пустой кусок аудио")
    if arr.size > MAX_CHUNK_SEC * SAMPLE_RATE:
        raise ValueError(f"Кусок длиннее {MAX_CHUNK_SEC:.0f} с — его должна была разрезать нарезка")
    return arr.astype(np.float32, copy=False)


class GigaAMRecognizer:
    def __init__(self, models_dir: Path, *, fp16_encoder: bool = True) -> None:
        self.models_dir = Path(models_dir)
        self.fp16_encoder = fp16_encoder
        self.model: Any = None
        self._torch: Any = None

    def info(self) -> str:
        return f"gigaam {models_store.GIGAAM_MODEL} @{models_store.GIGAAM_COMMIT[:10]}"

    def load(self) -> None:
        if self.model is not None:
            return
        problems = models_store.check_gigaam(self.models_dir)
        if problems:
            raise ModelError("; ".join(str(p) for p in problems))
        models_store.offline_env()
        try:
            import gigaam
            import torch
        except ImportError as exc:
            raise ModelError(f"Не установлен GPU-стек: {exc.name}") from None
        if not torch.cuda.is_available():
            raise ModelError("CUDA недоступна: GigaAM работает только на GPU")
        try:
            model = gigaam.load_model(
                models_store.GIGAAM_MODEL,
                fp16_encoder=self.fp16_encoder,
                device="cuda",
                download_root=str(models_store.gigaam_dir(self.models_dir)),
            )
        except Exception as exc:
            raise ModelError(f"GigaAM не загрузился: {type(exc).__name__}") from exc
        for attr in ("forward", "_decode", "parameters"):
            if not hasattr(model, attr):
                raise ModelError(f"API GigaAM изменился: у модели нет {attr} — проверьте коммит в requirements-gpu.in")
        if next(model.parameters()).device.type != "cuda":
            raise ModelError("GigaAM загрузился не на CUDA")
        self.model, self._torch = model, torch
        logger.info("GigaAM загружен", extra={"model": self.info()})

    def transcribe(self, samples: np.ndarray) -> str:
        return self.transcribe_batch([samples])[0]

    def transcribe_batch(self, chunks: Sequence[np.ndarray]) -> list[str]:
        if self.model is None:
            raise ModelError("GigaAM не загружен")
        if not chunks:
            return []
        torch, model = self._torch, self.model
        arrays = [check_chunk(c) for c in chunks]
        lengths = torch.tensor([a.size for a in arrays], dtype=torch.long)
        batch = torch.zeros(len(arrays), int(lengths.max().item()), dtype=torch.float32)
        for i, arr in enumerate(arrays):
            batch[i, : arr.size] = torch.from_numpy(arr)
        params = next(model.parameters())
        with torch.inference_mode():
            wav = batch.to(params.device).to(params.dtype)
            lens = lengths.to(params.device)
            encoded, encoded_len = model.forward(wav, lens)
            decoded = model._decode(encoded, encoded_len, lens)
        texts = [result_text(item) for item in decoded]
        if len(texts) != len(arrays):
            raise ModelError("GigaAM вернул не столько результатов, сколько кусков")
        return texts

    def unload(self) -> None:
        self.model = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
