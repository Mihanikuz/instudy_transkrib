"""Общие модели данных конвейера."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Segment:
    """Распознанный кусок речи: время в секундах от начала записи, номер спикера (1, 2, …) и текст.

    Подпись «Спикер N» делает render — конвейер работает только с номерами.
    """

    start: float
    end: float
    speaker: int
    text: str
