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


class PipelineError(RuntimeError):
    """Обработка записи не удалась: reason — машинный код из docs/contract.md, message — короткий текст без путей."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(f"{reason}: {message}" if message else reason)
        self.reason = reason
        self.message = message


class PipelineCancelled(Exception):
    """Работу отменили (флаг cancel): аренда отозвана или демон останавливают."""
