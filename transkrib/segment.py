"""Нарезка разметки pyannote на куски для ASR (docs/architecture.md, «Обработка одной работы», шаг 6).

Чистые функции без GPU и ввода-вывода:
- merge_turns — склеить соседние отрезки одного спикера с паузой меньше max_gap;
- split_long — разрезать длинный отрезок в точках минимальной энергии (RMS по окнам 30 мс);
- drop_short — отбросить слишком короткие отрезки;
- number_speakers — метки pyannote → номера 1, 2, 3… по первому появлению;
- plan_chunks — всё вместе в порядке архитектуры.

Работают с diarize.Turn: время в секундах, спикер — метка pyannote. Номера назначаются в самом конце, уже по
распознанным сегментам, чтобы спикер без единой реплики не оставил дыру в нумерации.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from transkrib.asr import SAMPLE_RATE
from transkrib.audio import rms_frames
from transkrib.diarize import Turn

MERGE_GAP_SEC = 0.5
MAX_SEGMENT_SEC = 22.0  # по умолчанию; в работе — Settings.max_segment_sec
HARD_MAX_SEC = 24.0
CUT_WINDOW_SEC = (15.0, 22.0)
MIN_SEGMENT_SEC = 0.3
RMS_FRAME_MS = 30


class _HasSpeaker(Protocol):
    @property
    def speaker(self) -> str: ...


def merge_turns(turns: Iterable[Turn], max_gap: float = MERGE_GAP_SEC) -> list[Turn]:
    """Склеить подряд идущие отрезки одного спикера, если пауза между ними меньше max_gap."""
    out: list[Turn] = []
    for turn in sorted(turns, key=lambda t: (t.start, t.end)):
        if out and out[-1].speaker == turn.speaker and turn.start - out[-1].end < max_gap:
            prev = out[-1]
            out[-1] = Turn(prev.start, max(prev.end, turn.end), prev.speaker)
        else:
            out.append(turn)
    return out


def _cut_point(samples: np.ndarray, lo: float, hi: float, sample_rate: int) -> float | None:
    """Время (с) середины самой тихой зоны в окне [lo; hi]; None — если сигнала в окне нет."""
    a, b = max(0, int(lo * sample_rate)), min(samples.size, int(hi * sample_rate))
    if b - a <= 0:
        return None
    rms = rms_frames(samples[a:b], RMS_FRAME_MS, sample_rate)
    frame_sec = RMS_FRAME_MS / 1000
    i = int(np.argmin(rms))
    # Тишина обычно — ряд одинаково тихих окон: режем посередине ряда, а не на его краю.
    quiet = rms <= rms[i] + 1e-6
    j = i
    while j + 1 < rms.size and quiet[j + 1]:
        j += 1
    cut = a / sample_rate + (i + j + 1) / 2 * frame_sec
    return min(max(cut, lo), hi)


def split_long(
    region: Turn,
    samples: np.ndarray,
    max_sec: float = MAX_SEGMENT_SEC,
    hard_sec: float = HARD_MAX_SEC,
    window: tuple[float, float] = CUT_WINDOW_SEC,
    sample_rate: int = SAMPLE_RATE,
) -> list[Turn]:
    """Разрезать отрезок длиннее max_sec на куски ≤ max_sec (и никогда > hard_sec).

    Каждый разрез — в точке минимального RMS внутри окна [window[0]; window[1]] от начала текущего куска.
    samples — весь сигнал записи (float32, sample_rate); вне сигнала режется ровно по max_sec.
    """
    hi = min(window[1], max_sec, hard_sec)
    lo = min(window[0], hi)
    pieces: list[Turn] = []
    start = region.start
    while region.end - start > max_sec:
        cut = _cut_point(samples, start + lo, start + hi, sample_rate)
        if cut is None or cut <= start:
            cut = start + hi
        cut = min(cut, start + hard_sec, start + max_sec)
        pieces.append(Turn(start, cut, region.speaker))
        start = cut
    pieces.append(Turn(start, region.end, region.speaker))
    return pieces


def drop_short(turns: Iterable[Turn], min_sec: float = MIN_SEGMENT_SEC) -> list[Turn]:
    return [t for t in turns if t.end - t.start >= min_sec]


def number_speakers(turns: Sequence[_HasSpeaker]) -> list[int]:
    """Номер спикера для каждого отрезка: 1, 2, 3… по порядку первого появления метки."""
    numbers: dict[str, int] = {}
    return [numbers.setdefault(t.speaker, len(numbers) + 1) for t in turns]


def plan_chunks(
    turns: Iterable[Turn],
    samples: np.ndarray,
    *,
    max_sec: float = MAX_SEGMENT_SEC,
    sample_rate: int = SAMPLE_RATE,
) -> list[Turn]:
    """Разметка pyannote → куски для ASR: склейка, разрезание длинных, отсев коротких; по времени."""
    chunks: list[Turn] = []
    for region in merge_turns(turns):
        chunks.extend(split_long(region, samples, max_sec=max_sec, sample_rate=sample_rate))
    return drop_short(chunks)
