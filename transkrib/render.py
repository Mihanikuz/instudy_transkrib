"""Сегменты → .vtt и .txt (docs/contract.md, «Формат результата»).

Чистые функции без ввода-вывода. Обе функции принимают один и тот же список сегментов и сами отбрасывают пустые
тексты. Пустой результат (нет ни одного сегмента с текстом) — ValueError: конвейер в этом случае отвечает no_speech
и ничего не отправляет.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date as Date

from transkrib.types import Segment

SPEAKER_LABEL = "Спикер {n}"
ENGINE_LINE = "GigaAM v3 e2e RNNT, pyannote community-1"  # вторая строка шапки .txt

PARAGRAPH_GAP_SEC = 60.0  # пауза больше — новый абзац того же спикера
PARAGRAPH_MAX_SEC = 180.0  # абзац длиннее — новый абзац того же спикера

_WS_RX = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RX = re.compile(r"\s+([.,!?;:…»)\]])")
_SPACE_AFTER_OPEN_RX = re.compile(r"([«(\[])\s+")


def speaker_label(n: int) -> str:
    return SPEAKER_LABEL.format(n=n)


def _clean(text: str) -> str:
    text = _WS_RX.sub(" ", text.replace("\xa0", " ")).strip()
    text = _SPACE_BEFORE_PUNCT_RX.sub(r"\1", text)
    return _SPACE_AFTER_OPEN_RX.sub(r"\1", text)


@dataclass(frozen=True, slots=True)
class _Cue:
    start_ms: int
    end_ms: int
    speaker: int
    text: str


def _prepare(segments: Iterable[Segment]) -> list[_Cue]:
    """Отбросить пустые, упорядочить, проверить спикеров и время, убрать пересечения."""
    cues: list[_Cue] = []
    for seg in segments:
        text = _clean(seg.text)
        if not text:
            continue
        if seg.start < 0 or seg.end <= seg.start:
            raise ValueError(f"Неверное время сегмента: {seg.start}–{seg.end}")
        if seg.speaker < 1:
            raise ValueError(f"Номер спикера должен быть ≥ 1, получен {seg.speaker}")
        cues.append(_Cue(round(seg.start * 1000), round(seg.end * 1000), seg.speaker, text))
    if not cues:
        raise ValueError("Нет ни одного сегмента с текстом")

    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    speakers = {c.speaker for c in cues}
    if speakers != set(range(1, max(speakers) + 1)):
        raise ValueError(f"Номера спикеров должны идти без пропусков: {sorted(speakers)}")

    # Пересечение: конец предыдущей реплики подрезается до начала следующей.
    out: list[_Cue] = []
    for cue in cues:
        if out and cue.start_ms < out[-1].end_ms:
            prev = out[-1]
            if cue.start_ms <= prev.start_ms:
                raise ValueError("Две реплики начинаются одновременно — нарезка сломана")
            out[-1] = _Cue(prev.start_ms, cue.start_ms, prev.speaker, prev.text)
        out.append(cue)
    return out


def _vtt_time(ms: int) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _txt_time(ms: int) -> str:
    total = ms // 1000
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"


def _vtt_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_vtt(segments: Iterable[Segment]) -> str:
    """WebVTT: заголовок, нумерованные реплики, голос <v Спикер N> в каждой."""
    blocks = ["WEBVTT"]
    for i, cue in enumerate(_prepare(segments), start=1):
        blocks.append(
            f"{i}\n{_vtt_time(cue.start_ms)} --> {_vtt_time(cue.end_ms)}\n"
            f"<v {speaker_label(cue.speaker)}>{_vtt_escape(cue.text)}"
        )
    return "\n\n".join(blocks) + "\n"


def _format_date(value: Date | str) -> str:
    if isinstance(value, str):
        value = Date.fromisoformat(value)
    return value.strftime("%d.%m.%Y")


def render_txt(segments: Iterable[Segment], *, discipline: str, date: Date | str, engine_line: str) -> str:
    """Читаемый текст: шапка и абзацы «[ЧЧ:ММ:СС] Спикер N: текст» через пустую строку."""
    header = [f"{_clean(discipline) or 'Занятие'} — {_format_date(date)}", f"Автотранскрипт: {_clean(engine_line)}"]

    paragraphs: list[tuple[int, int, list[str]]] = []  # (начало, спикер, тексты)
    para_start = prev_end = 0
    for cue in _prepare(segments):
        new_paragraph = (
            not paragraphs
            or cue.speaker != paragraphs[-1][1]
            or cue.start_ms - prev_end > PARAGRAPH_GAP_SEC * 1000
            or cue.end_ms - para_start > PARAGRAPH_MAX_SEC * 1000
        )
        if new_paragraph:
            paragraphs.append((cue.start_ms, cue.speaker, []))
            para_start = cue.start_ms
        paragraphs[-1][2].append(cue.text)
        prev_end = cue.end_ms

    lines = [
        f"{_txt_time(start)} {speaker_label(speaker)}: {_clean(' '.join(texts))}"
        for start, speaker, texts in paragraphs
    ]
    return "\n".join(header) + "\n\n" + "\n\n".join(lines) + "\n"


def build_meta(
    *,
    server: str,
    asr: str,
    diarization: str,
    audio_duration_sec: float,
    processing_sec: float,
    speakers: int,
    segments: int,
) -> dict[str, str | int]:
    return {
        "server": server,
        "asr": asr,
        "diarization": diarization,
        "audioDurationSec": round(audio_duration_sec),
        "processingSec": round(processing_sec),
        "speakers": speakers,
        "segments": segments,
    }


# ------------------------------------------------------------------- самопроверка результата

_TIMECODE_LINE_RX = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})$")
_VOICE_LINE_RX = re.compile(r"^<v (Спикер (\d+))>(.+)$")


@dataclass(frozen=True, slots=True)
class ParsedCue:
    start_ms: int
    end_ms: int
    speaker: int
    text: str


def _ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def parse_vtt(text: str) -> list[ParsedCue]:
    """Строгий разбор нашего собственного .vtt. Любое отклонение от формата контракта — ValueError."""
    if text.startswith("﻿"):
        raise ValueError("BOM в начале файла")
    if "\r" in text:
        raise ValueError("Переводы строк должны быть \\n")
    if not text.endswith("\n"):
        raise ValueError("Файл должен заканчиваться переводом строки")
    blocks = text[:-1].split("\n\n")
    if blocks[0] != "WEBVTT":
        raise ValueError("Первая строка должна быть WEBVTT")
    cues: list[ParsedCue] = []
    for i, block in enumerate(blocks[1:], start=1):
        lines = block.split("\n")
        if len(lines) != 3 or lines[0] != str(i):
            raise ValueError(f"Реплика {i}: ожидались номер, таймкод и текст")
        tc = _TIMECODE_LINE_RX.match(lines[1])
        voice = _VOICE_LINE_RX.match(lines[2])
        if not tc or not voice:
            raise ValueError(f"Реплика {i}: неверный таймкод или голос")
        start, end = _ms(*tc.groups()[:4]), _ms(*tc.groups()[4:])
        if end <= start or (cues and start < cues[-1].end_ms):
            raise ValueError(f"Реплика {i}: время не возрастает или пересекается с предыдущей")
        body = voice.group(3)
        if re.search(r"<|>", body):
            raise ValueError(f"Реплика {i}: неэкранированные < или >")
        cues.append(ParsedCue(start, end, int(voice.group(2)), body))
    if not cues:
        raise ValueError("В файле нет реплик")
    return cues


def validate_vtt(text: str, segments: Sequence[Segment]) -> None:
    """Проверить, что .vtt разбирается и соответствует сегментам: число реплик и спикеры."""
    parsed = parse_vtt(text)
    expected = _prepare(segments)
    if len(parsed) != len(expected):
        raise ValueError(f"Реплик в файле {len(parsed)}, ожидалось {len(expected)}")
    for got, want in zip(parsed, expected, strict=True):
        if (got.start_ms, got.end_ms, got.speaker) != (want.start_ms, want.end_ms, want.speaker):
            raise ValueError("Таймкоды или спикеры в файле не совпадают с сегментами")
