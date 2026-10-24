"""Эталоны из instudy — копии, чтобы проверять без instudy, что наш результат instudy примет и поймёт.

- extract_subtitles — разбор субтитров (app/worker/transcript_extract.py, BE-7): реплики, голоса, никаких таймкодов
  и тегов в тексте;
- decode_strict, check_vtt, parse_meta — проверка результата автотранскрибации перед приёмом (app/autotx_result.py,
  BE-18): нарушение — 422 в instudy. Фейковый instudy (tests/fake_instudy.py) прогоняет через них каждый result.

Сверено с instudy main 4e55604 (23.10.2026, T-9), перепроверено на c5ad409 (24.10.2026): extract_subtitles и
app/autotx_result.py не менялись. Если instudy поменяет эти функции — обновить копию и tests/test_contract_instudy.py.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from email.parser import BytesParser
from email.policy import HTTP
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

_TIMECODE_RX = re.compile(r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->")
_TAG_RX = re.compile(r"<[^>\n]*>")
_ASS_TAG_RX = re.compile(r"\{\\[^}\n]*\}")
_VOICE_RX = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>\n]+)>")
_VTT_BLOCKS = ("NOTE", "STYLE", "REGION")


def extract_subtitles(text: str) -> str:
    blocks = re.split(r"\n\s*\n", text)
    cues: list[tuple[str | None, str]] = []
    for i, block in enumerate(blocks):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if i == 0 and lines[0].startswith("WEBVTT"):
            continue
        if lines[0].split(" ", 1)[0] in _VTT_BLOCKS and not any(_TIMECODE_RX.match(ln) for ln in lines):
            continue
        tc = next((j for j, ln in enumerate(lines) if _TIMECODE_RX.match(ln)), None)
        body = [ln for ln in lines if not ln.isdigit()] if tc is None else lines[tc + 1 :]
        speaker = None
        parts: list[str] = []
        for ln in body:
            m = _VOICE_RX.search(ln)
            if m and speaker is None:
                speaker = m.group(1).strip()
            ln = _ASS_TAG_RX.sub("", _TAG_RX.sub("", ln))
            ln = html.unescape(ln).replace("\xa0", " ").strip()
            if ln:
                parts.append(ln)
        if parts:
            cues.append((speaker, " ".join(parts)))

    out: list[str] = []
    prev: tuple[str | None, str] | None = None
    current_speaker: str | None = None
    for speaker, line in cues:
        if prev == (speaker, line):
            continue
        prev = (speaker, line)
        if speaker and speaker != current_speaker:
            current_speaker = speaker
            out.append(f"\n{speaker}: {line}")
        else:
            out.append(line if not out else f" {line}")
    return "".join(out).strip()


# ------------------------------------------------------------------ app/autotx_result.py (BE-18)

MAX_HOURS = 12
_TIMING_RX = re.compile(r"(\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3}) --> (\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3})")
_RESULT_VOICE_RX = re.compile(r"<v Спикер ([1-9]\d?)>")
_OK_CONTROLS = {"\t", "\n", "\r"}


class ResultError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def decode_strict(raw: bytes, field: str) -> str:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ResultError(f"{field}: файл должен быть в кодировке UTF-8") from None
    if any(unicodedata.category(ch) == "Cc" and ch not in _OK_CONTROLS for ch in text):
        raise ResultError(f"{field}: недопустимые управляющие символы или NUL")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _ms(groups: tuple[str, ...]) -> int:
    h, m, s, ms = (int(g) for g in groups)
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def check_vtt(text: str) -> int:
    lines = text.split("\n")
    head = lines[0] if lines else ""
    if not (head == "WEBVTT" or head.startswith(("WEBVTT ", "WEBVTT\t"))):
        raise ResultError("vtt: файл должен начинаться со строки WEBVTT")
    if len(lines) > 1 and lines[1].strip():
        raise ResultError("vtt: после WEBVTT нужна пустая строка")
    body = "\n".join(lines[1:])
    cues = 0
    prev_start = -1
    for block in re.split(r"\n[ \t]*\n", body):
        rows = [r for r in block.split("\n") if r.strip()]
        if not rows:
            continue
        if rows[0].split(" ", 1)[0].split("\t", 1)[0] in _VTT_BLOCKS:
            raise ResultError("vtt: блоки NOTE, STYLE и REGION не допускаются")
        timing = 0 if "-->" in rows[0] else 1
        if timing >= len(rows) or "-->" not in rows[timing]:
            raise ResultError("vtt: у реплики нет строки таймкода «ЧЧ:ММ:СС.ммм --> ЧЧ:ММ:СС.ммм»")
        m = _TIMING_RX.fullmatch(rows[timing].strip())
        if m is None:
            raise ResultError("vtt: таймкод — только «ЧЧ:ММ:СС.ммм --> ЧЧ:ММ:СС.ммм», без настроек реплики")
        start, end = _ms(m.groups()[:4]), _ms(m.groups()[4:])
        if start > end:
            raise ResultError("vtt: начало реплики позже её конца")
        if start < prev_start:
            raise ResultError("vtt: начала реплик должны идти по возрастанию")
        if end > MAX_HOURS * 3600 * 1000:
            raise ResultError(f"vtt: таймкод больше {MAX_HOURS} часов")
        prev_start = start
        payload = rows[timing + 1 :]
        text_parts = []
        for row in payload:
            if "-->" in row:
                raise ResultError("vtt: между репликами нужна пустая строка")
            for tag in _TAG_RX.findall(row):
                if tag == "</v>":
                    continue
                if tag.startswith(("<v ", "<v.", "<v>")):
                    if not _RESULT_VOICE_RX.fullmatch(tag):
                        raise ResultError("vtt: голос — только «<v Спикер N>», N от 1 до 99, без имён")
                    continue
                raise ResultError("vtt: теги, кроме <v Спикер N>, не допускаются")
            plain = _TAG_RX.sub("", row)
            if "<" in plain or ">" in plain:
                raise ResultError("vtt: символы < и > в тексте нужно экранировать (&lt; &gt;)")
            text_parts.append(html.unescape(plain).strip())
        if any(text_parts):
            cues += 1
    if cues == 0:
        raise ResultError("vtt: нет ни одной реплики с текстом")
    return cues


def _no_controls(value: str) -> str:
    if any(unicodedata.category(ch) == "Cc" for ch in value):
        raise ValueError("строка с управляющими символами")
    return value


class MetaIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    server: Annotated[str, Field(min_length=1, max_length=100)]
    asr: Annotated[str, Field(min_length=1, max_length=200)]
    diarization: Annotated[str, Field(min_length=1, max_length=200)]
    audioDurationSec: Annotated[StrictInt, Field(ge=0)] | None = None  # noqa: N815 — имя поля договора
    processingSec: Annotated[StrictInt, Field(ge=0)] | None = None  # noqa: N815
    speakers: Annotated[StrictInt, Field(ge=1, le=99)] | None = None
    segments: Annotated[StrictInt, Field(ge=0)] | None = None

    _strings = field_validator("server", "asr", "diarization")(_no_controls)


def parse_meta(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except ValueError:
        raise ResultError("meta: должно быть JSON") from None
    if not isinstance(data, dict):
        raise ResultError("meta: должен быть JSON-объект")
    try:
        meta = MetaIn.model_validate(data)
    except ValidationError as e:
        first = e.errors(include_input=False)[0]
        where = ".".join(str(x) for x in first.get("loc", ())) or "meta"
        raise ResultError(f"meta: поле «{where}» не по договору") from None
    return meta.model_dump(exclude_none=True)


# ------------------------------------------------------------------ multipart (не копия: разбор для фейка)


def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str, bytes]] | None:
    """Поля формы: имя → (тип с charset, байты). None — форма оборвана (нет закрывающей границы)."""
    boundary = re.search(r'boundary="?([^";]+)"?', content_type)
    if boundary is None or f"--{boundary.group(1)}--".encode() not in body:
        return None
    msg = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    parts = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        charset = part.get_content_charset()
        parts[name] = (
            part.get_content_type() + (f"; charset={charset}" if charset else ""),
            part.get_payload(decode=True),
        )
    return parts
