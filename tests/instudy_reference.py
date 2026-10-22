"""Эталон: разбор субтитров, как его делает instudy (app/worker/transcript_extract.py, extract_subtitles, BE-7).

Скопирован, чтобы проверять без instudy, что наш .vtt instudy поймёт так, как мы задумали: реплики, голоса,
никаких таймкодов и тегов в тексте. Если instudy поменяет разбор — обновить эту копию (T-9).
"""

from __future__ import annotations

import html
import re

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
