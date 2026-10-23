"""Эксперимент: исправление терминов в готовом .txt локальной нейросетью (Ollama), без ручного словаря.

Нейросеть получает название дисциплины и кусок текста и возвращает ТОЛЬКО список замен {wrong, right}.
Текст она не переписывает: замену применяем сами и только если:
- wrong дословно есть в этом куске (целыми словами);
- wrong — 1–6 слов, right — до 8 слов, right != wrong;
- написание похоже (difflib по буквам ≥ MIN_SIMILARITY) — иначе это уже не исправление опечатки, а подмена смысла.
Все предложенные замены, принятые и отклонённые, пишутся в <out>.fixes.tsv для разбора.

Только stdlib: запускается прямо на сервере, без контейнера.
    python3 scripts/exp_llm_fix.py data/run-file/audio/audio.txt --discipline "…" --out data/run-file/llm-fix/audio.txt
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

CHUNK_CHARS = 3000
MIN_SIMILARITY = 0.35

PROMPT = """Ты — корректор автоматической расшифровки лекции на русском языке.
Дисциплина: «{discipline}».

Система распознавания речи часто искажает фамилии учёных, названия научных школ, теорий, методов и специальные термины.
Найди в тексте ниже ТОЛЬКО такие искажения и верни их правильное написание.

Правила:
- исправляй только имена собственные и специальные термины дисциплины;
- не исправляй обычные слова, грамматику, падежи, пунктуацию, порядок слов, слова-паразиты;
- поле "wrong" — точный фрагмент из текста, символ в символ, 1–6 слов;
- поле "right" — то же место, написанное правильно, в том же падеже;
- если не уверен — не включай;
- если исправлять нечего — верни пустой список.

Текст:
<<<
{text}
>>>"""

SCHEMA = {
    "type": "object",
    "properties": {
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"wrong": {"type": "string"}, "right": {"type": "string"}},
                "required": ["wrong", "right"],
            },
        }
    },
    "required": ["fixes"],
}


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Режем по абзацам (пустая строка), не длиннее size; длинный абзац — по предложениям."""
    out: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        pieces = [para] if len(para) <= size else re.split(r"(?<=[.!?…])\s+", para)
        for piece in pieces:
            sep = "\n\n" if piece is para else " "
            if cur and len(cur) + len(piece) + 2 > size:
                out.append(cur)
                cur = ""
            cur = f"{cur}{sep}{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out


def ask(url: str, model: str, discipline: str, text: str) -> list[dict]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(discipline=discipline, text=text)}],
        "format": SCHEMA,
        "think": False,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    if not url.startswith("http://127.0.0.1") and not url.startswith("http://localhost"):
        raise ValueError("Ollama — только локальный адрес")
    req = urllib.request.Request(  # noqa: S310 — только локальный http, проверено выше
        f"{url}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 — адрес локального Ollama из аргумента
        content = json.load(resp)["message"]["content"]
    try:
        fixes = json.loads(content).get("fixes", [])
    except (ValueError, AttributeError):
        return []
    return [
        f for f in fixes if isinstance(f, dict) and isinstance(f.get("wrong"), str) and isinstance(f.get("right"), str)
    ]


def letters(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower().replace("ё", "е"))


def check(fix: dict, chunk: str) -> tuple[str | None, float]:
    """Причина отказа (None — принять) и похожесть написания."""
    wrong, right = fix["wrong"].strip(), fix["right"].strip()
    sim = difflib.SequenceMatcher(None, letters(wrong), letters(right)).ratio()
    if not wrong or not right or letters(wrong) == letters(right):
        return "без изменений", sim
    if not 1 <= len(wrong.split()) <= 6 or len(right.split()) > 8:
        return "слишком длинно", sim
    if not re.search(rf"(?<!\w){re.escape(wrong)}(?!\w)", chunk):
        return "нет в тексте", sim
    if sim < MIN_SIMILARITY:
        return "непохоже", sim
    return None, sim


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("txt", type=Path)
    p.add_argument("--discipline", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--ollama", default="http://127.0.0.1:11434")
    args = p.parse_args(argv)

    lines = args.txt.read_text(encoding="utf-8").split("\n")
    header, body = "\n".join(lines[:3]), "\n".join(lines[3:])
    parts = chunks(body)
    fixed_parts: list[str] = []
    rows = ["chunk\tstatus\tsimilarity\twrong\tright"]
    applied = rejected = 0
    started = time.monotonic()
    for i, chunk in enumerate(parts, 1):
        t = time.monotonic()
        fixes = ask(args.ollama, args.model, args.discipline, chunk)
        new = chunk
        for fix in fixes:
            reason, sim = check(fix, chunk)
            status = reason or "принято"
            if reason is None:
                new = re.sub(rf"(?<!\w){re.escape(fix['wrong'].strip())}(?!\w)", fix["right"].strip(), new)
                applied += 1
            else:
                rejected += 1
            rows.append(f"{i}\t{status}\t{sim:.2f}\t{fix['wrong'].strip()}\t{fix['right'].strip()}")
        fixed_parts.append(new)
        print(
            f"кусок {i}/{len(parts)}: предложено {len(fixes)}, {time.monotonic() - t:.1f} с",
            file=sys.stderr,
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + "\n" + "\n\n".join(fixed_parts).rstrip("\n") + "\n", encoding="utf-8")
    tsv = args.out.with_suffix(".fixes.tsv")
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": args.model,
                "chunks": len(parts),
                "applied": applied,
                "rejected": rejected,
                "sec": round(time.monotonic() - started),
                "out": str(args.out),
                "fixes": str(tsv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
