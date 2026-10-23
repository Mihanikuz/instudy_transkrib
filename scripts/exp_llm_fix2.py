"""Эксперимент v2: исправление терминов нейросетью (Ollama) — двойной проход, фильтр по словарю, единый словарь лекции.

Этапы:
1. Два независимых прохода по тексту с разной нарезкой на куски. Нейросеть возвращает только список замен
   {wrong, right}. Принимаем пару, только если её предложили ОБА прохода (случайные ошибки отсеиваются).
2. Каждая замена проходит фильтр:
   - wrong дословно есть в куске, 1–6 слов; right до 8 слов; без комментариев в скобках;
   - написание похоже (difflib по буквам ≥ 0.35);
   - не только смена окончания (одинаковые леммы pymorphy3) — грамматику не трогаем;
   - wrong — не набор правильных русских слов по словарю pymorphy3 (так «педология» не станет «педагогией»).
3. Единый словарь лекции: все незнакомые словарю слова (и результаты этапа 1) отдаются нейросети одним списком,
   дважды в разном порядке; принимаем совпавшие ответы и применяем ко всему тексту — «бихеоризм» везде станет
   одинаковым.
Все решения — в <out>.fixes.tsv.

Нужен pymorphy3 (pip install pymorphy3==2.0.6). Ollama — локальный адрес.
    python scripts/exp_llm_fix2.py data/run-file/audio/audio.txt --discipline "…" --out data/run-file/llm-fix2/audio.txt
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pymorphy3

CHUNK_CHARS = 6000
MIN_SIM_TEXT = 0.35
MIN_SIM_WORD = 0.5
VOCAB_BATCH = 120
WORD_RX = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
COMMENT_RX = re.compile(r"[()\[\]{}]|вероятно|возможно|ошибк|исправ|имеется в виду", re.IGNORECASE)

RULES = """Правила:
- исправляй только искажённые фамилии, имена учёных, названия школ, теорий, методов и специальные термины;
- не трогай обычные слова, грамматику, окончания, пунктуацию, порядок слов, разговорные обороты;
- не заменяй один настоящий термин другим настоящим термином;
- если не уверен — не включай;
- никаких пояснений и комментариев: только пары wrong/right."""

PROMPT_TEXT = """Ты — корректор автоматической расшифровки лекции на русском языке. Дисциплина: «{discipline}».
Распознавание речи искажает фамилии и термины. Найди такие искажения в тексте и верни правильное написание.

{rules}
- "wrong" — точный фрагмент из текста, символ в символ, 1–6 слов; "right" — то же место правильно, в том же падеже.

Текст:
<<<
{text}
>>>"""

PROMPT_VOCAB = """Ниже слова из автоматической расшифровки лекции по дисциплине «{discipline}» (в скобках — сколько раз
встретилось). Часть из них — искажённые распознаванием фамилии и специальные термины.
Для каждого искажённого слова дай правильное написание в той же грамматической форме.

{rules}
- "wrong" — слово из списка как есть; "right" — одно слово (или фамилия через дефис).

Слова:
{words}"""

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

morph = pymorphy3.MorphAnalyzer()


# ----------------------------------------------------------------------------- нейросеть


def ask(url: str, model: str, prompt: str, think: bool) -> list[tuple[str, str]]:
    if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
        raise ValueError("Ollama — только локальный адрес")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": SCHEMA,
        "think": think,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 16384},
    }
    req = urllib.request.Request(  # noqa: S310 — только локальный http, проверено выше
        f"{url}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:  # noqa: S310
        content = json.load(resp)["message"]["content"]
    try:
        items = json.loads(content).get("fixes", [])
    except (ValueError, AttributeError):
        return []
    out = []
    for f in items:
        if isinstance(f, dict) and isinstance(f.get("wrong"), str) and isinstance(f.get("right"), str):
            w, r = f["wrong"].strip(), f["right"].strip()
            if w and r:
                out.append((w, r))
    return out


# ----------------------------------------------------------------------------- фильтр


def letters(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower().replace("ё", "е"))


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, letters(a), letters(b)).ratio()


def known_common(word: str) -> bool:
    """Правильное русское слово (не имя, не фамилия, не топоним) по словарю pymorphy3."""
    for p in morph.parse(word.lower()):
        if p.is_known and not ({"Name", "Surn", "Patr", "Geox", "Orgn"} & set(p.tag.grammemes)):
            return True
    return False


def lemmas(s: str) -> list[str]:
    return [morph.parse(w.lower())[0].normal_form for w in WORD_RX.findall(s)]


def reject_reason(wrong: str, right: str, *, max_words: int = 6, min_sim: float = MIN_SIM_TEXT) -> str | None:
    if letters(wrong) == letters(right):
        return "без изменений"
    if COMMENT_RX.search(right):
        return "комментарий"
    if not 1 <= len(wrong.split()) <= max_words or len(right.split()) > max_words + 2:
        return "слишком длинно"
    if similarity(wrong, right) < min_sim:
        return "непохоже"
    if lemmas(wrong) == lemmas(right):
        return "грамматика"
    words = WORD_RX.findall(wrong)
    if words and all(known_common(w) for w in words):
        return "обычные слова"
    return None


def pattern(wrong: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(wrong)}(?![\w-])")


# ----------------------------------------------------------------------------- нарезка


def spans(text: str, size: int, start: int = 0) -> list[tuple[int, int]]:
    para = [m.end() for m in re.finditer(r"\n\n", text)]
    sent = sorted(set(para + [m.end() for m in re.finditer(r"[.!?…]\s", text)]))
    out: list[tuple[int, int]] = []
    s = start
    while s < len(text):
        limit = s + size
        if limit >= len(text):
            out.append((s, len(text)))
            break
        lo = s + size // 2
        p = [b for b in para[bisect.bisect_right(para, lo) : bisect.bisect_right(para, limit)]]
        c = sent[bisect.bisect_right(sent, lo) : bisect.bisect_right(sent, limit)]
        e = (p or c or [limit])[-1]
        out.append((s, e))
        s = e
    return out


# ----------------------------------------------------------------------------- этапы


def text_pass(body, chunk_spans, args, label, log) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Пара → куски, где она предложена и прошла фильтр."""
    found: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for i, (s, e) in enumerate(chunk_spans, 1):
        chunk = body[s:e]
        t = time.monotonic()
        fixes = ask(
            args.ollama, args.model, PROMPT_TEXT.format(discipline=args.discipline, rules=RULES, text=chunk), args.think
        )
        for w, r in fixes:
            reason = "нет в тексте" if not pattern(w).search(chunk) else reject_reason(w, r)
            log.append(("текст", label, i, reason or "кандидат", f"{similarity(w, r):.2f}", w, r))
            if reason is None:
                found.setdefault((w, r), []).append((s, e))
        print(
            f"проход {label}, кусок {i}/{len(chunk_spans)}: {len(fixes)} предложено, {time.monotonic() - t:.0f} с",
            file=sys.stderr,
            flush=True,
        )
    return found


def apply_in_spans(body: str, pairs: dict[tuple[str, str], list[tuple[int, int]]]) -> tuple[str, int]:
    edits: list[tuple[int, int, str]] = []
    for (w, r), sp in pairs.items():
        for s, e in sp:
            for m in pattern(w).finditer(body, s, e):
                edits.append((m.start(), m.end(), r))
    edits.sort()
    out, pos, n = [], 0, 0
    for s, e, r in edits:
        if s < pos:
            continue  # пересечение с уже применённой заменой — пропускаем
        out.append(body[pos:s])
        out.append(r)
        pos, n = e, n + 1
    out.append(body[pos:])
    return "".join(out), n


def vocab_stage(body: str, extra: set[str], args, log) -> dict[str, str]:
    counts = Counter(WORD_RX.findall(body))
    cand = sorted(w for w, _ in counts.items() if len(w) >= 4 and (w in extra or not known_common(w)))
    runs: list[dict[str, str]] = []
    for run in range(2):
        order = cand[:]
        random.Random(run).shuffle(order)  # noqa: S311 — перемешивание порядка, не криптография
        mapping: dict[str, str] = {}
        for i in range(0, len(order), VOCAB_BATCH):
            batch = order[i : i + VOCAB_BATCH]
            listing = "\n".join(f"{w} ({counts[w]})" for w in batch)
            t = time.monotonic()
            fixes = ask(
                args.ollama,
                args.model,
                PROMPT_VOCAB.format(discipline=args.discipline, rules=RULES, words=listing),
                args.think,
            )
            for w, r in fixes:
                if w not in counts:
                    reason = "нет в тексте"
                else:
                    reason = reject_reason(w, r, max_words=1, min_sim=MIN_SIM_WORD)
                log.append(
                    (
                        "словарь",
                        str(run + 1),
                        i // VOCAB_BATCH + 1,
                        reason or "кандидат",
                        f"{similarity(w, r):.2f}",
                        w,
                        r,
                    )
                )
                if reason is None:
                    mapping[w] = r
            print(
                f"словарь, проход {run + 1}, пачка {i // VOCAB_BATCH + 1}: {len(fixes)} предложено, "
                f"{time.monotonic() - t:.0f} с",
                file=sys.stderr,
                flush=True,
            )
        runs.append(mapping)
    agreed = {w: r for w, r in runs[0].items() if runs[1].get(w) == r}
    for w, r in agreed.items():
        log.append(("словарь", "оба", 0, "принято", f"{similarity(w, r):.2f}", w, r))
    print(f"словарь: кандидатов слов {len(cand)}, совпало в двух проходах {len(agreed)}", file=sys.stderr)
    return agreed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("txt", type=Path)
    p.add_argument("--discipline", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="qwen3:30b-a3b")
    p.add_argument("--no-think", dest="think", action="store_false")
    p.add_argument("--ollama", default="http://127.0.0.1:11434")
    args = p.parse_args(argv)

    lines = args.txt.read_text(encoding="utf-8").split("\n")
    header, body = "\n".join(lines[:3]), "\n".join(lines[3:])
    log: list[tuple] = []
    started = time.monotonic()

    a_spans = spans(body, CHUNK_CHARS)
    shift = next((e for _, e in spans(body, CHUNK_CHARS // 2)), 0)
    b_spans = [(0, shift)] + spans(body, CHUNK_CHARS, shift)
    found_a = text_pass(body, a_spans, args, "A", log)
    found_b = text_pass(body, b_spans, args, "B", log)
    agreed = {k: found_a[k] + found_b[k] for k in found_a.keys() & found_b.keys()}
    for w, r in agreed:
        log.append(("текст", "оба", 0, "принято", f"{similarity(w, r):.2f}", w, r))
    body, n_text = apply_in_spans(body, agreed)
    print(f"текст: пар в A {len(found_a)}, в B {len(found_b)}, совпало {len(agreed)}, замен {n_text}", file=sys.stderr)

    extra = {w for (_, r) in agreed for w in WORD_RX.findall(r)}
    mapping = vocab_stage(body, extra, args, log)
    n_vocab = 0
    for w, r in mapping.items():
        body, k = pattern(w).subn(r, body)
        n_vocab += k

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + "\n" + body.rstrip("\n") + "\n", encoding="utf-8")
    tsv = args.out.with_suffix(".fixes.tsv")
    rows = ["этап\tпроход\tкусок\tрешение\tпохожесть\tбыло\tстало"] + ["\t".join(map(str, r)) for r in log]
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": args.model,
                "think": args.think,
                "text_pairs_agreed": len(agreed),
                "text_replacements": n_text,
                "vocab_agreed": len(mapping),
                "vocab_replacements": n_vocab,
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
