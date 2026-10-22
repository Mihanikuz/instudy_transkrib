"""GPU-проверка на сервере: обе модели грузятся на CUDA офлайн и работают на коротком WAV.

Запуск (после T-7 — внутри контейнера):
    python scripts/gpu_smoke.py /data/smoke.wav

WAV — 16 кГц моно PCM16, лучше 30–120 с речи двух людей. Подготовить из любого файла:
    ffmpeg -i lecture.m4a -t 90 -ac 1 -ar 16000 -c:a pcm_s16le /data/smoke.wav

Печатает JSON: время загрузки и работы, число спикеров и отрезков, текст первых кусков, пиковую VRAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transkrib.asr import SAMPLE_RATE, GigaAMRecognizer  # noqa: E402
from transkrib.config import load_settings  # noqa: E402
from transkrib.diarize import PyannoteDiarizer, read_wav_mono16k  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--chunks", type=int, default=5, help="сколько кусков распознать")
    args = parser.parse_args(argv)

    import torch

    settings = load_settings()
    report: dict[str, object] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
    }
    torch.cuda.reset_peak_memory_stats()

    diarizer = PyannoteDiarizer(settings.models_dir)
    t = time.monotonic()
    diarizer.load()
    report["diarization_load_sec"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    turns = diarizer.diarize(args.wav)
    report["diarization_sec"] = round(time.monotonic() - t, 1)
    report["speakers"] = len({x.speaker for x in turns})
    report["turns"] = len(turns)
    report["vram_peak_diarization_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    diarizer.unload()

    torch.cuda.reset_peak_memory_stats()
    recognizer = GigaAMRecognizer(settings.models_dir)
    t = time.monotonic()
    recognizer.load()
    report["asr_load_sec"] = round(time.monotonic() - t, 1)

    samples = read_wav_mono16k(args.wav)
    chunks = []
    for turn in turns[: args.chunks]:
        start = int(turn.start * SAMPLE_RATE)
        end = min(int(turn.end * SAMPLE_RATE), start + 22 * SAMPLE_RATE)
        if end > start:
            chunks.append(samples[start:end])
    t = time.monotonic()
    texts = recognizer.transcribe_batch(chunks) if chunks else []
    report["asr_sec"] = round(time.monotonic() - t, 2)
    report["vram_peak_asr_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["samples"] = [
        {"speaker": turn.speaker, "start": round(turn.start, 1), "text": text}
        for turn, text in zip(turns, texts, strict=False)
    ]
    recognizer.unload()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if turns and any(texts) else 1


if __name__ == "__main__":
    sys.exit(main())
