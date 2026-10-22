"""Эксперимент: Whisper вместо GigaAM в том же конвейере (диаризация pyannote, нарезка, рендер — наши).

Не для продакшена: ветка exp/whisper. Нужен для честного сравнения моделей на одной и той же нарезке.

Запуск (в одноразовом контейнере с GPU-стеком, см. README эксперимента в описании ветки):
    python scripts/exp_whisper.py data/audio.m4a --model openai/whisper-large-v3 --out data/run-file/whisper-large-v3

Whisper получает только куски речи ≤ 22 с из разметки pyannote — тишину он не видит, поэтому галлюцинаций на паузах
(«Редактор субтитров…») должно быть заметно меньше, чем при прогоне целого файла.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transkrib import cli  # noqa: E402
from transkrib.asr import SAMPLE_RATE, check_chunk  # noqa: E402
from transkrib.diarize import PyannoteDiarizer  # noqa: E402


class WhisperRecognizer:
    """Recognizer на transformers: fp16, CUDA, язык зафиксирован русским, без таймкодов внутри куска."""

    def __init__(self, model_id: str, cache_dir: Path, *, beams: int = 1, batch: int = 16) -> None:
        self.model_id, self.cache_dir, self.beams, self.batch = model_id, cache_dir, beams, batch
        self.model: Any = None
        self.processor: Any = None
        self._torch: Any = None

    def info(self) -> str:
        return f"{self.model_id} (beams={self.beams})"

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        kw = {"cache_dir": str(self.cache_dir), "local_files_only": True}
        self.processor = WhisperProcessor.from_pretrained(self.model_id, **kw)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(self.model_id, dtype=torch.float16, **kw).to("cuda").eval()
        )
        self._torch = torch

    def transcribe(self, samples: np.ndarray) -> str:
        return self.transcribe_batch([samples])[0]

    def transcribe_batch(self, chunks: Sequence[np.ndarray]) -> list[str]:
        torch = self._torch
        out: list[str] = []
        for i in range(0, len(chunks), self.batch):
            part = [check_chunk(c) for c in chunks[i : i + self.batch]]
            feats = self.processor(part, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            feats = feats.to("cuda", torch.float16)
            with torch.inference_mode():
                ids = self.model.generate(feats, language="ru", task="transcribe", num_beams=self.beams)
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out

    def unload(self) -> None:
        self.model = None
        gc.collect()
        if self._torch is not None:
            self._torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Whisper в конвейере transkrib (эксперимент)")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    settings = cli.load_settings()
    cache = settings.data_dir / "hf-cache"
    # Скачать модель до старта конвейера: pyannote при загрузке включает офлайн-режим Hugging Face.
    from huggingface_hub import snapshot_download

    snapshot_download(args.model, cache_dir=str(cache), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    os.environ["HF_HUB_OFFLINE"] = "1"

    cli.ENGINE_LINE = f"{args.model.split('/')[-1]}, pyannote community-1"
    cli.make_models = lambda s: (
        PyannoteDiarizer(s.models_dir),
        WhisperRecognizer(args.model, cache, beams=args.beams),
    )
    return cli.main(["run-file", str(args.audio), "--out", str(args.out)])


if __name__ == "__main__":
    sys.exit(main())
