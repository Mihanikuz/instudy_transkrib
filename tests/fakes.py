"""Фейковые модели для тестов конвейера и демона (T-5, T-6). Детерминированные, без torch и GPU."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from transkrib.asr import SAMPLE_RATE, check_chunk
from transkrib.diarize import Turn


class FakeOOM(RuntimeError):
    """Похоже на torch.cuda.OutOfMemoryError: is_cuda_oom() узнаёт его по тексту."""

    def __init__(self) -> None:
        super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")


class FakeRecognizer:
    """Текст зависит только от длины куска: «фраза N (X.X с)».

    fail_on — номера вызовов распознавания одного куска (с 0), на которых бросить ошибку;
    oom_on — то же для нехватки памяти. Номер считается по кускам, а не по пакетам.
    """

    def __init__(self, *, fail_on: Sequence[int] = (), oom_on: Sequence[int] = ()) -> None:
        self.fail_on, self.oom_on = set(fail_on), set(oom_on)
        self.loaded = False
        self.calls = 0  # сколько кусков пытались распознать
        self.load_count = self.unload_count = 0

    def info(self) -> str:
        return "fake-asr"

    def load(self) -> None:
        self.loaded = True
        self.load_count += 1

    def unload(self) -> None:
        self.loaded = False
        self.unload_count += 1

    def transcribe(self, samples: np.ndarray) -> str:
        return self.transcribe_batch([samples])[0]

    def transcribe_batch(self, chunks: Sequence[np.ndarray]) -> list[str]:
        assert self.loaded, "распознавание до load()"
        out = []
        for chunk in chunks:
            arr = check_chunk(chunk)
            index = self.calls
            self.calls += 1
            if index in self.oom_on:
                self.oom_on.discard(index)
                raise FakeOOM()
            if index in self.fail_on:
                raise RuntimeError(f"сбой распознавания куска {index}")
            out.append(f"фраза {index + 1} ({arr.size / SAMPLE_RATE:.1f} с).")
        return out


class FakeDiarizer:
    """Возвращает заданный сценарий отрезков; без сценария — один спикер на всю длительность WAV."""

    def __init__(self, turns: Sequence[Turn] | None = None) -> None:
        self.turns = list(turns) if turns is not None else None
        self.loaded = False
        self.calls: list[Path] = []
        self.load_count = self.unload_count = 0

    def info(self) -> str:
        return "fake-diarization"

    def load(self) -> None:
        self.loaded = True
        self.load_count += 1

    def unload(self) -> None:
        self.loaded = False
        self.unload_count += 1

    def diarize(self, wav_path: Path) -> list[Turn]:
        assert self.loaded, "диаризация до load()"
        self.calls.append(Path(wav_path))
        if self.turns is not None:
            return list(self.turns)
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        return [Turn(0.0, duration, "SPEAKER_00")] if duration > 0 else []
