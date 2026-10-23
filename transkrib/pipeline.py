"""Одна запись целиком: аудио → диаризация → нарезка → ASR → сегменты (docs/architecture.md, шаги 4–8).

Конвейер не знает про instudy: на входе файл, модели за интерфейсами Diarizer/Recognizer, флаг отмены и колбэк
прогресса; на выходе — сегменты для render или исключение:
- PipelineError(reason) — reason из docs/contract.md: decode_failed, no_speech, model_error, oom;
- ServerFault(PipelineError) — модель не загрузилась: неисправность сервера, а не записи (демон не засчитывает попытку);
- PipelineCancelled — флаг cancel выставлен (проверяется между этапами и перед каждым куском ASR).

torch на уровне модуля не импортируется: CUDA OOM узнаётся по типу/тексту исключения, кэш CUDA чистится через
уже загруженный бэкендом модуль torch.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from transkrib import audio, segment
from transkrib.asr import SAMPLE_RATE, Recognizer, is_cuda_oom
from transkrib.config import Settings, load_settings
from transkrib.diarize import Diarizer, Turn
from transkrib.types import PipelineCancelled, PipelineError, Segment, ServerFault

logger = logging.getLogger(__name__)

MAX_SKIPPED_SHARE = 0.05  # больше 5 % пропущенных кусков — model_error
DURATION_MISMATCH_SHARE = 0.02  # расхождение с ожидаемой длительностью — только предупреждение

STAGE_PREPARE = "prepare"
STAGE_DIARIZE = "diarize"
STAGE_SEGMENT = "segment"
STAGE_ASR = "asr"
STAGE_DONE = "done"

Progress = Callable[[str, float], None]


@dataclass(slots=True)
class PipelineResult:
    segments: list[Segment]
    speakers: int
    duration_sec: float
    processing_sec: float
    skipped_chunks: int
    chunks: int = 0
    stage_sec: dict[str, float] = field(default_factory=dict)


def empty_cuda_cache() -> None:
    """torch.cuda.empty_cache(), если torch уже загружен бэкендом; сам torch не импортирует."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # уборка памяти не должна ронять конвейер
        logger.warning("Не удалось очистить кэш CUDA", extra={"error": type(exc).__name__})


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise PipelineCancelled()


def _model_failure(exc: BaseException, stage: str) -> PipelineError:
    if is_cuda_oom(exc):
        return PipelineError("oom", f"Нехватка памяти GPU: {stage}")
    return PipelineError("model_error", f"Ошибка модели: {stage} ({type(exc).__name__})")


def load_model(model: Diarizer | Recognizer, what: str) -> None:
    """model.load(); любая ошибка загрузки — неисправность сервера (ServerFault), а не записи."""
    try:
        model.load()
    except Exception as exc:
        reason = "oom" if is_cuda_oom(exc) else "model_error"
        raise ServerFault(reason, f"Модель не загрузилась: {what} ({type(exc).__name__})") from exc


def prepare_wav(src: Path, work_dir: Path, cancel: threading.Event) -> Path:
    """Готовый WAV 16 кГц моно используется как есть; остальное перекодируется в work_dir/audio.wav."""
    if audio.is_wav16k_mono(src):
        return src
    dst = work_dir / "audio.wav"
    if dst.resolve() == src.resolve():
        dst = work_dir / "audio.16k.wav"
    return audio.to_wav16k_mono(src, dst, cancel=cancel)


def _diarize(wav: Path, diarizer: Diarizer, keep_loaded: bool) -> list[Turn]:
    try:
        load_model(diarizer, "диаризация")
        return diarizer.diarize(wav)
    except PipelineError:
        raise
    except Exception as exc:
        raise _model_failure(exc, "диаризация") from exc
    finally:
        if not keep_loaded:
            diarizer.unload()
            empty_cuda_cache()


def _recognize_chunk(recognizer: Recognizer, samples: np.ndarray, index: int) -> str | None:
    """Текст куска; None — кусок пропущен. Повторный OOM → PipelineError("oom")."""
    for attempt in (1, 2):
        try:
            return recognizer.transcribe(samples)
        except Exception as exc:
            if not is_cuda_oom(exc):
                logger.warning(
                    "Кусок пропущен: ошибка распознавания", extra={"chunk": index, "error": type(exc).__name__}
                )
                return None
            empty_cuda_cache()
            if attempt == 2:
                raise PipelineError("oom", "Нехватка памяти GPU при распознавании") from exc
            logger.warning("CUDA OOM на куске, повтор", extra={"chunk": index})
    return None  # pragma: no cover — цикл всегда завершается return или raise


def _recognize(
    chunks: list[Turn],
    samples: np.ndarray,
    recognizer: Recognizer,
    *,
    cancel: threading.Event,
    progress: Progress,
    keep_loaded: bool,
) -> tuple[list[str | None], int]:
    total = len(chunks)
    max_skipped = int(total * MAX_SKIPPED_SHARE)
    texts: list[str | None] = []
    skipped = 0
    try:
        load_model(recognizer, "ASR")
        for i, chunk in enumerate(chunks):
            _check_cancel(cancel)
            a, b = int(round(chunk.start * SAMPLE_RATE)), int(round(chunk.end * SAMPLE_RATE))
            text = _recognize_chunk(recognizer, samples[a:b], i)
            if text is None:
                skipped += 1
                if skipped > max_skipped:
                    raise PipelineError("model_error", f"Пропущено больше {MAX_SKIPPED_SHARE:.0%} кусков")
            texts.append(text)
            progress(STAGE_ASR, (i + 1) / total)
    finally:
        if not keep_loaded:
            recognizer.unload()
            empty_cuda_cache()
    return texts, skipped


def build_segments(chunks: list[Turn], texts: list[str | None]) -> list[Segment]:
    """Куски с непустым текстом → сегменты; спикеры нумеруются по первому появлению среди них."""
    spoken = [(c, t.strip()) for c, t in zip(chunks, texts, strict=True) if t and t.strip()]
    numbers = segment.number_speakers([c for c, _ in spoken])
    return [Segment(c.start, c.end, n, text) for (c, text), n in zip(spoken, numbers, strict=True)]


def run(
    wav_or_m4a: Path,
    *,
    diarizer: Diarizer,
    recognizer: Recognizer,
    cancel: threading.Event,
    progress: Progress,
    settings: Settings | None = None,
    work_dir: Path | None = None,
    expected_duration_sec: float | None = None,
) -> PipelineResult:
    """Обработать одну запись. work_dir — куда положить WAV (по умолчанию рядом с исходным файлом)."""
    settings = settings or load_settings()
    src = Path(wav_or_m4a)
    work_dir = Path(work_dir) if work_dir is not None else src.parent
    keep = settings.keep_models_loaded
    started = time.monotonic()
    stage_sec: dict[str, float] = {}

    def stage(name: str) -> None:
        stage_sec[name] = time.monotonic()
        progress(name, 0.0)

    def stage_done(name: str) -> None:
        stage_sec[name] = round(time.monotonic() - stage_sec[name], 3)

    _check_cancel(cancel)
    stage(STAGE_PREPARE)
    wav = prepare_wav(src, work_dir, cancel)
    duration = audio.wav_frames(wav) / SAMPLE_RATE
    if (
        expected_duration_sec
        and abs(duration - expected_duration_sec) > DURATION_MISMATCH_SHARE * expected_duration_sec
    ):
        logger.warning(
            "Длительность аудио расходится с заявленной",
            extra={"duration_sec": round(duration, 1), "expected_sec": round(expected_duration_sec, 1)},
        )
    stage_done(STAGE_PREPARE)

    _check_cancel(cancel)
    stage(STAGE_DIARIZE)
    turns = _diarize(wav, diarizer, keep)
    stage_done(STAGE_DIARIZE)
    if not turns:
        raise PipelineError("no_speech", "Диаризация не нашла речи")

    _check_cancel(cancel)
    stage(STAGE_SEGMENT)
    samples = audio.load_wav(wav)
    chunks = segment.plan_chunks(turns, samples, max_sec=settings.max_segment_sec)
    stage_done(STAGE_SEGMENT)
    if not chunks:
        raise PipelineError("no_speech", "Нет отрезков речи длиннее порога")
    logger.info("Нарезка готова", extra={"turns": len(turns), "chunks": len(chunks)})

    _check_cancel(cancel)
    stage(STAGE_ASR)
    texts, skipped = _recognize(chunks, samples, recognizer, cancel=cancel, progress=progress, keep_loaded=keep)
    stage_done(STAGE_ASR)
    del samples

    segments = build_segments(chunks, texts)
    if not segments:
        raise PipelineError("no_speech", "Распознавание не дало текста")
    speakers = max(s.speaker for s in segments)
    processing = time.monotonic() - started
    progress(STAGE_DONE, 1.0)
    logger.info(
        "Запись обработана",
        extra={
            "duration_sec": round(duration, 1),
            "processing_sec": round(processing, 1),
            "chunks": len(chunks),
            "segments": len(segments),
            "skipped_chunks": skipped,
            "speakers": speakers,
            "text_chars": sum(len(s.text) for s in segments),
        },
    )
    return PipelineResult(
        segments=segments,
        speakers=speakers,
        duration_sec=duration,
        processing_sec=processing,
        skipped_chunks=skipped,
        chunks=len(chunks),
        stage_sec=stage_sec,
    )
