"""CLI: python -m transkrib.cli <команда>.

T-1 — version; T-2 — models-check, models-import, models-download; T-5 — run-file. Дальше добавит T-6
(status, pause, resume, check, health).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from transkrib import __version__, models_store
from transkrib.config import ConfigError, Settings, load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m transkrib.cli", description="Управление сервером транскрибации")
    sub = parser.add_subparsers(dest="command", required=True, metavar="КОМАНДА")
    sub.add_parser("version", help="версия сервера")

    check = sub.add_parser("models-check", help="проверить модели на диске (без сети)")
    check.add_argument("--deep", action="store_true", help="пересчитать хэши файлов (долго)")

    imp = sub.add_parser("models-import", help="перенести модели из каталога models/ gigascribe")
    imp.add_argument("source", type=Path, help="каталог models/ gigascribe (в нём gigaam/ и pyannote/)")

    sub.add_parser("models-download", help="скачать модели из сети (нужен HF_TOKEN для pyannote)")

    run_file = sub.add_parser("run-file", help="прогнать конвейер на локальном файле настоящими моделями (GPU)")
    run_file.add_argument("audio", type=Path, help="аудиофайл (m4a, wav, …)")
    run_file.add_argument(
        "--out", type=Path, help="каталог для .vtt, .txt и meta.json (по умолчанию DATA_DIR/run-file/<имя>)"
    )
    return parser


def _models_dir() -> Path:
    return load_settings().models_dir


def cmd_models_check(deep: bool) -> int:
    models_dir = _models_dir()
    problems = models_store.check(models_dir, deep=deep)
    if problems:
        print(f"Модели в {models_dir} не готовы:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Модели в {models_dir} на месте: GigaAM {models_store.GIGAAM_MODEL}, {models_store.PYANNOTE_REPO}")
    return 0


def cmd_models_import(source: Path) -> int:
    models_store.import_models(source, _models_dir())
    return cmd_models_check(deep=False)


def cmd_models_download() -> int:
    settings = load_settings()
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    models_store.download_models(settings.models_dir, token)
    return cmd_models_check(deep=False)


ENGINE_LINE = "GigaAM v3 e2e RNNT, pyannote community-1"


def make_models(settings: Settings) -> tuple[Any, Any]:
    """Настоящие модели (CUDA). Тесты подменяют эту функцию фейками."""
    from transkrib.asr import GigaAMRecognizer
    from transkrib.diarize import PyannoteDiarizer

    return PyannoteDiarizer(settings.models_dir), GigaAMRecognizer(settings.models_dir)


class _StageReport:
    """Колбэк прогресса для run-file: печатает этапы и долю кусков ASR, замеряет пиковую VRAM по этапам."""

    def __init__(self) -> None:
        self.current: str | None = None
        self.vram_peak_gb: dict[str, float] = {}
        self._last_pct = -1

    @staticmethod
    def _torch() -> Any:
        torch = sys.modules.get("torch")
        return torch if torch is not None and torch.cuda.is_available() else None

    def finish_stage(self) -> None:
        torch = self._torch()
        if self.current is not None and torch is not None:
            self.vram_peak_gb[self.current] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()

    def __call__(self, stage: str, fraction: float) -> None:
        if stage != self.current:
            self.finish_stage()
            self.current = stage
            self._last_pct = -1
            print(f"[{time.strftime('%H:%M:%S')}] этап: {stage}", file=sys.stderr, flush=True)
        pct = int(fraction * 100) // 10 * 10
        if stage == "asr" and pct > self._last_pct:
            self._last_pct = pct
            print(f"  asr: {pct} %", file=sys.stderr, flush=True)


def cmd_run_file(audio_path: Path, out: Path | None) -> int:
    from transkrib.log import setup_logging
    from transkrib.pipeline import run
    from transkrib.render import build_meta, render_txt, render_vtt, validate_vtt
    from transkrib.types import PipelineCancelled, PipelineError

    settings = load_settings()
    setup_logging(settings.log_level, settings.secret_values(), stream=sys.stderr)
    if not audio_path.is_file():
        print(f"Нет файла: {audio_path}", file=sys.stderr)
        return 2
    out = out or settings.data_dir / "run-file" / audio_path.stem
    out.mkdir(parents=True, exist_ok=True)
    diarizer, recognizer = make_models(settings)
    report = _StageReport()
    cancel = threading.Event()
    try:
        with tempfile.TemporaryDirectory(prefix=".work-", dir=out) as work:
            result = run(
                audio_path,
                diarizer=diarizer,
                recognizer=recognizer,
                cancel=cancel,
                progress=report,
                settings=settings,
                work_dir=Path(work),
            )
    except PipelineError as exc:
        print(f"Ошибка конвейера: {exc.reason} — {exc.message}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, PipelineCancelled):
        print("Прервано", file=sys.stderr)
        return 130
    finally:
        report.finish_stage()
        diarizer.unload()
        recognizer.unload()

    vtt = render_vtt(result.segments)
    validate_vtt(vtt, result.segments)
    txt = render_txt(result.segments, discipline=audio_path.stem, date=date.today(), engine_line=ENGINE_LINE)
    meta = build_meta(
        server=f"instudy_transkrib {__version__}",
        asr=recognizer.info(),
        diarization=diarizer.info(),
        audio_duration_sec=result.duration_sec,
        processing_sec=result.processing_sec,
        speakers=result.speakers,
        segments=len(result.segments),
    )
    stem = audio_path.stem
    (out / f"{stem}.vtt").write_text(vtt, encoding="utf-8")
    (out / f"{stem}.txt").write_text(txt, encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        **meta,
        "chunks": result.chunks,
        "skippedChunks": result.skipped_chunks,
        "realtimeFactor": round(result.processing_sec / result.duration_sec, 4) if result.duration_sec else None,
        "stageSec": result.stage_sec,
        "vramPeakGb": report.vram_peak_gb or None,
        "out": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "version":
            print(f"instudy_transkrib {__version__}")
            return 0
        if args.command == "models-check":
            return cmd_models_check(args.deep)
        if args.command == "models-import":
            return cmd_models_import(args.source)
        if args.command == "models-download":
            return cmd_models_download()
        if args.command == "run-file":
            return cmd_run_file(args.audio, args.out)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 2  # argparse не пропустит неизвестную команду; на случай новых подкоманд без обработчика


if __name__ == "__main__":
    sys.exit(main())
