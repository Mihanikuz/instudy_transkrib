"""CLI: python -m transkrib.cli <команда>.

T-1 — version; T-2 — models-check, models-import, models-download; T-5 — run-file;
T-6 — status, pause, resume, check, health.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from transkrib import __version__, models_store
from transkrib.config import ConfigError, Settings, load_daemon_settings, load_settings
from transkrib.state import PAUSE_FILE, parse_iso, read_state

HEALTH_MAX_AGE_SEC = 180
REASON_TEXT = {
    "pause": "файл PAUSE, cli resume снимет",
    "low_disk": "мало места на диске",
    "unauthorized": "instudy отклонил токен (401)",
    "models": "нет файлов моделей, см. cli models-check",
    "models_load": "модели не загружаются — CUDA/драйвер? см. cli check",
    "model_breaker": "предохранитель: несколько работ подряд со сбоем модели",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m transkrib.cli", description="Управление сервером транскрибации")
    sub = parser.add_subparsers(dest="command", required=True, metavar="КОМАНДА")
    sub.add_parser("version", help="версия сервера")
    sub.add_parser("status", help="что делает демон (из state.json)")
    sub.add_parser("pause", help="не брать новые работы (текущая доработает)")
    sub.add_parser("resume", help="снова брать работы")
    sub.add_parser("check", help="проверить instudy (health), модели и GPU")
    sub.add_parser("health", help=f"код 0, если демон жив (heartbeat свежее {HEALTH_MAX_AGE_SEC} с) — для Docker")

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
    from transkrib.render import ENGINE_LINE, build_meta, render_txt, render_vtt, validate_vtt
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


def _age_sec(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    try:
        return (now - parse_iso(value)).total_seconds()
    except (TypeError, ValueError):
        return None


def cmd_status(data_dir: Path, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    paused = (data_dir / PAUSE_FILE).exists()
    state = read_state(data_dir)
    if state is None:
        print(f"Демон не запущен или ещё не записал {data_dir}/state.json")
        if paused:
            print("Пауза: включена (файл PAUSE)")
        return 1
    age = _age_sec(state.get("heartbeat_at"), now)
    alive = age is not None and age <= HEALTH_MAX_AGE_SEC
    status = state.get("status", "?")
    reason = state.get("reason")
    lines = [
        f"Демон: {'жив' if alive else 'НЕ ОТВЕЧАЕТ'} (pid {state.get('pid')}, версия {state.get('version')}, "
        f"запущен {state.get('started_at')}, отметка {round(age) if age is not None else '?'} с назад)",
        f"Состояние: {status}"
        + (f" ({reason}: {REASON_TEXT[reason]})" if reason in REASON_TEXT else f" ({reason})" if reason else ""),
        f"Пауза: {'включена (файл PAUSE)' if paused else 'нет'}",
    ]
    if state.get("next_check_at"):
        lines.append(f"Следующая проверка моделей: {state['next_check_at']}")
    job = state.get("job")
    if isinstance(job, dict):
        progress = job.get("progress")
        pct = f"{float(progress) * 100:.0f} %" if isinstance(progress, int | float) else "?"
        lines.append(
            f"Работа: #{job.get('id')} (день {job.get('dayId')}, {job.get('date')}), этап {job.get('stage')} {pct}, "
            f"начата {job.get('started_at')}"
        )
    counters = state.get("counters") or {}
    lines.append(
        "С запуска: готово {done}, отказов {failed}, пропущено {skipped}, отменено {cancelled}".format(
            **{k: counters.get(k, 0) for k in ("done", "failed", "skipped", "cancelled")}
        )
    )
    last = state.get("last_error")
    if isinstance(last, dict):
        lines.append(f"Последняя ошибка: {last.get('code')} в {last.get('at')}")
    print("\n".join(lines))
    return 0


def cmd_pause(data_dir: Path) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / PAUSE_FILE).touch()
    print("Пауза включена: новые работы не берутся, текущая доработает")
    return 0


def cmd_resume(data_dir: Path) -> int:
    (data_dir / PAUSE_FILE).unlink(missing_ok=True)
    print("Пауза снята: демон возьмёт работу при следующем опросе")
    return 0


def cmd_health(data_dir: Path, now: datetime | None = None) -> int:
    state = read_state(data_dir)
    age = _age_sec(state.get("heartbeat_at") if state else None, now or datetime.now(UTC))
    return 0 if age is not None and -60 <= age <= HEALTH_MAX_AGE_SEC else 1


def _check_instudy() -> tuple[bool, str]:
    from transkrib.client import ClientError, InstudyClient, Unauthorized
    from transkrib.log import register_secret

    try:
        settings = load_daemon_settings()
    except ConfigError as exc:
        return False, str(exc)
    for secret in settings.secret_values():
        register_secret(secret)
    token = settings.service_token.get_secret_value() if settings.service_token else ""
    try:
        with InstudyClient(
            settings.instudy_api_url or "",
            token,
            max_audio_bytes=settings.max_audio_bytes,
            max_audio_sec=settings.max_audio_sec,
        ) as client:
            data = client.health()
    except Unauthorized:
        return False, "токен отклонён (401) — нужен новый SERVICE_TOKEN"
    except ClientError as exc:
        return False, str(exc)
    return True, f"ok, клиент {data.get('client', '?')}, время сервера {data.get('serverTime', '?')}"


def _check_gpu() -> tuple[bool | None, str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, "nvidia-smi не найден — пропуск"
    query = "--query-gpu=name,driver_version,memory.used,memory.total"
    try:
        res = subprocess.run(  # noqa: S603 — путь из which, аргументы — константы
            [exe, query, "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"nvidia-smi не отработал: {type(exc).__name__}"
    if res.returncode != 0:
        return False, (res.stderr or res.stdout).strip()[:300] or f"код {res.returncode}"
    return True, res.stdout.strip()


def cmd_check() -> int:
    ok = True
    instudy_ok, instudy_msg = _check_instudy()
    print(f"instudy: {instudy_msg}")
    ok &= instudy_ok
    models_dir = load_settings().models_dir
    problems = models_store.check(models_dir)
    print(f"Модели ({models_dir}): " + ("ok" if not problems else "; ".join(str(p) for p in problems)))
    ok &= not problems
    gpu_ok, gpu_msg = _check_gpu()
    print(f"GPU: {gpu_msg}")
    ok &= gpu_ok is not False
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "version":
            print(f"instudy_transkrib {__version__}")
            return 0
        if args.command == "status":
            return cmd_status(load_settings().data_dir)
        if args.command == "pause":
            return cmd_pause(load_settings().data_dir)
        if args.command == "resume":
            return cmd_resume(load_settings().data_dir)
        if args.command == "health":
            return cmd_health(load_settings().data_dir)
        if args.command == "check":
            return cmd_check()
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
