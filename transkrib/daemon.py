"""Демон: python -m transkrib (docs/architecture.md, «Обработка одной работы», «Ошибки и устойчивость»).

Главный поток крутит цикл: пауза → место на диске → jobs/next → обработка → result/fail → уборка. Обработка
(скачивание, конвейер, рендер) идёт в рабочем потоке, чтобы главный поток оставался свободен для сигналов: если
этап не прерывается (диаризация — один вызов pyannote), демон после SHUTDOWN_GRACE_SEC сообщает fail shutdown,
убирает за собой и выходит, не дожидаясь GPU.

Потоки:
- heartbeat — на время работы, каждые HEARTBEAT_SEC; 409/410 → флаг отмены, fail не шлётся;
- тикер — всё время жизни, обновляет heartbeat_at в state.json раз в 30 с (healthcheck Docker).

Все ожидания идут через Clock: тесты подменяют часы и не спят по-настоящему.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from transkrib import __version__, models_store
from transkrib.client import (
    Cancelled,
    ClientError,
    DownloadFailed,
    FailReason,
    InstudyClient,
    InvalidJob,
    Job,
    LeaseLost,
    LinkExpired,
    RateLimited,
    Rejected,
    ResultOutcome,
    Transient,
    Unauthorized,
)
from transkrib.config import Settings
from transkrib.pipeline import run as run_pipeline
from transkrib.render import ENGINE_LINE, build_meta, render_txt, render_vtt, validate_vtt
from transkrib.state import PAUSE_FILE, STATUS_ERROR, STATUS_IDLE, STATUS_PAUSED, StateStore, write_json_atomic
from transkrib.types import PipelineCancelled, PipelineError

logger = logging.getLogger(__name__)

T = TypeVar("T")

BACKOFF_START_SEC = 5.0
BACKOFF_MAX_SEC = 300.0
UNAUTHORIZED_RETRY_SEC = 600.0  # 401: проверка раз в 10 минут
MODELS_RETRY_SEC = 600.0
STATE_TOUCH_SEC = 30.0
CLEANUP_EVERY = timedelta(days=1)
SHUTDOWN_GRACE_SEC = 20.0  # + fail и уборка — укладываемся в 30 с (stop_grace_period: 45s)
FAIL_ATTEMPTS = 3
GB = 1000**3


# ------------------------------------------------------------------------------- часы


class Clock:
    """Настоящие часы. wait() — прерываемый сон: True, если событие выставлено."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def wait(self, event: threading.Event, seconds: float) -> bool:
        return event.wait(max(0.0, seconds))


# ------------------------------------------------------------------------------- исходы


class _Stop(Exception):  # noqa: N818 — внутренний сигнал: демон останавливают
    pass


class _LeaseGone(Exception):  # noqa: N818 — внутренний сигнал: аренда потеряна, сообщать некому
    pass


@dataclass(frozen=True, slots=True)
class Rendered:
    vtt: bytes
    txt: bytes
    meta: dict[str, Any]


class _JobContext:
    def __init__(self, job: Job) -> None:
        self.job = job
        self.cancel = threading.Event()  # для конвейера и скачивания: потеря аренды или остановка
        self.lease_lost = threading.Event()
        self.done = threading.Event()  # останавливает heartbeat


def _fail_reason(code: str) -> FailReason:
    try:
        return FailReason(code)
    except ValueError:
        return FailReason.MODEL_ERROR


# ------------------------------------------------------------------------------- демон


class Daemon:
    def __init__(
        self,
        settings: Settings,
        client: InstudyClient,
        diarizer: Any,
        recognizer: Any,
        *,
        clock: Clock | None = None,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        models_check: Callable[[Path], list[Any]] = models_store.check,
        state_observer: Callable[[Path], None] | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.diarizer = diarizer
        self.recognizer = recognizer
        self.clock = clock or Clock()
        self.disk_usage = disk_usage
        self.models_check = models_check
        self.data_dir = Path(settings.data_dir)
        self.work_root = self.data_dir / "work"
        self.results_root = self.data_dir / "results"
        self.state = StateStore(self.data_dir, now=self.clock.now, observer=state_observer)
        self.shutdown = threading.Event()
        self.abandoned = False  # рабочий поток не успел остановиться — выходить через os._exit
        self._current: _JobContext | None = None
        self._signals = 0
        self._last_cleanup: datetime | None = None

    # ----------------------------------------------------------------- сигналы

    def request_stop(self) -> None:
        self.shutdown.set()
        ctx = self._current
        if ctx is not None:
            ctx.cancel.set()

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self._signals += 1
        if self._signals > 1:
            logger.warning("Повторный сигнал — немедленный выход", extra={"signal": signum})
            os._exit(128 + signum)
        logger.info("Получен сигнал остановки", extra={"signal": signum})
        self.request_stop()

    def install_signal_handlers(self) -> dict[int, Any]:
        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, self._on_signal)
        return previous

    # ----------------------------------------------------------------- служебное

    def _sleep(self, seconds: float) -> None:
        """Сон, прерываемый остановкой. Остановка → _Stop."""
        if self.clock.wait(self.shutdown, seconds):
            raise _Stop()

    def _ticker(self) -> None:
        stop = self.shutdown
        while not self.clock.wait(stop, 1.0):
            if self.clock.now() - self.state.last_write() >= timedelta(seconds=STATE_TOUCH_SEC):
                try:
                    self.state.touch()
                except OSError as exc:
                    logger.error("Не удалось записать state.json", extra={"error": type(exc).__name__})

    def _retry(self, fn: Callable[[], T], what: str, *, lease: _JobContext | None = None) -> T:
        """Повторять fn при Transient/RateLimited (5 с → 5 мин). Остановка → _Stop, потеря аренды → _LeaseGone."""
        delay = BACKOFF_START_SEC
        while True:
            if lease is not None and lease.lease_lost.is_set():
                raise _LeaseGone()
            try:
                return fn()
            except RateLimited as exc:
                wait = max(exc.retry_after, 1.0)
                logger.warning("instudy: лимит запросов", extra={"op": what, "retry_in": wait})
            except Transient as exc:
                wait = delay
                logger.warning("instudy недоступен", extra={"op": what, "error": str(exc), "retry_in": wait})
            delay = min(delay * 2, BACKOFF_MAX_SEC)
            self._sleep(wait)

    def _wait_unauthorized(self) -> None:
        logger.critical("instudy отклонил сервисный токен (401): выдача остановлена, повтор через 10 минут")
        self.state.set_status(STATUS_ERROR, "unauthorized")
        self.state.error("unauthorized")
        self._sleep(UNAUTHORIZED_RETRY_SEC)

    # ----------------------------------------------------------------- старт

    def _clean_work(self) -> None:
        if self.work_root.exists():
            for child in self.work_root.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)

    def _startup(self) -> None:
        self._clean_work()
        self.state.touch()
        while True:
            problems = self.models_check(self.settings.models_dir)
            if not problems:
                break
            logger.critical("Модели не готовы: работа не берётся", extra={"problems": [str(p) for p in problems]})
            self.state.set_status(STATUS_ERROR, "models")
            self.state.error("models")
            self._sleep(MODELS_RETRY_SEC)
        while True:
            try:
                self._retry(self.client.health, "health")
                break
            except Unauthorized:
                self._wait_unauthorized()
        self.state.set_status(STATUS_IDLE)
        logger.info("Демон запущен", extra={"version": __version__})

    # ----------------------------------------------------------------- уборка результатов

    def cleanup_results(self) -> int:
        """Удалить копии результатов старше RESULTS_KEEP_DAYS. Возвращает число удалённых каталогов."""
        now = self.clock.now()
        self._last_cleanup = now
        border = (now - timedelta(days=self.settings.results_keep_days)).timestamp()
        removed = 0
        if not self.results_root.is_dir():
            return 0
        for child in self.results_root.iterdir():
            try:
                if child.is_dir() and not child.is_symlink() and child.stat().st_mtime < border:
                    shutil.rmtree(child)
                    removed += 1
            except OSError as exc:
                logger.warning("Не удалось удалить старый результат", extra={"error": type(exc).__name__})
        if removed:
            logger.info("Удалены старые результаты", extra={"removed": removed})
        return removed

    def _maybe_cleanup(self) -> None:
        if self._last_cleanup is None or self.clock.now() - self._last_cleanup >= CLEANUP_EVERY:
            self.cleanup_results()

    # ----------------------------------------------------------------- цикл

    def _blocked_reason(self) -> str | None:
        if (self.data_dir / PAUSE_FILE).exists():
            return "pause"
        try:
            free = self.disk_usage(self.data_dir).free
        except OSError:
            free = 0
        if free < self.settings.min_free_gb * GB:
            logger.warning(
                "Мало места на диске: работа не берётся",
                extra={"free_gb": round(free / GB, 1), "min_free_gb": self.settings.min_free_gb},
            )
            return "low_disk"
        return None

    def iteration(self) -> None:
        """Одна итерация цикла. Остановка → _Stop."""
        self._maybe_cleanup()
        reason = self._blocked_reason()
        if reason is not None:
            self.state.set_status(STATUS_PAUSED, reason)
            self._sleep(self.settings.poll_interval_sec)
            return
        try:
            job = self._retry(self.client.next_job, "jobs/next")
        except Unauthorized:
            self._wait_unauthorized()
            return
        except InvalidJob as exc:
            logger.error("Работа не прошла проверку", extra={"job_id": exc.job_id, "error": str(exc)})
            if exc.job_id is not None:
                self.state.error(FailReason.INVALID_JOB)
                self._send_fail(exc.job_id, FailReason.INVALID_JOB, str(exc))
            return
        if job is None:
            self.state.set_status(STATUS_IDLE)
            self._sleep(self.settings.poll_interval_sec)
            return
        self.process(job)

    def run(self) -> int:
        ticker = threading.Thread(target=self._ticker, name="state-ticker", daemon=True)
        try:
            self._startup()
            ticker.start()
            while not self.shutdown.is_set():
                self.iteration()
        except _Stop:
            pass
        finally:
            if not self.abandoned:
                for model in (self.diarizer, self.recognizer):
                    try:
                        model.unload()
                    except Exception as exc:  # выгрузка не должна мешать выходу
                        logger.warning("Ошибка выгрузки модели", extra={"error": type(exc).__name__})
            self.state.set_status(STATUS_IDLE, "stopped")
        logger.info("Демон остановлен")
        return 0

    # ----------------------------------------------------------------- одна работа

    def _heartbeat(self, ctx: _JobContext) -> None:
        interval = float(self.settings.heartbeat_sec)
        while not self.clock.wait(ctx.done, interval):
            try:
                self.client.heartbeat(ctx.job.id)
                self.state.touch()
            except LeaseLost:
                logger.warning("Аренда потеряна: обработка прекращается", extra={"job_id": ctx.job.id})
                ctx.lease_lost.set()
                ctx.cancel.set()
                return
            except Unauthorized:
                logger.critical("instudy отклонил токен на heartbeat", extra={"job_id": ctx.job.id})
            except (Transient, RateLimited) as exc:
                logger.warning("heartbeat не прошёл, повтор в следующий тик", extra={"error": str(exc)})
            except Exception as exc:  # поток не должен умирать молча
                logger.error("Ошибка heartbeat", extra={"error": type(exc).__name__})

    def _work(self, ctx: _JobContext, job_dir: Path) -> Rendered:
        """Рабочий поток: скачать, обработать, отрендерить."""
        job = ctx.job
        self.state.progress("download", 0.0)
        audio_path = self.client.download(job, job_dir / "audio.m4a", cancel=ctx.cancel)
        if ctx.cancel.is_set():
            raise PipelineCancelled()
        result = run_pipeline(
            audio_path,
            diarizer=self.diarizer,
            recognizer=self.recognizer,
            cancel=ctx.cancel,
            progress=self.state.progress,
            settings=self.settings,
            work_dir=job_dir,
            expected_duration_sec=job.audio_duration_sec,
        )
        self.state.progress("render", 0.0)
        vtt = render_vtt(result.segments)
        validate_vtt(vtt, result.segments)
        txt = render_txt(result.segments, discipline=job.discipline, date=job.date, engine_line=ENGINE_LINE)
        meta = build_meta(
            server=f"instudy_transkrib {__version__}",
            asr=self.recognizer.info(),
            diarization=self.diarizer.info(),
            audio_duration_sec=result.duration_sec,
            processing_sec=result.processing_sec,
            speakers=result.speakers,
            segments=len(result.segments),
        )
        return Rendered(vtt.encode("utf-8"), txt.encode("utf-8"), meta)

    def _run_worker(self, ctx: _JobContext, job_dir: Path) -> Rendered:
        """Запустить _work в потоке и дождаться; при остановке ждать не дольше SHUTDOWN_GRACE_SEC."""
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = self._work(ctx, job_dir)
            except BaseException as exc:  # передаём в главный поток
                box["error"] = exc

        worker = threading.Thread(target=target, name=f"job-{ctx.job.id}", daemon=True)
        worker.start()
        stop_seen: float | None = None
        while worker.is_alive():
            worker.join(0.2)
            if self.shutdown.is_set():
                stop_seen = stop_seen if stop_seen is not None else time.monotonic()
                if worker.is_alive() and time.monotonic() - stop_seen > SHUTDOWN_GRACE_SEC:
                    logger.warning("Этап не прервался за отведённое время — выход без ожидания GPU")
                    self.abandoned = True
                    raise _Stop()
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _send_fail(self, job_id: int, reason: FailReason, message: str) -> None:
        for attempt in range(1, FAIL_ATTEMPTS + 1):
            try:
                accepted = self.client.fail(job_id, reason, message)
                logger.info(
                    "Отказ отправлен" if accepted else "Отказ не нужен: работа уже не наша",
                    extra={"job_id": job_id, "reason": str(reason)},
                )
                return
            except Unauthorized:
                logger.critical("instudy отклонил токен на fail", extra={"job_id": job_id})
                return
            except (Transient, RateLimited) as exc:
                logger.warning("fail не отправлен", extra={"job_id": job_id, "attempt": attempt, "error": str(exc)})
                if attempt < FAIL_ATTEMPTS:
                    self.clock.wait(self.shutdown, BACKOFF_START_SEC * attempt)
            except ClientError as exc:
                logger.error("fail не принят", extra={"job_id": job_id, "error": str(exc)})
                return

    def _submit(self, ctx: _JobContext, rendered: Rendered) -> ResultOutcome:
        return self._retry(
            lambda: self.client.submit_result(ctx.job.id, rendered.vtt, rendered.txt, rendered.meta),
            "result",
            lease=ctx,
        )

    def _keep_copy(self, job_id: int, rendered: Rendered) -> None:
        final = self.results_root / str(int(job_id))
        tmp = self.results_root / f".tmp-{int(job_id)}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "transcript.vtt").write_bytes(rendered.vtt)
        (tmp / "transcript.txt").write_bytes(rendered.txt)
        write_json_atomic(tmp / "meta.json", rendered.meta)
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)

    def process(self, job: Job) -> str:
        """Обработать работу целиком. Возвращает исход: done / failed / skipped / cancelled."""
        ctx = _JobContext(job)
        job_dir = self.work_root / str(int(job.id))
        job_dir.mkdir(parents=True, exist_ok=True)
        self._current = ctx
        if self.shutdown.is_set():
            ctx.cancel.set()
        self.state.start_job(job.id, job.day_id, job.date)
        logger.info("Работа взята", extra={"job_id": job.id, "day_id": job.day_id, "bytes": job.audio_size_bytes})
        heartbeat = threading.Thread(target=self._heartbeat, args=(ctx,), name=f"heartbeat-{job.id}", daemon=True)
        heartbeat.start()
        outcome = "failed"
        fail: tuple[FailReason, str] | None = None
        try:
            try:
                rendered = self._run_worker(ctx, job_dir)
                if self.shutdown.is_set():
                    raise _Stop()
                if ctx.lease_lost.is_set():
                    raise _LeaseGone()
                self.state.progress("submit", 0.0)
                result = self._submit(ctx, rendered)
                if result is ResultOutcome.ACCEPTED:
                    self._keep_copy(job.id, rendered)
                    outcome = "done"
                elif result is ResultOutcome.ALREADY_HAS_TRANSCRIPT:
                    logger.info("У дня уже есть транскрипт (409): работа закрыта", extra={"job_id": job.id})
                    outcome = "skipped"
                else:
                    logger.info("Работа больше не наша (410)", extra={"job_id": job.id})
                    outcome = "cancelled"
            except _Stop:
                fail = (FailReason.SHUTDOWN, "Демон остановлен посреди работы")
            except (PipelineCancelled, Cancelled, LeaseLost, _LeaseGone):
                if self.shutdown.is_set() and not ctx.lease_lost.is_set():
                    fail = (FailReason.SHUTDOWN, "Демон остановлен посреди работы")
                else:
                    logger.info("Работа отменена: аренда потеряна", extra={"job_id": job.id})
                    outcome = "cancelled"
            except LinkExpired:
                logger.warning("Ссылка на аудио истекла: работа вернётся в очередь сама", extra={"job_id": job.id})
                outcome = "skipped"
            except DownloadFailed as exc:
                fail = (FailReason.DOWNLOAD_FAILED, str(exc))
            except PipelineError as exc:
                fail = (_fail_reason(exc.reason), exc.message or exc.reason)
            except Rejected as exc:
                if exc.reason is FailReason.RESULT_REJECTED:
                    logger.critical("instudy не принял формат результата — это баг", extra={"status": exc.status})
                fail = (exc.reason, str(exc))
            except Unauthorized:
                logger.critical("instudy отклонил токен посреди работы", extra={"job_id": job.id})
                self.state.error("unauthorized")
                outcome = "cancelled"
            except (Transient, RateLimited) as exc:
                fail = (FailReason.DOWNLOAD_FAILED, str(exc))
            except Exception as exc:
                logger.error("Непредвиденная ошибка обработки", extra={"job_id": job.id, "error": type(exc).__name__})
                fail = (FailReason.MODEL_ERROR, f"Внутренняя ошибка: {type(exc).__name__}")
            if fail is not None:
                reason, message = fail
                logger.warning("Работа не выполнена", extra={"job_id": job.id, "reason": str(reason)})
                self.state.error(reason)
                if not ctx.lease_lost.is_set():
                    self._send_fail(job.id, reason, message)
                outcome = "failed"
        finally:
            ctx.done.set()
            ctx.cancel.set()
            heartbeat.join(timeout=10)
            shutil.rmtree(job_dir, ignore_errors=True)
            self._current = None
            self.state.finish_job(outcome)
        logger.info("Работа завершена", extra={"job_id": job.id, "outcome": outcome})
        if self.shutdown.is_set():
            raise _Stop()
        return outcome


# ------------------------------------------------------------------------------- запуск


def build_models(settings: Settings) -> tuple[Any, Any]:
    from transkrib.asr import GigaAMRecognizer
    from transkrib.diarize import PyannoteDiarizer

    return PyannoteDiarizer(settings.models_dir), GigaAMRecognizer(settings.models_dir)


def main() -> int:
    from transkrib.config import ConfigError, load_daemon_settings
    from transkrib.log import setup_logging

    try:
        settings = load_daemon_settings()
    except ConfigError as exc:
        print(exc, flush=True)
        return 2
    setup_logging(settings.log_level, settings.secret_values())
    models_store.offline_env()
    token = settings.service_token.get_secret_value() if settings.service_token else ""
    client = InstudyClient(
        settings.instudy_api_url or "",
        token,
        max_audio_bytes=settings.max_audio_bytes,
        max_audio_sec=settings.max_audio_sec,
    )
    diarizer, recognizer = build_models(settings)
    daemon = Daemon(settings, client, diarizer, recognizer)
    daemon.install_signal_handlers()
    try:
        code = daemon.run()
    finally:
        client.close()
    if daemon.abandoned:
        logging.shutdown()
        os._exit(code)
    return code
