"""data/state.json: что демон делает сейчас (docs/architecture.md, «Состояние»).

Файл читают `cli status` и `cli health` (healthcheck Docker по свежести heartbeat_at). Запись атомарная: временный
файл в том же каталоге + os.replace, поэтому читатель никогда не видит половину JSON.

В файл попадают только служебные поля: номера работы и дня, дата, этап, счётчики, машинный код ошибки. Ни токена,
ни ссылок на аудио, ни названий дисциплин, ни текстов транскриптов.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from transkrib import __version__

STATE_FILE = "state.json"
PAUSE_FILE = "PAUSE"

STATUS_IDLE = "idle"
STATUS_PAUSED = "paused"
STATUS_WORKING = "working"
STATUS_ERROR = "error"

COUNTERS = ("done", "failed", "skipped", "cancelled")
PROGRESS_WRITE_SEC = 5.0  # прогресс ASR приходит на каждый кусок — пишем не чаще


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_state(data_dir: Path) -> dict[str, Any] | None:
    """Прочитать state.json; None — файла нет или он не разбирается."""
    try:
        data = json.loads((Path(data_dir) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class StateStore:
    """Состояние демона в памяти + атомарная запись на диск. Потокобезопасно (главный поток, heartbeat, тикер)."""

    def __init__(
        self,
        data_dir: Path,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        observer: Callable[[Path], None] | None = None,
    ) -> None:
        self.path = Path(data_dir) / STATE_FILE
        self._now = now
        self._observer = observer  # для тестов: вызывается после каждой записи
        self._lock = threading.RLock()
        self._last_progress_write = 0.0
        started = iso(now())
        self._state: dict[str, Any] = {
            "pid": os.getpid(),
            "version": __version__,
            "started_at": started,
            "heartbeat_at": started,
            "status": STATUS_IDLE,
            "reason": None,
            "next_check_at": None,  # когда демон снова проверит модели (статус error: models_load / model_breaker)
            "job": None,
            "counters": dict.fromkeys(COUNTERS, 0),
            "last_error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def _write(self) -> None:
        self._state["heartbeat_at"] = iso(self._now())
        write_json_atomic(self.path, self._state)
        if self._observer is not None:
            self._observer(self.path)

    def touch(self) -> None:
        with self._lock:
            self._write()

    def last_write(self) -> datetime:
        with self._lock:
            return parse_iso(self._state["heartbeat_at"])

    def set_status(self, status: str, reason: str | None = None, *, next_check_at: datetime | None = None) -> None:
        with self._lock:
            next_check = iso(next_check_at) if next_check_at is not None else None
            if (self._state["status"], self._state["reason"], self._state["next_check_at"]) == (
                status,
                reason,
                next_check,
            ):
                return
            self._state["status"], self._state["reason"] = status, reason
            self._state["next_check_at"] = next_check
            self._write()

    def error(self, code: str) -> None:
        """Последняя ошибка: машинный код (reason из контракта или служебный: unauthorized, models, …)."""
        with self._lock:
            self._state["last_error"] = {"code": str(code), "at": iso(self._now())}
            self._write()

    def start_job(self, job_id: int, day_id: int, job_date: date) -> None:
        with self._lock:
            self._state["status"], self._state["reason"] = STATUS_WORKING, None
            self._state["next_check_at"] = None
            self._state["job"] = {
                "id": int(job_id),
                "dayId": int(day_id),
                "date": job_date.isoformat(),
                "stage": "lease",
                "progress": 0.0,
                "started_at": iso(self._now()),
            }
            self._write()

    def progress(self, stage: str, fraction: float) -> None:
        with self._lock:
            job = self._state["job"]
            if job is None:
                return
            changed = job["stage"] != stage
            job["stage"], job["progress"] = stage, round(float(fraction), 3)
            now = time.monotonic()
            if changed or now - self._last_progress_write >= PROGRESS_WRITE_SEC:
                self._last_progress_write = now
                self._write()

    def finish_job(self, outcome: str) -> None:
        """outcome — один из COUNTERS."""
        with self._lock:
            self._state["counters"][outcome] += 1
            self._state["job"] = None
            self._state["status"], self._state["reason"] = STATUS_IDLE, None
            self._state["next_check_at"] = None
            self._write()
