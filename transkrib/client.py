"""Клиент сервисного API instudy (docs/contract.md).

Клиент только выполняет запросы и классифицирует исходы типизированными исключениями. Повторы, задержки и решение,
слать ли fail, — у вызывающего кода (демон, T-6).

Безопасность:
- Authorization ставится только на запросы к INSTUDY_API_URL; скачивание по audioUrl — отдельным клиентом без него.
- audioUrl принимается только с того же https-хоста, что и INSTUDY_API_URL, — иначе работа невалидна.
- В тексты исключений не попадают ни токен, ни audioUrl; message для fail проходит фильтр секретов и обрезается.
- Ответы instudy — недоверенные данные: работа проверяется pydantic-моделью с лимитами из конфига.
"""

from __future__ import annotations

import email.utils
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from datetime import date as Date
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from transkrib import __version__
from transkrib.log import redact

logger = logging.getLogger(__name__)

API_PREFIX = "/api/service/v1"
MESSAGE_MAX = 300
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


# ------------------------------------------------------------------------------- исходы


class FailReason(StrEnum):
    DOWNLOAD_FAILED = "download_failed"
    DECODE_FAILED = "decode_failed"
    NO_SPEECH = "no_speech"
    OOM = "oom"
    MODEL_ERROR = "model_error"
    INVALID_JOB = "invalid_job"
    RESULT_TOO_LARGE = "result_too_large"
    RESULT_REJECTED = "result_rejected"
    SHUTDOWN = "shutdown"


RETRYABLE: dict[FailReason, bool] = {
    FailReason.DOWNLOAD_FAILED: True,
    FailReason.DECODE_FAILED: False,
    FailReason.NO_SPEECH: False,
    FailReason.OOM: True,
    FailReason.MODEL_ERROR: True,
    FailReason.INVALID_JOB: False,
    FailReason.RESULT_TOO_LARGE: False,
    FailReason.RESULT_REJECTED: False,
    FailReason.SHUTDOWN: True,
}


class ResultOutcome(StrEnum):
    ACCEPTED = "accepted"  # 202 — instudy обработает как обычный транскрипт
    ALREADY_HAS_TRANSCRIPT = "already_has_transcript"  # 409 — транскрипт загрузили вручную, работа закрыта
    GONE = "gone"  # 410 — работа не наша, аренда истекла или день исчез


class ClientError(Exception):
    """База ошибок клиента. Текст — без токенов и ссылок."""


class Unauthorized(ClientError):
    """401: токен неверный или отозван."""


class RateLimited(ClientError):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"Лимит запросов instudy, повтор через {retry_after:.0f} с")
        self.retry_after = retry_after


class Transient(ClientError):
    """5xx, сеть, таймаут, неожиданный ответ — повторить позже."""


class LeaseLost(ClientError):
    """409/410 на heartbeat: работа больше не наша."""


class LinkExpired(ClientError):
    """403 на audioUrl: срок подписанной ссылки истёк."""


class InvalidJob(ClientError):
    """Работа не прошла проверку. job_id — если его удалось извлечь, чтобы отправить fail invalid_job."""

    def __init__(self, message: str, job_id: int | None = None) -> None:
        super().__init__(message)
        self.job_id = job_id


class DownloadFailed(ClientError):
    """Аудио не скачалось или его размер не совпал с audioSizeBytes."""


class Cancelled(ClientError):
    """Скачивание остановлено флагом отмены."""


class Rejected(ClientError):
    """instudy не принял результат (413, 415, 422): повтор не поможет."""

    def __init__(self, status: int, reason: FailReason) -> None:
        super().__init__(f"instudy не принял результат: HTTP {status}")
        self.status = status
        self.reason = reason


# --------------------------------------------------------------------------------- работа


class Job(BaseModel):
    """Работа из GET /jobs/next. audioUrl не попадает в repr."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    id: int = Field(gt=0, strict=True)
    day_id: int = Field(alias="dayId", gt=0, strict=True)
    date: Date
    discipline: str = Field(default="", max_length=500)
    audio_url: str = Field(alias="audioUrl", repr=False, max_length=4096)
    audio_size_bytes: int = Field(alias="audioSizeBytes", gt=0, strict=True)
    audio_duration_sec: int | None = Field(default=None, alias="audioDurationSec", gt=0, strict=True)
    lease_until: datetime = Field(alias="leaseUntil")

    @field_validator("lease_until")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=UTC)


# --------------------------------------------------------------------------------- клиент


_PATH_RX = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[\w.@-]+[\\/])+[\w.@-]*")


def sanitize_message(message: str) -> str:
    """Текст причины отказа для instudy: без секретов, ссылок, путей и переводов строк, не длиннее 300 символов."""
    text = redact(message or "")
    text = re.sub(r"https?://\S+", "<url>", text)
    text = _PATH_RX.sub("<path>", text)
    text = " ".join(text.split())
    return text[:MESSAGE_MAX]


def _retry_after(response: httpx.Response) -> float:
    value = response.headers.get("Retry-After", "").strip()
    if value.isdigit():
        return float(value)
    if value:
        parsed = email.utils.parsedate_to_datetime(value) if "," in value else None
        if parsed is not None:
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
    return 60.0


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return parts.scheme.lower(), (parts.hostname or "").lower() + (f":{parts.port}" if parts.port else "")


class InstudyClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        max_audio_bytes: int,
        max_audio_sec: int,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if _origin(base_url)[0] != "https":
            raise ValueError("INSTUDY_API_URL должен быть https://")
        self.base_url = base_url.rstrip("/")
        self.max_audio_bytes = max_audio_bytes
        self.max_audio_sec = max_audio_sec
        self._sleep = sleep
        user_agent = f"instudy_transkrib/{__version__}"
        self._api = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            verify=True,
        )
        # Отдельный клиент без Authorization: подписанная ссылка не должна получить токен.
        self._media = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            verify=True,
        )

    # ----------------------------------------------------------------- служебное

    def close(self) -> None:
        self._api.close()
        self._media.close()

    def __enter__(self) -> InstudyClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._api.request(method, f"{API_PREFIX}{path}", **kwargs)
        except httpx.TimeoutException:
            raise Transient(f"Таймаут запроса {method} {path}") from None
        except httpx.TransportError as exc:
            raise Transient(f"Сетевая ошибка {method} {path}: {type(exc).__name__}") from None
        status = response.status_code
        if status == 401:
            raise Unauthorized("instudy отклонил сервисный токен (401)")
        if status == 429:
            raise RateLimited(_retry_after(response))
        if status >= 500:
            raise Transient(f"instudy ответил {status} на {method} {path}")
        return response

    @staticmethod
    def _unexpected(response: httpx.Response, what: str) -> Transient:
        return Transient(f"Неожиданный ответ instudy на {what}: HTTP {response.status_code}")

    @staticmethod
    def _json(response: httpx.Response, what: str) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise Transient(f"instudy вернул не JSON на {what}") from None

    # ----------------------------------------------------------------- методы API

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise self._unexpected(response, "health")
        data = self._json(response, "health")
        if not isinstance(data, dict):
            raise Transient("instudy вернул неверный ответ на health")
        return data

    def next_job(self) -> Job | None:
        response = self._request("GET", "/jobs/next")
        if response.status_code == 204:
            return None
        if response.status_code != 200:
            raise self._unexpected(response, "jobs/next")
        data = self._json(response, "jobs/next")
        raw = data.get("job") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            raise InvalidJob("В ответе jobs/next нет объекта job")
        raw_id = raw.get("id")
        job_id = raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0 else None
        try:
            job = Job.model_validate(raw)
        except ValidationError as exc:
            fields = ", ".join(sorted({str(e["loc"][0]) for e in exc.errors(include_input=False) if e.get("loc")}))
            raise InvalidJob(f"Работа не прошла проверку: {fields or 'формат'}", job_id) from None
        self._check_job(job)
        return job

    def _check_job(self, job: Job) -> None:
        if job.audio_size_bytes > self.max_audio_bytes:
            raise InvalidJob(f"Аудио больше лимита: {job.audio_size_bytes} байт", job.id)
        if job.audio_duration_sec is not None and job.audio_duration_sec > self.max_audio_sec:
            raise InvalidJob(f"Аудио длиннее лимита: {job.audio_duration_sec} с", job.id)
        if _origin(job.audio_url) != _origin(self.base_url):
            raise InvalidJob("Ссылка на аудио ведёт не на сервер instudy", job.id)

    def heartbeat(self, job_id: int) -> datetime:
        response = self._request("POST", f"/jobs/{int(job_id)}/heartbeat")
        if response.status_code in (409, 410):
            raise LeaseLost(f"Аренда работы {job_id} потеряна (HTTP {response.status_code})")
        if response.status_code != 200:
            raise self._unexpected(response, "heartbeat")
        data = self._json(response, "heartbeat")
        try:
            value = datetime.fromisoformat(str(data["leaseUntil"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            raise Transient("instudy вернул неверный leaseUntil") from None
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def download(
        self,
        job: Job,
        dest: Path,
        *,
        cancel: threading.Event | None = None,
        max_attempts: int = 5,
        backoff: float = 5.0,
    ) -> Path:
        """Скачать аудио работы в dest с докачкой по Range. Возвращает dest."""
        expected = job.audio_size_bytes
        dest.parent.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled("Скачивание отменено")
            have = dest.stat().st_size if dest.exists() else 0
            if have > expected:
                dest.unlink()
                have = 0
            if have == expected:
                break
            attempt += 1
            if attempt > max_attempts:
                raise DownloadFailed(f"Аудио не скачалось за {max_attempts} попыток: {have} из {expected} байт")
            try:
                self._download_once(job, dest, have, expected, cancel)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                got = dest.stat().st_size if dest.exists() else 0
                logger.warning(
                    "Обрыв скачивания аудио",
                    extra={"job_id": job.id, "attempt": attempt, "bytes": got, "error": type(exc).__name__},
                )
                if attempt < max_attempts:
                    self._sleep(backoff * attempt)
            except _Retry:
                if attempt < max_attempts:
                    self._sleep(backoff * attempt)
        size = dest.stat().st_size
        if size != expected:
            raise DownloadFailed(f"Размер аудио {size} не совпал с ожидаемым {expected}")
        return dest

    def _download_once(self, job: Job, dest: Path, have: int, expected: int, cancel: threading.Event | None) -> None:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with self._media.stream("GET", job.audio_url, headers=headers) as response:
            status = response.status_code
            if status == 403:
                raise LinkExpired(f"Ссылка на аудио работы {job.id} истекла (403)")
            if status == 416:
                if have != expected:  # сервер считает, что докачивать нечего, а размер не тот — начать заново
                    dest.unlink(missing_ok=True)
                    raise _Retry
                return
            if status >= 500:
                raise _Retry
            if status not in (200, 206):
                raise DownloadFailed(f"Сервер аудио ответил {status}")
            mode = "ab"
            if status == 200 and have:
                mode = "wb"  # Range проигнорирован — пишем с начала
            elif status == 206 and not _content_range_starts_at(response, have):
                dest.unlink(missing_ok=True)
                raise _Retry
            written = have if mode == "ab" else 0
            with dest.open(mode) as fh:
                # Без буферизации: при обрыве на диске остаётся всё, что пришло, — докачка продолжит с этого места.
                for chunk in response.iter_bytes():
                    if cancel is not None and cancel.is_set():
                        raise Cancelled("Скачивание отменено")
                    written += len(chunk)
                    if written > expected:
                        fh.close()
                        dest.unlink(missing_ok=True)
                        raise DownloadFailed(f"Сервер прислал больше {expected} байт")
                    fh.write(chunk)

    def submit_result(self, job_id: int, vtt: bytes, txt: bytes, meta: dict[str, Any]) -> ResultOutcome:
        files = {
            "vtt": ("transcript.vtt", vtt, "text/vtt; charset=utf-8"),
            "txt": ("transcript.txt", txt, "text/plain; charset=utf-8"),
        }
        data = {"meta": json.dumps(meta, ensure_ascii=False, separators=(",", ":"))}
        response = self._request("POST", f"/jobs/{int(job_id)}/result", files=files, data=data)
        status = response.status_code
        if status == 202:
            return ResultOutcome.ACCEPTED
        if status == 409:
            return ResultOutcome.ALREADY_HAS_TRANSCRIPT
        if status == 410:
            return ResultOutcome.GONE
        if status == 413:
            raise Rejected(status, FailReason.RESULT_TOO_LARGE)
        if status in (415, 422):
            raise Rejected(status, FailReason.RESULT_REJECTED)
        raise self._unexpected(response, "result")

    def fail(self, job_id: int, reason: FailReason, message: str, retryable: bool | None = None) -> bool:
        """Сообщить об отказе. False — работа уже не наша (409/410), сообщать некому."""
        reason = FailReason(reason)
        body = {
            "reason": reason.value,
            "retryable": RETRYABLE[reason] if retryable is None else bool(retryable),
            "message": sanitize_message(message),
        }
        response = self._request("POST", f"/jobs/{int(job_id)}/fail", json=body)
        if response.status_code in (200, 202, 204):
            return True
        if response.status_code in (409, 410):
            return False
        raise self._unexpected(response, "fail")


class _Retry(Exception):  # noqa: N818 — внутренний сигнал «повторить попытку скачивания»
    pass


def _content_range_starts_at(response: httpx.Response, start: int) -> bool:
    match = re.match(r"bytes (\d+)-", response.headers.get("Content-Range", ""))
    return bool(match) and int(match.group(1)) == start
