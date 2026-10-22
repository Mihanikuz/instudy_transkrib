"""Фейковый instudy для тестов: сервисный API по docs/contract.md поверх httpx.MockTransport.

Моделирует очередь работ, аренды с подменяемыми часами, heartbeat, подписанные ссылки с exp, Range, ручную загрузку
(409), исчезнувший день (410), неверный токен (401), лимит (429 + Retry-After), сбои по сценарию и обрывы скачивания.
Каждый запрос пишется в журнал — тесты проверяют заголовки и тела.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx

BASE_URL = "https://api.example.ru"
TOKEN = "svc_tok_fake_1234567890abcdef"
PREFIX = "/api/service/v1"
LEASE = timedelta(minutes=30)
SIGN_KEY = b"fake-sign-key"


@dataclass
class FakeJob:
    id: int
    day_id: int
    date: str
    discipline: str
    audio: bytes
    duration_sec: int = 3600
    status: str = "queued"  # queued | leased | done | failed | skipped
    leased_until: datetime | None = None
    attempts: int = 0
    manual_transcript: bool = False  # студент загрузил свой — result получит 409
    gone: bool = False  # день исчез — heartbeat/result получат 410
    result: dict | None = None
    fails: list[dict] = field(default_factory=list)
    size_override: int | None = None  # прислать в работе неверный audioSizeBytes


@dataclass
class Logged:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


class _BrokenStream(httpx.SyncByteStream):
    """Отдаёт первые cut байт и рвёт соединение."""

    def __init__(self, data: bytes, cut: int) -> None:
        self.data, self.cut = data, cut

    def __iter__(self) -> Iterator[bytes]:
        yield self.data[: self.cut]
        raise httpx.ReadError("connection reset by peer")


class FakeInstudy:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
        self.jobs: dict[int, FakeJob] = {}
        self.log: list[Logged] = []
        self.token = TOKEN
        self.failures: dict[str, list[int]] = {}  # «METHOD путь-шаблон» → очередь статусов для подмены
        self.rate_limited: int = 0  # сколько следующих запросов получат 429
        self.break_download_at: list[int] = []  # обрывы скачивания: после скольких байт
        self.ignore_range = False
        self.transport = httpx.MockTransport(self.handle)

    # ------------------------------------------------------------------ сценарии

    def add_job(
        self,
        job_id: int,
        *,
        day_id: int | None = None,
        date: str = "2026-09-29",
        size: int = 3_000_000,
        discipline: str = "Общая психология",
        duration_sec: int = 3600,
    ) -> FakeJob:
        pattern = bytes((i * 31 + job_id) % 251 for i in range(251))
        audio = (pattern * (size // 251 + 1))[:size]
        job = FakeJob(job_id, day_id or job_id + 400, date, discipline, audio, duration_sec)
        self.jobs[job_id] = job
        return job

    def fail_next(self, key: str, *statuses: int) -> None:
        """Подменить ответы: key — например 'POST result' или 'GET audio'."""
        self.failures.setdefault(key, []).extend(statuses)

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)

    def requests(self, method: str | None = None, contains: str = "") -> list[Logged]:
        return [r for r in self.log if (method is None or r.method == method) and contains in r.path]

    def sign(self, job: FakeJob, exp: datetime) -> str:
        msg = f"{job.day_id}:{int(exp.timestamp())}".encode()
        return hmac.new(SIGN_KEY, msg, hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------ обработчик

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        url = urlsplit(str(request.url))
        self.log.append(
            Logged(request.method, url.path + (f"?{url.query}" if url.query else ""), dict(request.headers), body)
        )
        path = url.path
        if path.startswith(f"{PREFIX}/audio/"):
            return self._audio(request, path, url.query)
        if not path.startswith(PREFIX):
            return httpx.Response(404, json={"detail": "Не найдено"})
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return httpx.Response(401, json={"detail": "Нет доступа"})
        if self.rate_limited:
            self.rate_limited -= 1
            return httpx.Response(429, headers={"Retry-After": "17"}, json={"detail": "Слишком много запросов"})
        rest = path[len(PREFIX) :]
        key = self._key(request.method, rest)
        if self.failures.get(key):
            return httpx.Response(self.failures[key].pop(0), json={"detail": "сбой по сценарию"})
        if rest == "/health" and request.method == "GET":
            return httpx.Response(200, json={"ok": True, "serverTime": self.now.isoformat(), "client": "home-gpu"})
        if rest == "/jobs/next" and request.method == "GET":
            return self._next()
        m = re.fullmatch(r"/jobs/(\d+)/(heartbeat|result|fail)", rest)
        if m and request.method == "POST":
            job = self.jobs.get(int(m.group(1)))
            if job is None:
                return httpx.Response(404, json={"detail": "Нет такой работы"})
            return getattr(self, f"_{m.group(2)}")(job, request, body)
        return httpx.Response(404, json={"detail": "Не найдено"})

    @staticmethod
    def _key(method: str, rest: str) -> str:
        if rest.startswith("/jobs/") and rest.count("/") == 3:
            return f"{method} {rest.rsplit('/', 1)[1]}"
        return f"{method} {rest.strip('/')}"

    def _lease_alive(self, job: FakeJob) -> bool:
        return job.status == "leased" and job.leased_until is not None and job.leased_until > self.now

    def _next(self) -> httpx.Response:
        for job in sorted(self.jobs.values(), key=lambda j: (j.date, j.id)):
            expired = job.status == "leased" and not self._lease_alive(job)
            if (job.status == "queued" or expired) and not job.gone and not job.manual_transcript:
                job.status, job.leased_until = "leased", self.now + LEASE
                exp = job.leased_until
                url = f"{BASE_URL}{PREFIX}/audio/{job.day_id}?exp={int(exp.timestamp())}&sig={self.sign(job, exp)}"
                return httpx.Response(
                    200,
                    json={
                        "job": {
                            "id": job.id,
                            "dayId": job.day_id,
                            "date": job.date,
                            "discipline": job.discipline,
                            "audioUrl": url,
                            "audioSizeBytes": job.size_override or len(job.audio),
                            "audioDurationSec": job.duration_sec,
                            "leaseUntil": job.leased_until.isoformat().replace("+00:00", "Z"),
                        }
                    },
                )
        return httpx.Response(204)

    def _heartbeat(self, job: FakeJob, request: httpx.Request, body: bytes) -> httpx.Response:
        if job.gone:
            return httpx.Response(410, json={"detail": "День исчез"})
        if not self._lease_alive(job) or job.manual_transcript:
            return httpx.Response(409, json={"detail": "Работа не ваша"})
        job.leased_until = self.now + LEASE
        return httpx.Response(200, json={"leaseUntil": job.leased_until.isoformat().replace("+00:00", "Z")})

    def _result(self, job: FakeJob, request: httpx.Request, body: bytes) -> httpx.Response:
        if job.gone or not self._lease_alive(job):
            job.status = "skipped" if job.gone else job.status
            return httpx.Response(410, json={"detail": "Работа не ваша"})
        if job.manual_transcript:
            job.status = "done"
            return httpx.Response(409, json={"detail": "Транскрипт уже загрузил(а) Анна К."})
        ctype = request.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return httpx.Response(415, json={"detail": "Нужен multipart"})
        job.result = {"raw": body, "content_type": ctype}
        job.status = "done"
        return httpx.Response(202, json={"transcript": {"status": "processing"}})

    def _fail(self, job: FakeJob, request: httpx.Request, body: bytes) -> httpx.Response:
        if not self._lease_alive(job):
            return httpx.Response(409, json={"detail": "Работа не ваша"})
        job.fails.append(json.loads(body))
        job.attempts += 1
        job.status = "failed" if job.attempts >= 3 else "queued"
        job.leased_until = None
        return httpx.Response(204)

    def _audio(self, request: httpx.Request, path: str, query: str) -> httpx.Response:
        day_id = int(path.rsplit("/", 1)[1])
        job = next((j for j in self.jobs.values() if j.day_id == day_id), None)
        params = parse_qs(query)
        try:
            exp = datetime.fromtimestamp(int(params["exp"][0]), UTC)
            sig = params["sig"][0]
        except (KeyError, ValueError):
            return httpx.Response(403, json={"detail": "Ссылка недействительна"})
        if job is None or exp <= self.now or not hmac.compare_digest(sig, self.sign(job, exp)):
            return httpx.Response(403, json={"detail": "Ссылка недействительна"})
        if self.failures.get("GET audio"):
            return httpx.Response(self.failures["GET audio"].pop(0))
        data = job.audio
        start = 0
        rng = request.headers.get("Range")
        if rng and not self.ignore_range:
            m = re.fullmatch(r"bytes=(\d+)-", rng)
            start = int(m.group(1)) if m else 0
            if start >= len(data):
                return httpx.Response(416, headers={"Content-Range": f"bytes */{len(data)}"})
        part = data[start:]
        headers = {"Accept-Ranges": "bytes", "Content-Type": "audio/mp4"}
        status = 200
        if rng and not self.ignore_range:
            status = 206
            headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        if self.break_download_at:
            cut = self.break_download_at.pop(0)
            return httpx.Response(status, headers=headers, stream=_BrokenStream(part, cut))
        return httpx.Response(status, headers=headers, content=part)
