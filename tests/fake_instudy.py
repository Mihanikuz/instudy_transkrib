"""Фейковый instudy для тестов: сервисный API по instudy/docs/transcription-api.md поверх httpx.MockTransport.

Моделирует очередь работ, аренды с подменяемыми часами, heartbeat, подписанные ссылки (job, exp, sig; гаснут вместе
с арендой), audioSha256, Range, ручную загрузку (409 на result, 410 на heartbeat), исчезнувший день (410), неверный
токен (401), лимит (429 + Retry-After), паузу выдачи (health paused, jobs/next 204), лимиты результата (413) и его
проверку тем же валидатором, что в instudy (tests/instudy_reference.py, 422), сбои по сценарию и обрывы скачивания.
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

from tests.instudy_reference import ResultError, check_vtt, decode_strict, parse_meta, parse_multipart

BASE_URL = "https://api.example.ru"
TOKEN = "svc_tok_fake_1234567890abcdef"
PREFIX = "/api/service/v1"
LEASE = timedelta(minutes=30)
LINK_TTL = timedelta(minutes=60)
MiB = 1024 * 1024
FILE_MAX, META_MAX, BODY_MAX = 10 * MiB, 4096, 50 * MiB
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
    sha_override: str | None = None  # прислать в работе неверный audioSha256
    leased_at: datetime | None = None  # время выдачи аренды — входит в подпись ссылки
    transcript_id: int | None = None


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
        self.paused = False  # autotx-pause: health → paused, jobs/next → 204
        self.next_transcript_id = 1534
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
        issued = int(job.leased_at.timestamp()) if job.leased_at else 0
        msg = f"audio\n{job.id}\n{job.day_id}\n{int(exp.timestamp())}\n{issued}".encode()
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
            server_time = self.now.isoformat(timespec="seconds").replace("+00:00", "Z")
            return httpx.Response(
                200, json={"ok": True, "serverTime": server_time, "client": "home-gpu", "paused": self.paused}
            )
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
        if self.paused:
            return httpx.Response(204)
        for job in sorted(self.jobs.values(), key=lambda j: (j.date, j.id)):
            expired = job.status == "leased" and not self._lease_alive(job)
            if (job.status == "queued" or expired) and not job.gone and not job.manual_transcript:
                job.status, job.leased_until, job.leased_at = "leased", self.now + LEASE, self.now
                exp = self.now + LINK_TTL
                query = f"job={job.id}&exp={int(exp.timestamp())}&sig={self.sign(job, exp)}"
                url = f"{BASE_URL}{PREFIX}/audio/{job.day_id}?{query}"
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
                            "audioSha256": job.sha_override or hashlib.sha256(job.audio).hexdigest(),
                            "leaseUntil": job.leased_until.isoformat().replace("+00:00", "Z"),
                        }
                    },
                )
        return httpx.Response(204)

    def _heartbeat(self, job: FakeJob, request: httpx.Request, body: bytes) -> httpx.Response:
        # Потеря работы — только 410, причина — в detail после двоеточия (409 instudy на heartbeat не отдаёт).
        if not self._lease_alive(job):
            return httpx.Response(410, json={"detail": "Работа больше не ваша"})
        if job.gone:
            job.status = "skipped"
            return httpx.Response(410, json={"detail": "Работа больше не ваша: день исчез из расписания или скрыт"})
        if job.manual_transcript:
            job.status = "skipped"
            return httpx.Response(410, json={"detail": "Работа больше не ваша: у дня появился транскрипт"})
        job.leased_until = self.now + LEASE
        return httpx.Response(200, json={"leaseUntil": job.leased_until.isoformat().replace("+00:00", "Z")})

    def _result(self, job: FakeJob, request: httpx.Request, body: bytes) -> httpx.Response:
        # Порядок проверок — как в instudy (transcription-api.md, «POST …/result»).
        if not self._lease_alive(job):
            return httpx.Response(410, json={"detail": "Работа больше не ваша"})
        if job.gone:
            job.status = "skipped"
            return httpx.Response(410, json={"detail": "Работа больше не ваша: день исчез из расписания или скрыт"})
        if job.manual_transcript:
            job.status = "skipped"
            return httpx.Response(409, json={"detail": "У дня уже есть транскрипт"})
        ctype = request.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return httpx.Response(415, json={"detail": "Нужен multipart/form-data"})
        if len(body) > BODY_MAX:
            return httpx.Response(413, json={"detail": "Тело больше 50 МБ"})
        parts = parse_multipart(ctype, body)
        if parts is None:
            return httpx.Response(400, json={"detail": "Некорректные данные формы"})
        if set(parts) != {"vtt", "txt", "meta"}:
            return httpx.Response(422, json={"detail": "Нужны ровно поля vtt, txt и meta"})
        if len(parts["vtt"][1]) > FILE_MAX or len(parts["txt"][1]) > FILE_MAX or len(parts["meta"][1]) > META_MAX:
            return httpx.Response(413, json={"detail": "Файл больше 10 МБ или meta больше 4 КБ"})
        try:
            check_vtt(decode_strict(parts["vtt"][1], "vtt"))
            if not decode_strict(parts["txt"][1], "txt").strip():
                raise ResultError("txt: файл пустой")
            parse_meta(parts["meta"][1].decode("utf-8"))
        except (ResultError, UnicodeDecodeError) as exc:
            return httpx.Response(422, json={"detail": str(exc)})
        job.result = {"raw": body, "content_type": ctype, "parts": parts}
        job.status = "done"
        job.transcript_id = self.next_transcript_id
        self.next_transcript_id += 1
        return httpx.Response(202, json={"transcriptId": job.transcript_id})

    def _fail(self, job: FakeJob, request: httpx.Request, body: bytes) -> httpx.Response:
        if not self._lease_alive(job):
            return httpx.Response(410, json={"detail": "Работа больше не ваша"})
        data = json.loads(body)
        job.fails.append(data)
        job.leased_until = None
        # shutdown попытку не засчитывает; retryable: false — сразу failed; третья засчитанная неудача — failed.
        if data.get("reason") != "shutdown":
            job.attempts += 1
        if data.get("retryable") is False or job.attempts >= 3:
            job.status = "failed"
        else:
            job.status = "queued"
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
        try:
            link_job = int(params["job"][0])
        except (KeyError, ValueError):
            return httpx.Response(403, json={"detail": "Ссылка недействительна или устарела"})
        if (
            job is None
            or link_job != job.id
            or exp <= self.now
            or not self._lease_alive(job)
            or not hmac.compare_digest(sig, self.sign(job, exp))
        ):
            return httpx.Response(403, json={"detail": "Ссылка недействительна или устарела"})
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
