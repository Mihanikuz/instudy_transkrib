"""JSON-логи в stdout и маскировка секретов.

Что маскируется (CLAUDE.md, «Безопасность»):
- значения, зарегистрированные через register_secret() (SERVICE_TOKEN, HF_TOKEN);
- «Bearer <что угодно>» — заголовок авторизации;
- query-строка ссылок на /api/service/…/audio и /api/service/…/media (подписанные ссылки целиком после «?»);
- параметры sig= и exp= в любом тексте.

Маскировка идёт в фильтре обработчика: и в сообщении с подставленными аргументами, и в полях extra, и в тексте
исключения и стека. redact() доступна остальному коду — например, для текста причины отказа, отправляемого в instudy.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import traceback
from datetime import UTC, datetime
from typing import Any

MASK = "***"

_secrets: set[str] = set()
_secrets_lock = threading.Lock()

_BEARER_RX = re.compile(r"(?i)\b(bearer)\s+[^\s\"',;]+")
_SERVICE_URL_QUERY_RX = re.compile(r"(/api/service/[^\s\"'<>?]*)\?[^\s\"'<>]*")
_SIGNED_PARAM_RX = re.compile(r"(?i)\b(sig|exp|signature|token)=[^\s&\"'<>]+")

# Атрибуты LogRecord, которые не являются полями extra.
_STD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}

_MIN_SECRET_LEN = 6  # короткие строки не маскируем, чтобы не портить обычный текст


def register_secret(value: str | None) -> None:
    """Добавить значение, которое никогда не должно попасть в логи."""
    if value and len(value) >= _MIN_SECRET_LEN:
        with _secrets_lock:
            _secrets.add(value)


def clear_secrets() -> None:
    """Для тестов."""
    with _secrets_lock:
        _secrets.clear()


def redact(text: str) -> str:
    """Вернуть текст без секретов."""
    if not text:
        return text
    with _secrets_lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, MASK)
    text = _BEARER_RX.sub(lambda m: f"{m.group(1)} {MASK}", text)
    text = _SERVICE_URL_QUERY_RX.sub(lambda m: f"{m.group(1)}?{MASK}", text)
    text = _SIGNED_PARAM_RX.sub(lambda m: f"{m.group(1)}={MASK}", text)
    return text


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _redact_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_redact_value(v) for v in value]
    return redact(str(value))


class SecretFilter(logging.Filter):
    """Переписывает запись так, чтобы дальше по цепочке секретов уже не было."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # сломанный формат не должен ронять логирование
            message = f"{record.msg!s} {record.args!r}"
        record.msg = redact(message)
        record.args = None

        if record.exc_info:
            text = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = redact(text)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)

        for key, value in list(vars(record).items()):
            if key not in _STD_ATTRS:
                setattr(record, key, _redact_value(value))
        return True


class JsonFormatter(logging.Formatter):
    """Одна строка JSON на запись: ts (UTC), level, logger, msg, поля extra, exc."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STD_ATTRS and key not in payload:
                payload[key] = _redact_value(value)
        if record.exc_info and not record.exc_text:
            record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)))
        if record.exc_text:
            payload["exc"] = redact(record.exc_text)
        if record.stack_info:
            payload["stack"] = redact(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_HANDLER_NAME = "transkrib-stdout"


def setup_logging(level: str = "INFO", secrets: list[str] | None = None, stream: Any = None) -> logging.Handler:
    """Настроить корневой логгер: JSON в stdout с фильтром секретов. Повторный вызов заменяет обработчик."""
    for secret in secrets or []:
        register_secret(secret)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(SecretFilter())
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    # httpx пишет полные URL запросов на INFO — подписанные ссылки туда попадать не должны даже маскированными.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return handler
