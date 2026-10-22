"""Настройки из окружения (таблица «Конфигурация» в docs/architecture.md).

Две точки входа:
- load_settings() — для CLI: адрес instudy и токен не обязательны (run-file, models-* работают без них);
- load_daemon_settings() — для демона: без INSTUDY_API_URL и SERVICE_TOKEN не стартует.

Токены хранятся как SecretStr: их нет ни в repr, ни в model_dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MiB = 1024 * 1024


class ConfigError(RuntimeError):
    """Конфигурация неполна или неверна. Текст — для человека, без значений секретов."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    instudy_api_url: str | None = None
    service_token: SecretStr | None = None
    hf_token: SecretStr | None = None  # только для разовой загрузки моделей (models-download)

    poll_interval_sec: int = Field(default=60, gt=0)
    heartbeat_sec: int = Field(default=300, gt=0)
    max_audio_bytes: int = Field(default=300 * MiB, gt=0)
    max_audio_sec: int = Field(default=6 * 3600, gt=0)
    max_segment_sec: float = Field(default=22.0, gt=0, le=24.0)  # GigaAM transcribe — до 25 с
    keep_models_loaded: bool = False
    results_keep_days: int = Field(default=14, ge=0)
    min_free_gb: float = Field(default=10.0, ge=0)

    models_dir: Path = Path("/models")
    data_dir: Path = Path("/data")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("instudy_api_url")
    @classmethod
    def _https_only(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        value = value.strip().rstrip("/")
        if not value.lower().startswith("https://") or len(value) <= len("https://"):
            raise ValueError("INSTUDY_API_URL должен начинаться с https://")
        if "?" in value or "#" in value or "@" in value:
            raise ValueError("INSTUDY_API_URL — только схема, хост и путь, без query, фрагмента и логина")
        return value

    @field_validator("service_token", "hf_token")
    @classmethod
    def _empty_secret_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            return None
        return value

    def secret_values(self) -> list[str]:
        """Значения секретов — для фильтра логов (transkrib.log.register_secret)."""
        return [s.get_secret_value() for s in (self.service_token, self.hf_token) if s is not None]


def _describe(exc: ValidationError) -> str:
    # Только имя поля и сообщение — без input, чтобы значение секрета не попало в текст ошибки.
    parts = []
    for err in exc.errors(include_input=False, include_url=False):
        field = ".".join(str(p) for p in err.get("loc", ())) or "?"
        parts.append(f"{field.upper()}: {err.get('msg', 'неверное значение')}")
    return "; ".join(parts)


def load_settings(**overrides: object) -> Settings:
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        raise ConfigError(f"Неверная конфигурация: {_describe(exc)}") from None


def load_daemon_settings(**overrides: object) -> Settings:
    settings = load_settings(**overrides)
    missing = []
    if settings.instudy_api_url is None:
        missing.append("INSTUDY_API_URL")
    if settings.service_token is None:
        missing.append("SERVICE_TOKEN")
    if missing:
        raise ConfigError(f"Для работы демона задайте в .env: {', '.join(missing)}")
    return settings
