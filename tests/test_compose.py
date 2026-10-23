"""Проверки compose.yml и образа без Docker: что уходит в контейнеры и как они ограничены (T-7)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "compose.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    return compose["services"]


def env(service: dict) -> dict:
    value = service.get("environment") or {}
    if isinstance(value, list):
        return dict(item.split("=", 1) if "=" in item else (item, None) for item in value)
    return value


def volumes(service: dict) -> list[str]:
    return [v if isinstance(v, str) else f"{v['source']}:{v['target']}" for v in service.get("volumes", [])]


def test_services(services: dict) -> None:
    assert set(services) == {"transkrib", "models"}


def test_no_ports_anywhere(services: dict) -> None:
    for name, service in services.items():
        assert "ports" not in service, name
        assert "expose" not in service, name
        assert service.get("network_mode") != "host", name


def test_no_env_file(services: dict) -> None:
    for name, service in services.items():
        assert "env_file" not in service, name


def test_main_service_hardening(services: dict) -> None:
    main = services["transkrib"]
    assert main["read_only"] is True
    assert str(main["user"]).split(":")[0] == "10001"
    assert main["gpus"] == "all"
    assert main["restart"] == "unless-stopped"
    assert main["stop_grace_period"] == "45s"
    assert any(str(t).split(":")[0] == "/tmp" for t in main["tmpfs"])  # noqa: S108 — tmpfs в контейнере
    assert main["healthcheck"]["test"] == ["CMD", "python", "-m", "transkrib.cli", "health"]
    assert main["logging"] == {"driver": "json-file", "options": {"max-size": "10m", "max-file": "5"}}


def test_main_volumes(services: dict) -> None:
    vols = volumes(services["transkrib"])
    assert "./models:/models:ro" in vols
    assert "./data:/data" in vols
    assert len(vols) == 2


def test_models_service_writes_models_only(services: dict) -> None:
    models = services["models"]
    assert models["profiles"] == ["models"]
    assert models["read_only"] is True
    assert str(models["user"]).split(":")[0] == "10001"
    assert volumes(models) == ["./models:/models"]
    assert "gpus" not in models


def test_tokens_are_separated(services: dict) -> None:
    main, models = env(services["transkrib"]), env(services["models"])
    assert "SERVICE_TOKEN" in main and "HF_TOKEN" not in main
    assert "HF_TOKEN" in models and "SERVICE_TOKEN" not in models
    assert "INSTUDY_API_URL" not in models
    # HF_TOKEN не попадает в основной сервис и через подстановку в другие поля
    main_text = yaml.safe_dump(services["transkrib"])
    assert "HF_TOKEN" not in main_text
    assert "SERVICE_TOKEN" not in yaml.safe_dump(services["models"])


def test_main_env_covers_config_with_defaults(services: dict) -> None:
    """Каждая переменная из .env.example (кроме HF_TOKEN) доходит до демона; у необязательных есть значение
    по умолчанию — пустая строка сломала бы проверку типов в Settings."""
    names = [
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if re.match(r"^[A-Z_]+=", line)
    ]
    main = env(services["transkrib"])
    for name in names:
        if name == "HF_TOKEN":
            continue
        assert name in main, name
        if name not in ("INSTUDY_API_URL", "SERVICE_TOKEN"):
            assert re.fullmatch(rf"\$\{{{name}:-[^}}]+\}}", str(main[name])), (name, main[name])


def test_dockerfile_basics() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"python:3\.12-slim-bookworm@sha256:[0-9a-f]{64}", text)
    assert "USER 10001" in text
    assert 'ENTRYPOINT ["/usr/bin/tini", "--"]' in text
    for var in (
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "PYANNOTE_METRICS_ENABLED=0",
        "PYTHONDONTWRITEBYTECODE=1",
    ):
        assert var in text
    assert "MPLCONFIGDIR=/tmp/matplotlib" in text
    assert "--require-hashes -r requirements-gpu.txt" in text
    assert not re.search(r"^COPY\s+.*models", text, re.M)


def test_gpu_requirements_are_hashed() -> None:
    text = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=[a-z0-9])", text)
    pins = [b for b in blocks if re.match(r"^[a-z0-9][a-z0-9._-]*==", b)]
    assert len(pins) > 50
    for block in pins:
        assert "--hash=sha256:" in block, block.splitlines()[0]
    for name in ("torch==2.10.0", "torchaudio==2.10.0", "torchcodec==0.10.0", "pyannote-audio==4.0.7"):
        assert any(b.startswith(name) for b in pins), name
    assert "git+" not in text  # git-ссылки — только в requirements-gigaam.txt


def test_gigaam_commit_matches_models_store() -> None:
    from transkrib import models_store

    text = (ROOT / "requirements-gigaam.txt").read_text(encoding="utf-8")
    assert f"GigaAM.git@{models_store.GIGAAM_COMMIT}" in text


def test_update_script() -> None:
    text = (ROOT / "deploy" / "update.sh").read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    order = ["git pull --ff-only", "docker compose build", "docker compose up -d", "Health", "transkrib.cli status"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)
