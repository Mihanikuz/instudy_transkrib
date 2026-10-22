"""Модели на диске: пути, маркеры готовности, проверка, импорт из gigascribe и разовая загрузка.

Раскладка (совпадает с gigascribe, поэтому его модели можно перенести командой models-import):

    <MODELS_DIR>/gigaam/v3_e2e_rnnt/v3_e2e_rnnt.ckpt
    <MODELS_DIR>/gigaam/v3_e2e_rnnt/v3_e2e_rnnt_tokenizer.model
    <MODELS_DIR>/gigaam/v3_e2e_rnnt/.ready.json
    <MODELS_DIR>/pyannote/community-1/config.yaml (+ подкаталоги снимка репозитория)
    <MODELS_DIR>/pyannote/community-1/.ready.json

check() работает без сети и без torch: только файлы, размеры и (в режиме deep) хэши.
Сеть нужна только download() — по явной команде models-download (сервис загрузки моделей в compose, T-7).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GIGAAM_MODEL = "v3_e2e_rnnt"
# Должен совпадать с коммитом в requirements-gpu.in: другой код GigaAM — повод перепроверить модель.
GIGAAM_COMMIT = "7447938d791c4f3e643386ee22c33777004293a5"
GIGAAM_CKPT_MD5 = "2730de7545ac43ad256485a462b0a27a"  # gigaam/__init__.py, _MODEL_HASHES["v3_e2e_rnnt"]
GIGAAM_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
GIGAAM_FILES = (f"{GIGAAM_MODEL}.ckpt", f"{GIGAAM_MODEL}_tokenizer.model")

PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"
PYANNOTE_REQUIRED = ("config.yaml",)

MARKER = ".ready.json"
MARKER_VERSION = 1
_MIN_BYTES = {GIGAAM_FILES[0]: 1024 * 1024, GIGAAM_FILES[1]: 128}


def gigaam_dir(models_dir: Path) -> Path:
    return Path(models_dir) / "gigaam" / GIGAAM_MODEL


def pyannote_dir(models_dir: Path) -> Path:
    return Path(models_dir) / "pyannote" / "community-1"


@dataclass(frozen=True)
class Problem:
    model: str  # gigaam | pyannote
    message: str

    def __str__(self) -> str:
        return f"{self.model}: {self.message}"


# ------------------------------------------------------------------------------ хэши и маркеры


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return _hash(path, "sha256")


def md5_file(path: Path) -> str:
    return _hash(path, "md5")


def _model_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != MARKER and not p.name.startswith("."))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_marker(root: Path, model: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    files = {str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in _model_files(root)}
    payload = {
        "version": MARKER_VERSION,
        "model": model,
        "files": files,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    _atomic_write_json(root / MARKER, payload)
    return payload


def _read_marker(root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((root / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _check_marked_files(model: str, root: Path, marker: dict[str, Any], deep: bool) -> list[Problem]:
    problems: list[Problem] = []
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        return [Problem(model, f"маркер {root / MARKER} без списка файлов — перепроверьте модель")]
    for rel, info in files.items():
        path = root / rel
        if not path.is_file():
            problems.append(Problem(model, f"нет файла {rel}"))
        elif path.stat().st_size != info.get("size"):
            problems.append(Problem(model, f"размер {rel} не совпадает с маркером"))
        elif deep and sha256_file(path) != info.get("sha256"):
            problems.append(Problem(model, f"хэш {rel} не совпадает с маркером"))
    return problems


# ------------------------------------------------------------------------------- проверка


def check_gigaam(models_dir: Path, *, deep: bool = False) -> list[Problem]:
    root = gigaam_dir(models_dir)
    problems = []
    for name in GIGAAM_FILES:
        path = root / name
        if not path.is_file():
            problems.append(Problem("gigaam", f"нет файла {root / name}"))
        elif path.stat().st_size < _MIN_BYTES[name]:
            problems.append(Problem("gigaam", f"файл {name} подозрительно мал"))
    if problems:
        return problems
    marker = _read_marker(root)
    if marker is None or marker.get("model") != GIGAAM_MODEL:
        return [
            Problem("gigaam", f"нет маркера готовности {root / MARKER} — выполните models-import или models-download")
        ]
    if marker.get("gigaam_commit") != GIGAAM_COMMIT:
        problems.append(
            Problem(
                "gigaam", "модель проверялась с другим коммитом GigaAM — повторите models-import или models-download"
            )
        )
    problems += _check_marked_files("gigaam", root, marker, deep)
    if deep and not problems and md5_file(root / GIGAAM_FILES[0]) != GIGAAM_CKPT_MD5:
        problems.append(Problem("gigaam", "контрольная сумма чекпоинта не совпадает с эталоном GigaAM"))
    return problems


def check_pyannote(models_dir: Path, *, deep: bool = False) -> list[Problem]:
    root = pyannote_dir(models_dir)
    missing = [name for name in PYANNOTE_REQUIRED if not (root / name).is_file()]
    if missing:
        return [Problem("pyannote", f"нет файла {root / name}") for name in missing]
    marker = _read_marker(root)
    if marker is None or marker.get("model") != PYANNOTE_REPO:
        return [
            Problem("pyannote", f"нет маркера готовности {root / MARKER} — выполните models-import или models-download")
        ]
    return _check_marked_files("pyannote", root, marker, deep)


def check(models_dir: Path, *, deep: bool = False) -> list[Problem]:
    """Пустой список — обе модели на месте. Без сети и без torch."""
    return check_gigaam(models_dir, deep=deep) + check_pyannote(models_dir, deep=deep)


# ------------------------------------------------------------------------- импорт и загрузка


def _gigaam_marker(root: Path) -> None:
    ckpt = root / GIGAAM_FILES[0]
    if md5_file(ckpt) != GIGAAM_CKPT_MD5:
        raise RuntimeError("Контрольная сумма чекпоинта GigaAM не совпадает с эталоном — файл повреждён или другой")
    write_marker(root, GIGAAM_MODEL, {"gigaam_commit": GIGAAM_COMMIT, "ckpt_md5": GIGAAM_CKPT_MD5})


def import_models(src_models_dir: Path, models_dir: Path, *, log: Callable[[str], None] = print) -> None:
    """Скопировать модели из каталога models/ gigascribe (та же раскладка) и проверить их."""
    src_models_dir, models_dir = Path(src_models_dir), Path(models_dir)
    src_g, dst_g = gigaam_dir(src_models_dir), gigaam_dir(models_dir)
    for name in GIGAAM_FILES:
        if not (src_g / name).is_file():
            raise FileNotFoundError(f"В {src_g} нет {name}")
    dst_g.mkdir(parents=True, exist_ok=True)
    for name in GIGAAM_FILES:
        log(f"gigaam: копирую {name}")
        shutil.copy2(src_g / name, dst_g / name)
    _gigaam_marker(dst_g)
    log("gigaam: готово")

    src_p, dst_p = pyannote_dir(src_models_dir), pyannote_dir(models_dir)
    if not (src_p / "config.yaml").is_file():
        raise FileNotFoundError(f"В {src_p} нет config.yaml — это не снимок {PYANNOTE_REPO}")
    log("pyannote: копирую снимок")
    shutil.copytree(src_p, dst_p, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*"))
    write_marker(dst_p, PYANNOTE_REPO)
    log("pyannote: готово")


def download_models(models_dir: Path, hf_token: str | None, *, log: Callable[[str], None] = print) -> None:
    """Разовая загрузка из сети. GigaAM — с CDN Сбера, pyannote — с Hugging Face (нужен HF_TOKEN с доступом)."""
    import httpx  # есть в requirements.txt

    root = gigaam_dir(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in GIGAAM_FILES:
        target = root / name
        if target.is_file() and target.stat().st_size >= _MIN_BYTES[name]:
            log(f"gigaam: {name} уже есть")
            continue
        log(f"gigaam: скачиваю {name}")
        tmp = target.with_suffix(target.suffix + ".part")
        with httpx.stream(
            "GET", f"{GIGAAM_URL}/{name}", timeout=httpx.Timeout(60, connect=15), follow_redirects=True
        ) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        os.replace(tmp, target)
    _gigaam_marker(root)
    log("gigaam: готово")

    if not hf_token:
        raise RuntimeError(f"Для загрузки {PYANNOTE_REPO} нужен HF_TOKEN с принятыми условиями модели")
    from huggingface_hub import snapshot_download  # ставится вместе с pyannote.audio

    log("pyannote: скачиваю снимок")
    snapshot_download(PYANNOTE_REPO, local_dir=str(pyannote_dir(models_dir)), token=hf_token)
    write_marker(pyannote_dir(models_dir), PYANNOTE_REPO)
    log("pyannote: готово")


def offline_env() -> None:
    """Переменные офлайн-режима — до импорта pyannote/transformers/huggingface_hub."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
