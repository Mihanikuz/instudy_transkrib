"""Аудио: ffmpeg (любой формат → WAV 16 кГц моно PCM16), длительность через ffprobe, чтение WAV и RMS.

ffmpeg и ffprobe запускаются в своей группе процессов: при отмене или таймауте убивается вся группа, чтобы
после работы не оставалось декодеров. Ошибка декодирования → PipelineError("decode_failed"); stderr ffmpeg
пишется в лог без путей пользователя — вместо пути остаётся только имя каталога работы и имя файла.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

from transkrib.asr import SAMPLE_RATE
from transkrib.types import PipelineCancelled, PipelineError

logger = logging.getLogger(__name__)

DECODE_TIMEOUT_SEC = 1800.0  # 5 часов m4a декодируются за минуты; таймаут — от зависшего процесса
PROBE_TIMEOUT_SEC = 60.0
_POLL_SEC = 0.2
_STDERR_MAX_CHARS = 2000

_ABS_PATH_RX = re.compile(r"(?<![\w.])/(?:[^\s/:'\"]+/)+([^\s/:'\"]*)")


def _job_name(path: Path) -> str:
    return path.parent.name or "-"


def _sanitize(text: str, paths: tuple[Path, ...]) -> str:
    """Убрать пути пользователя: известные файлы → «<каталог работы>/<файл>», прочие абсолютные пути → имя файла."""
    replacements: dict[str, str] = {}
    for path in paths:
        short = f"{_job_name(path)}/{path.name}"
        for variant in {str(path), str(path.resolve())}:
            replacements[variant] = short
    for full in sorted(replacements, key=len, reverse=True):
        text = text.replace(full, replacements[full])
    text = _ABS_PATH_RX.sub(lambda m: m.group(1), text)
    text = text.strip()
    return text[-_STDERR_MAX_CHARS:]


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover — SIGKILL не игнорируется
        logger.error("Процесс ffmpeg не завершился после SIGKILL", extra={"pid": proc.pid})


def _run(
    cmd: list[str], *, timeout: float, cancel: threading.Event | None, paths: tuple[Path, ...], what: str
) -> tuple[int, bytes, bytes]:
    """Запустить процесс в своей группе; вернуть (код, stdout, stderr). Отмена → PipelineCancelled."""
    try:
        proc = subprocess.Popen(  # noqa: S603 — команда собирается из констант и путей, без shell
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
    except OSError as exc:
        raise PipelineError("decode_failed", f"{what} не запустился: {type(exc).__name__}") from None
    deadline = time.monotonic() + timeout
    out: list[bytes] = []
    err: list[bytes] = []
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=_POLL_SEC)
                out.append(stdout or b"")
                err.append(stderr or b"")
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                _kill_group(proc)
                raise PipelineCancelled()
            if time.monotonic() > deadline:
                _kill_group(proc)
                logger.warning(f"{what} не уложился в таймаут", extra={"job": _job_name(paths[0]), "timeout": timeout})
                raise PipelineError("decode_failed", f"{what} не уложился в {timeout:.0f} с")
    except BaseException:
        if proc.poll() is None:
            _kill_group(proc)
        raise
    return proc.returncode, b"".join(out), b"".join(err)


def _ffmpeg_cmd(src: Path, dst: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(dst),
    ]


def to_wav16k_mono(
    src: Path, dst: Path, *, cancel: threading.Event | None = None, timeout: float = DECODE_TIMEOUT_SEC
) -> Path:
    """Перекодировать src в WAV PCM16 16 кГц моно. Пишет во временный файл рядом и переименовывает в dst."""
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise PipelineError("decode_failed", "Аудиофайл не найден")
    tmp = dst.with_name(dst.name + ".part")
    paths = (src, dst, tmp)
    try:
        code, _, stderr = _run(_ffmpeg_cmd(src, tmp), timeout=timeout, cancel=cancel, paths=paths, what="ffmpeg")
        if code != 0 or not tmp.is_file():
            logger.warning(
                "ffmpeg не смог разобрать аудио",
                extra={
                    "job": _job_name(src),
                    "returncode": code,
                    "stderr": _sanitize(stderr.decode("utf-8", "replace"), paths),
                },
            )
            raise PipelineError("decode_failed", "ffmpeg не смог разобрать аудио")
        try:
            frames = wav_frames(tmp)
        except (wave.Error, EOFError, ValueError) as exc:
            raise PipelineError("decode_failed", f"ffmpeg выдал неверный WAV: {type(exc).__name__}") from None
        if frames == 0:
            raise PipelineError("decode_failed", "В файле нет аудиодорожки")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def probe_duration(path: Path, *, timeout: float = PROBE_TIMEOUT_SEC) -> float:
    """Длительность контейнера по ffprobe, секунды."""
    path = Path(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    code, stdout, stderr = _run(cmd, timeout=timeout, cancel=None, paths=(path,), what="ffprobe")
    text = stdout.decode("utf-8", "replace").strip()
    try:
        duration = float(text.splitlines()[0]) if code == 0 and text else float("nan")
    except ValueError:
        duration = float("nan")
    if not np.isfinite(duration) or duration < 0:
        logger.warning(
            "ffprobe не определил длительность",
            extra={
                "job": _job_name(path),
                "returncode": code,
                "stderr": _sanitize(stderr.decode("utf-8", "replace"), (path,)),
            },
        )
        raise PipelineError("decode_failed", "Не удалось определить длительность аудио")
    return duration


def is_wav16k_mono(path: Path) -> bool:
    """WAV уже в нужном формате (PCM16, моно, 16 кГц) — перекодировать не нужно."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnchannels() == 1 and wf.getframerate() == SAMPLE_RATE and wf.getsampwidth() == 2
    except (wave.Error, EOFError, OSError):
        return False


def wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != 2:
            raise ValueError("Ожидается WAV PCM 16 бит, моно, 16 кГц")
        return wf.getnframes()


def load_wav(path: Path) -> np.ndarray:
    """WAV PCM16 моно 16 кГц → float32 в [-1, 1]. Другой формат — PipelineError("decode_failed")."""
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != 2:
                raise ValueError("Ожидается WAV PCM 16 бит, моно, 16 кГц")
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, ValueError) as exc:
        raise PipelineError("decode_failed", f"Неверный WAV: {type(exc).__name__}") from None
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return samples


def rms_frames(samples: np.ndarray, frame_ms: float = 30, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """RMS по неперекрывающимся окнам frame_ms; неполное последнее окно учитывается. Пустой вход — пустой массив."""
    frame = max(1, round(sample_rate * frame_ms / 1000))
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    full = arr.size // frame
    rms = np.sqrt(np.mean(np.square(arr[: full * frame].reshape(full, frame), dtype=np.float64), axis=1))
    if arr.size % frame:
        tail = arr[full * frame :]
        rms = np.append(rms, np.sqrt(np.mean(np.square(tail, dtype=np.float64))))
    return rms.astype(np.float32)
