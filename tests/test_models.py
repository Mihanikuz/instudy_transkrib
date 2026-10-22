from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fakes import FakeDiarizer, FakeOOM, FakeRecognizer
from transkrib import cli, models_store
from transkrib.asr import GigaAMRecognizer, ModelError, Recognizer, check_chunk, is_cuda_oom, result_text
from transkrib.diarize import Diarizer, PyannoteDiarizer, Turn, normalize_turns, read_wav_mono16k


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args, **kwargs):
        raise AssertionError("Проверка моделей не должна ходить в сеть")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)


def make_gigascribe_models(root: Path) -> Path:
    """Раскладка models/ как у gigascribe: те же пути плюс его маркеры (их импорт не переносит)."""
    g = root / "gigaam" / "v3_e2e_rnnt"
    g.mkdir(parents=True)
    (g / "v3_e2e_rnnt.ckpt").write_bytes(os.urandom(2 * 1024 * 1024))
    (g / "v3_e2e_rnnt_tokenizer.model").write_bytes(b"t" * 4096)
    (g / ".gigaam-ready.json").write_text("{}")
    p = root / "pyannote" / "community-1"
    (p / "embedding").mkdir(parents=True)
    (p / "config.yaml").write_text("pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n")
    (p / "embedding" / "pytorch_model.bin").write_bytes(os.urandom(4096))
    (p / ".pyannote-ready.json").write_text("{}")
    return root


@pytest.fixture
def ckpt_md5(monkeypatch: pytest.MonkeyPatch):
    """Подменить эталонную md5 чекпоинта на md5 нашего тестового файла."""

    def apply(models_dir: Path) -> None:
        ckpt = models_dir / "gigaam" / "v3_e2e_rnnt" / "v3_e2e_rnnt.ckpt"
        monkeypatch.setattr(models_store, "GIGAAM_CKPT_MD5", models_store.md5_file(ckpt))

    return apply


# ---------------------------------------------------------------------------- лёгкий импорт


def test_backends_do_not_import_gpu_stack() -> None:
    code = (
        "import sys, transkrib.asr, transkrib.diarize, transkrib.models_store, transkrib.cli, tests.fakes;"
        "heavy = [m for m in ('torch', 'gigaam', 'pyannote', 'huggingface_hub') if m in sys.modules];"
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stdout + res.stderr


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeRecognizer(), Recognizer)
    assert isinstance(FakeDiarizer(), Diarizer)
    assert isinstance(GigaAMRecognizer(Path("/nonexistent")), Recognizer)
    assert isinstance(PyannoteDiarizer(Path("/nonexistent")), Diarizer)


# ---------------------------------------------------------------------------- models_store


def test_check_empty_dir_lists_what_is_missing(tmp_path: Path) -> None:
    problems = models_store.check(tmp_path)
    text = "\n".join(map(str, problems))
    assert "v3_e2e_rnnt.ckpt" in text
    assert "v3_e2e_rnnt_tokenizer.model" in text
    assert "config.yaml" in text


def test_files_without_marker_are_not_ready(tmp_path: Path) -> None:
    src = make_gigascribe_models(tmp_path / "src")
    problems = models_store.check(src)
    assert {p.model for p in problems} == {"gigaam", "pyannote"}
    assert all("маркер" in p.message for p in problems)


def test_import_from_gigascribe(tmp_path: Path, ckpt_md5) -> None:
    src = make_gigascribe_models(tmp_path / "src")
    ckpt_md5(src)
    dst = tmp_path / "models"
    models_store.import_models(src, dst, log=lambda _: None)
    assert models_store.check(dst) == []
    assert models_store.check(dst, deep=True) == []
    assert not (dst / "gigaam" / "v3_e2e_rnnt" / ".gigaam-ready.json").exists()
    marker = json.loads((dst / "pyannote" / "community-1" / ".ready.json").read_text())
    assert set(marker["files"]) == {"config.yaml", "embedding/pytorch_model.bin"}
    g_marker = json.loads((dst / "gigaam" / "v3_e2e_rnnt" / ".ready.json").read_text())
    assert g_marker["gigaam_commit"] == models_store.GIGAAM_COMMIT


def test_import_refuses_wrong_checkpoint(tmp_path: Path) -> None:
    src = make_gigascribe_models(tmp_path / "src")
    with pytest.raises(RuntimeError, match="Контрольная сумма"):
        models_store.import_models(src, tmp_path / "models", log=lambda _: None)


def test_changed_file_is_detected(tmp_path: Path, ckpt_md5) -> None:
    src = make_gigascribe_models(tmp_path / "src")
    ckpt_md5(src)
    dst = tmp_path / "models"
    models_store.import_models(src, dst, log=lambda _: None)
    weights = dst / "pyannote" / "community-1" / "embedding" / "pytorch_model.bin"
    weights.write_bytes(os.urandom(4096))  # тот же размер, другое содержимое
    assert models_store.check(dst) == []  # быстрая проверка — только размеры
    assert any("хэш" in p.message for p in models_store.check(dst, deep=True))
    weights.write_bytes(b"x")
    assert any("размер" in p.message for p in models_store.check(dst))
    weights.unlink()
    assert any("нет файла" in p.message for p in models_store.check(dst))


def test_other_gigaam_commit_requires_recheck(tmp_path: Path, ckpt_md5, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_gigascribe_models(tmp_path / "src")
    ckpt_md5(src)
    dst = tmp_path / "models"
    models_store.import_models(src, dst, log=lambda _: None)
    monkeypatch.setattr(models_store, "GIGAAM_COMMIT", "0" * 40)
    assert any("коммитом" in p.message for p in models_store.check(dst))


def test_download_requires_hf_token_for_pyannote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt_md5) -> None:
    # GigaAM уже на месте — сеть не нужна; без HF_TOKEN загрузка pyannote должна отказать до обращения в сеть.
    src = make_gigascribe_models(tmp_path / "models")
    ckpt_md5(src)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        models_store.download_models(src, None, log=lambda _: None)


# ---------------------------------------------------------------------------- CLI


def test_cli_models_check_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt_md5, capsys) -> None:
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    assert cli.main(["models-check"]) == 1
    assert "не готовы" in capsys.readouterr().out
    src = make_gigascribe_models(tmp_path / "src")
    ckpt_md5(src)
    assert cli.main(["models-import", str(src)]) == 0
    assert cli.main(["models-check", "--deep"]) == 0
    assert "на месте" in capsys.readouterr().out


def test_cli_models_import_missing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    assert cli.main(["models-import", str(tmp_path / "nope")]) == 1
    assert "Ошибка" in capsys.readouterr().err


# ---------------------------------------------------------------------------- бэкенды без GPU


def test_backends_refuse_to_load_without_models(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="нет файла"):
        GigaAMRecognizer(tmp_path).load()
    with pytest.raises(ModelError, match="нет файла"):
        PyannoteDiarizer(tmp_path).load()


def test_backends_refuse_work_before_load(tmp_path: Path) -> None:
    with pytest.raises(ModelError):
        GigaAMRecognizer(tmp_path).transcribe(np.zeros(16000, dtype=np.float32))
    with pytest.raises(ModelError):
        PyannoteDiarizer(tmp_path).diarize(tmp_path / "a.wav")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Привет. ", "Привет."),
        ({"text": "Текст"}, "Текст"),
        ({"transcript": "Т2"}, "Т2"),
        (("кортеж", None), "кортеж"),
        (SimpleNamespace(text="объект"), "объект"),
        (SimpleNamespace(transcript="старый"), "старый"),
        ([SimpleNamespace(text="а"), "", {"text": "б"}], "а б"),
        (None, ""),
    ],
)
def test_result_text(value, expected: str) -> None:
    assert result_text(value) == expected


def test_is_cuda_oom() -> None:
    assert is_cuda_oom(FakeOOM())
    assert is_cuda_oom(type("OutOfMemoryError", (Exception,), {})("x"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))


def test_check_chunk() -> None:
    assert check_chunk(np.zeros(100, dtype=np.float64)).dtype == np.float32
    with pytest.raises(ValueError):
        check_chunk(np.zeros((2, 100)))
    with pytest.raises(ValueError):
        check_chunk(np.zeros(0))
    with pytest.raises(ValueError, match="25"):
        check_chunk(np.zeros(26 * 16000))


# ---------------------------------------------------------------------------- нормализация pyannote


class FakeAnnotation:
    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False):
        for start, end, label in self.tracks:
            yield SimpleNamespace(start=start, end=end), "A", label


def test_normalize_prefers_exclusive() -> None:
    out = SimpleNamespace(
        exclusive_speaker_diarization=FakeAnnotation([(5.0, 7.0, "SPEAKER_01"), (0.5, 4.0, "SPEAKER_00")]),
        speaker_diarization=FakeAnnotation([(0.0, 100.0, "X")]),
    )
    assert normalize_turns(out) == [Turn(0.5, 4.0, "SPEAKER_00"), Turn(5.0, 7.0, "SPEAKER_01")]


def test_normalize_falls_back_to_speaker_diarization() -> None:
    out = SimpleNamespace(speaker_diarization=FakeAnnotation([(1.0, 2.0, "S")]))
    assert normalize_turns(out) == [Turn(1.0, 2.0, "S")]


def test_normalize_accepts_bare_annotation_and_cleans() -> None:
    ann = FakeAnnotation([(-0.2, 1.0, "S"), (3.0, 3.0, "S"), (4.0, 2.0, "S")])
    assert normalize_turns(ann) == [Turn(0.0, 1.0, "S")]


def test_normalize_empty_and_unknown() -> None:
    assert normalize_turns(FakeAnnotation([])) == []
    with pytest.raises(ModelError):
        normalize_turns(object())


# ---------------------------------------------------------------------------- WAV


def write_wav(path: Path, samples: np.ndarray, *, rate: int = 16000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def test_read_wav_mono16k(tmp_path: Path) -> None:
    tone = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32) * 0.5
    data = read_wav_mono16k(write_wav(tmp_path / "a.wav", tone))
    assert data.dtype == np.float32 and data.size == 16000
    assert np.max(np.abs(data - tone)) < 1e-3
    with pytest.raises(ValueError):
        read_wav_mono16k(write_wav(tmp_path / "b.wav", tone, rate=44100))


def test_fake_diarizer_covers_whole_file(tmp_path: Path) -> None:
    d = FakeDiarizer()
    d.load()
    assert d.diarize(write_wav(tmp_path / "a.wav", np.zeros(32000))) == [Turn(0.0, 2.0, "SPEAKER_00")]


def test_fake_recognizer_failures() -> None:
    r = FakeRecognizer(fail_on=[1], oom_on=[2])
    r.load()
    chunk = np.zeros(16000, dtype=np.float32)
    assert r.transcribe(chunk) == "фраза 1 (1.0 с)."
    with pytest.raises(RuntimeError):
        r.transcribe(chunk)
    with pytest.raises(FakeOOM):
        r.transcribe(chunk)
    assert r.transcribe(chunk) == "фраза 4 (1.0 с)."


# ---------------------------------------------------------------------------- только на сервере


@pytest.mark.gpu
def test_gpu_smoke() -> None:
    wav = os.environ.get("TRANSKRIB_SMOKE_WAV")
    if not wav:
        pytest.skip("задайте TRANSKRIB_SMOKE_WAV — путь к WAV 16 кГц моно")
    from scripts.gpu_smoke import main

    assert main([wav]) == 0
