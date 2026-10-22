"""Демон в отдельном процессе для тестов сигналов: фейковый instudy, фейковые модели, настоящие сигналы.

python -m tests.daemon_harness <data_dir> <режим>
  asr   — медленное распознавание (много кусков), печатает READY перед первым куском;
  stuck — «зависшая» диаризация (непрерываемый вызов), печатает READY в начале диаризации.
При выходе печатает строку RESULT {json}: отказы, отправленные в instudy, и содержимое data/work.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from tests.audio_gen import tone, write_wav
from tests.fake_instudy import BASE_URL, TOKEN, FakeInstudy
from tests.fakes import FakeDiarizer, FakeRecognizer
from transkrib import daemon as daemon_mod
from transkrib.client import InstudyClient
from transkrib.config import load_settings
from transkrib.diarize import Turn


class SlowRecognizer(FakeRecognizer):
    def transcribe_batch(self, chunks):
        if self.calls == 0:
            print("READY", flush=True)
        time.sleep(0.2)
        return super().transcribe_batch(chunks)


class StuckDiarizer(FakeDiarizer):
    def diarize(self, wav_path):
        print("READY", flush=True)
        time.sleep(600)  # как долгий вызов pyannote: флаг отмены не проверяет
        return super().diarize(wav_path)


def main() -> int:
    data_dir, mode = Path(sys.argv[1]), sys.argv[2]
    fake = FakeInstudy()
    job = fake.add_job(1)
    wav = write_wav(data_dir.parent / "src.wav", tone(120.0))
    job.audio, job.duration_sec = wav.read_bytes(), 120
    settings = load_settings(data_dir=data_dir, models_dir=data_dir.parent / "models", poll_interval_sec=1)
    client = InstudyClient(
        BASE_URL, TOKEN, max_audio_bytes=10**9, max_audio_sec=10**6, transport=fake.transport, sleep=lambda s: None
    )
    turns = [Turn(2.0 * i, 2.0 * i + 1.0, "SPEAKER_00") for i in range(60)]
    if mode == "stuck":
        daemon_mod.SHUTDOWN_GRACE_SEC = 1.0
        diarizer, recognizer = StuckDiarizer(turns), FakeRecognizer()
    else:
        diarizer, recognizer = FakeDiarizer(turns), SlowRecognizer()
    daemon = daemon_mod.Daemon(
        settings,
        client,
        diarizer,
        recognizer,
        models_check=lambda d: [],
        disk_usage=lambda p: SimpleNamespace(free=10**12),
    )
    daemon.install_signal_handlers()
    code = daemon.run()
    work = sorted(p.name for p in (data_dir / "work").iterdir())
    print("RESULT " + json.dumps({"fails": job.fails, "work": work, "abandoned": daemon.abandoned}), flush=True)
    if daemon.abandoned:
        os._exit(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
