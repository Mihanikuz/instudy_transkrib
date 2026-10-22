# Заглушка: настоящий образ (python:3.12-slim-bookworm по digest, ffmpeg, torch cu128, pyannote, GigaAM,
# пользователь 10001, tini, офлайн-переменные HF) собирает T-7 — docs/tasks/T-7.md.
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
COPY transkrib ./transkrib
USER 10001
CMD ["python", "-m", "transkrib"]
