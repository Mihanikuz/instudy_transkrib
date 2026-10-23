# Образ сервера транскрибации: python 3.12 + ffmpeg + GPU-стек (torch 2.10 CUDA 12.8, pyannote.audio 4.0, GigaAM).
# Моделей внутри нет: /models монтируется только для чтения (docs/architecture.md, «Развёртывание»).
# База закреплена по digest (индекс мультиархитектурного образа 3.12-slim-bookworm на 23.10.2026); обновлять вручную:
#   docker buildx imagetools inspect python:3.12-slim-bookworm
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

# ---------------------------------------------------------------- сборка окружения
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

# git — только для установки GigaAM из закреплённого коммита; в итоговый образ не попадает.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY requirements.txt requirements-gpu.txt requirements-gigaam.txt ./
# 1) демон без GPU-стека; 2) GPU-стек и зависимости GigaAM — всё с хэшами; 3) сам GigaAM без зависимостей
# (git-ссылку нельзя проверить хэшем — закреплена полным хэшем коммита). --no-build-isolation: сборка gigaam
# берёт setuptools из файла с хэшами, а не качает свежий.
RUN pip install --require-hashes -r requirements.txt \
 && pip install --require-hashes -r requirements-gpu.txt \
 && pip install --no-deps --no-build-isolation -r requirements-gigaam.txt \
 && pip check

# ---------------------------------------------------------------- итоговый образ
FROM ${PYTHON_IMAGE}

# ffmpeg — подготовка аудио; tini — PID 1: пересылает SIGTERM демону и собирает зомби (ffmpeg в своей группе).
# Версии пакетов не закрепляются: база закреплена по digest, а из Debian stable нужны обновления безопасности.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 transkrib \
 && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --no-create-home \
      --shell /usr/sbin/nologin transkrib

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY transkrib ./transkrib
COPY scripts ./scripts

# Офлайн после разовой загрузки моделей (CLAUDE.md). ФС контейнера только для чтения, писать можно в /data и /tmp:
# кэши matplotlib (тянет pyannote), Hugging Face и torch — в /tmp.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYANNOTE_METRICS_ENABLED=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    HOME=/tmp \
    MODELS_DIR=/models \
    DATA_DIR=/data

USER 10001:10001

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "transkrib"]
