#!/usr/bin/env bash
# Обновление сервера транскрибации: свежий код → сборка → перезапуск → ожидание healthcheck → статус.
# Текущая работа при перезапуске получает fail shutdown (попытка дня не засчитывается) и вернётся в очередь.
# Запуск: /srv/transkrib/deploy/update.sh
set -euo pipefail

cd "$(dirname "$0")/.."

HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-600}"

echo "==> git pull"
git pull --ff-only

echo "==> docker compose build"
docker compose build

echo "==> docker compose up -d"
docker compose up -d

echo "==> ожидание healthcheck (до ${HEALTH_TIMEOUT_SEC} с)"
container="$(docker compose ps -q transkrib)"
if [ -z "$container" ]; then
  echo "Контейнер transkrib не запущен" >&2
  exit 1
fi
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))
status="starting"
while [ "$(date +%s)" -lt "$deadline" ]; do
  status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container")"
  case "$status" in
    healthy) break ;;
    unhealthy) break ;;
  esac
  sleep 5
done
echo "healthcheck: $status"

echo "==> cli status"
docker compose exec -T transkrib python -m transkrib.cli status || true

if [ "$status" != "healthy" ]; then
  echo "Контейнер не стал healthy — смотрите: docker compose logs --tail 100 transkrib" >&2
  exit 1
fi
