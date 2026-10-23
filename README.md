# instudy_transkrib

Сервер автоматической транскрибации лекций для «Моего расписания» ([instudy](https://github.com/Mihanikuz/instudy)).

- Архитектура — [docs/architecture.md](docs/architecture.md)
- Контракт с instudy — [docs/contract.md](docs/contract.md); договор на стороне instudy — [`instudy/docs/transcription-api.md`](https://github.com/Mihanikuz/instudy/blob/main/docs/transcription-api.md)
- План и задачи — [docs/plan.md](docs/plan.md)
- Правила для агентов — [CLAUDE.md](CLAUDE.md)

## Что это

Демон на домашнем GPU-сервере. Он сам забирает из instudy дни с аудио без транскрипта, от самых старых к новым. Запись делится на спикеров моделью pyannote community-1, речь распознаётся GigaAM v3 e2e RNNT (с пунктуацией). В instudy уходят два файла:

- `.vtt` — субтитры с таймкодами и голосами `Спикер N`, для плеера;
- `.txt` — читаемый текст `[ЧЧ:ММ:СС] Спикер N: …` для скачивания.

Входящих портов нет: сервер работает за NAT и сам ходит в `api.<домен>` по сервисному токену. После разовой загрузки моделей он работает офлайн и обращается только к instudy. Работа одна за раз, видеокарта целиком его.

## Требования

- Linux x86_64 (проверяется на Ubuntu 26.04), NVIDIA GPU с поддержкой CUDA 12.8 (RTX 5060 Ti 16 ГБ, sm_120) и драйвер ≥ 570 (на сервере — 595.99).
- Docker ≥ 29, Docker Compose ≥ 2.40, NVIDIA Container Toolkit ≥ 1.19 (runtime `nvidia`).
- Диск: образ около 10 ГБ, модели около 1,5 ГБ, плюс `MIN_FREE_GB` (10 ГБ) свободного места для рабочих файлов.
- Исходящий HTTPS до `api.<домен>` instudy. Для первой установки также нужен доступ к github.com, pypi.org, deb.debian.org и Docker Hub; для загрузки моделей — к huggingface.co и CDN Сбера.

Проверка, что Docker видит видеокарту:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Должна напечататься таблица с видеокартой и версией драйвера. Ошибка `could not select device driver` означает, что не настроен NVIDIA Container Toolkit (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

## Первая установка

Все команды выполняются на сервере от своего пользователя с `sudo`.

**1. Код и каталоги.**

```bash
sudo mkdir -p /srv/transkrib && sudo chown "$USER": /srv/transkrib
git clone https://github.com/Mihanikuz/instudy_transkrib.git /srv/transkrib
cd /srv/transkrib
mkdir -p data models
```

**2. Права на `data/` и `models/`.** В контейнере работает пользователь `uid 10001` (не root). Все файлы в `data/` он создаёт от себя: `state.json`, рабочие файлы и копии результатов.

```bash
sudo groupadd --gid 10001 transkrib            # группа с тем же gid, что в контейнере
sudo usermod -aG transkrib "$USER"             # чтобы читать результаты без sudo (перелогиньтесь)
sudo chown -R 10001:10001 data/ models/
sudo chmod 750 data/                           # транскрипты — персональные данные лекций: другим пользователям не видно
```

Результаты потом читаются от своего пользователя через группу `transkrib`, например `ls data/results/`. Можно и через sudo: `sudo cat data/results/<id>/transcript.txt`.

**3. Настройки.**

```bash
cp .env.example .env
chmod 600 .env
nano .env        # INSTUDY_API_URL=https://api.<домен>, SERVICE_TOKEN — см. «Получение токена»
```

Описание переменных — в [docs/architecture.md](docs/architecture.md), раздел «Конфигурация». `HF_TOKEN` нужен только для загрузки моделей с Hugging Face (шаг 5, вариант Б), и его видит только сервис `models`.

**4. Сборка образа.**

```bash
docker compose build
```

**5. Модели.** Модели монтируются в контейнер только для чтения, а пишет в `models/` только отдельный сервис `models`.

Вариант А — перенести модели gigascribe, которые уже лежат на сервере:

```bash
docker compose run --rm -v /opt/gigascribe/models:/import:ro models models-import /import
```

Вариант Б — скачать. Нужен `HF_TOKEN` в `.env` от аккаунта, принявшего условия [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1):

```bash
docker compose run --rm models models-download
```

Проверка: `docker compose run --rm models models-check` должен ответить «Модели в /models на месте…». После загрузки `HF_TOKEN` из `.env` можно удалить.

**6. Проверка связи, моделей и GPU.**

```bash
docker compose run --rm transkrib python -m transkrib.cli check
```

Ожидается три строки: `instudy: ok, …`, `Модели (/models): ok`, `GPU: <имя карты>, <драйвер>, …`.

**7. Прогон на тестовой записи.** Демон и instudy здесь не участвуют. Положите файл в `data/` и запустите:

```bash
sudo cp ~/lecture.m4a data/ && sudo chown 10001:10001 data/lecture.m4a
docker compose run --rm transkrib python -m transkrib.cli run-file /data/lecture.m4a --out /data/run-file/lecture
```

В stderr печатаются этапы, в stdout — сводка JSON: время по этапам, `realtimeFactor`, пиковая VRAM. Файлы `.vtt`, `.txt` и `meta.json` появятся в `data/run-file/lecture/`. Ориентир: 3 часа лекции обрабатываются примерно за 6 минут.

**8. Запуск.**

```bash
docker compose up -d
docker compose exec transkrib python -m transkrib.cli status
```

Через минуту `docker compose ps` должен показать `healthy`.

## Получение токена

Сервисный токен выдаёт админ instudy на сервере instudy:

```bash
python -m app.cli service-add home-gpu
```

Токен показывается один раз. Его нужно вставить в `.env` этого сервера (`SERVICE_TOKEN=…`, файл с правами `600`) и перезапустить демон: `docker compose up -d`. Токен не пересылайте в чатах и не коммитьте. Демон не пишет его ни в логи, ни в `state.json`.

## Эксплуатация

| Что | Команда |
| --- | --- |
| Что делает демон (работа, этап, счётчики, последняя ошибка, причина паузы) | `docker compose exec transkrib python -m transkrib.cli status` |
| Не брать новые работы (текущая доработает) | `docker compose exec transkrib python -m transkrib.cli pause` |
| Снова брать работы | `docker compose exec transkrib python -m transkrib.cli resume` |
| Проверить instudy, модели и GPU | `docker compose exec transkrib python -m transkrib.cli check` |
| Логи (JSON, без токенов и текстов транскриптов) | `docker compose logs -f --tail 100 transkrib` |
| Состояние контейнера и healthcheck | `docker compose ps` |

- Пауза — это файл `data/PAUSE`. Пока он есть, демон работу не берёт, даже после перезапуска.
- Копии отправленных результатов лежат в `data/results/<id работы>/` (`transcript.vtt`, `transcript.txt`, `meta.json`) `RESULTS_KEEP_DAYS` дней (по умолчанию 14). Они нужны для разбора жалоб на качество.
- `data/work/` — временные файлы текущей работы. Они удаляются после каждой работы и при старте.
- Логи Docker ротируются: 10 МБ × 5 файлов.

## Обновление

```bash
cd /srv/transkrib
deploy/update.sh
```

Скрипт делает `git pull --ff-only`, `docker compose build` и `docker compose up -d`, ждёт healthcheck и печатает `cli status`. Если демон в этот момент обрабатывал работу, он получает SIGTERM, сообщает instudy `fail shutdown` (попытка дня не засчитывается) и выходит за 30 с или быстрее. Работа вернётся в очередь и будет взята снова.

## Отзыв токена и остановка

- **Остановить:** `docker compose stop`. Текущая работа вернётся в очередь (`fail shutdown`). Если демон должен перестать брать работы, но контейнер нужно оставить, используйте `cli pause`.
- **Отозвать токен** (сервер скомпрометирован, токен утёк): админ instudy отзывает его на стороне instudy. Демон получит `401`, остановит выдачу и будет перепроверять токен раз в 10 минут. Чтобы выдать новый, повторите «Получение токена».
- **Убрать совсем:** `docker compose down`, затем при необходимости удалить `/srv/transkrib/data` (там копии транскриптов) и `/srv/transkrib/models`.

## Чужие сервисы на GPU

Видеокарта должна принадлежать только этому демону. Другие процессы на GPU делят с ним видеопамять: модели могут не загрузиться (`models_load`), а работа может упасть по `oom`.

На этом сервере уже есть служба **Ollama**: она слушает `*:11434` без пароля и держит модели в VRAM. Её стоит выключить:

```bash
sudo systemctl disable --now ollama
```

Если она нужна, привяжите её к localhost и не давайте держать модели в памяти:

```bash
sudo systemctl edit ollama      # добавить:
# [Service]
# Environment="OLLAMA_HOST=127.0.0.1:11434"
# Environment="OLLAMA_KEEP_ALIVE=0"
sudo systemctl restart ollama
```

Кто занимает видеокарту, видно в `nvidia-smi` (раздел Processes).

## Если что-то не так

| Признак | Что делать |
| --- | --- |
| `cli status`: `error (models_load …)`, в логах `CUDA недоступна`, а `nvidia-smi` на хосте пишет `Failed to initialize NVML: Driver/library version mismatch` | Драйвер обновился, а модуль ядра остался старый. Перезагрузите сервер (`sudo reboot`). Демон сам проверяет модели раз в 10 минут и продолжит после перезагрузки; попытки дней при этом не тратятся |
| `error (models …)` | Нет файлов моделей или они неполные: `docker compose run --rm models models-check`, затем повторите шаг 5 установки |
| `error (model_breaker …)` | Три работы подряд закончились сбоем модели или нехваткой памяти. Посмотрите `docker compose logs --tail 200 transkrib` и `nvidia-smi` (чужие процессы на GPU — см. «Чужие сервисы на GPU»). Демон проверяет модели раз в 10 минут и снимает блокировку сам |
| В логах `oom` / `CUDA out of memory` | Освободите видеопамять от чужих процессов. Если OOM повторяется на длинных записях, приложите логи к задаче T-8 (замеры) |
| `error (unauthorized …)`, в логах `401` | Токен отозван или вставлен с ошибкой. Получите новый («Получение токена»), исправьте `.env`, выполните `docker compose up -d` |
| `paused (low_disk)` | Меньше `MIN_FREE_GB` свободного места на диске с `data/`. Освободите место (например, старые `data/results/`, `docker image prune`) |
| `paused (pause)` | Включена пауза: `cli resume` |
| `cli status`: «instudy: выдача работ на паузе (autotx-pause…)» | Выдачу выключил админ instudy. Это не ошибка: демон продолжает опрашивать очередь и возьмёт работу, как только пауза будет снята |
| В логах `instudy ответил 503 на GET /jobs/next` | В instudy не настроен сервисный API (не задан `SERVICE_LINK_SECRET`). Это чинится на стороне instudy, демон повторяет сам |
| Каждая работа уходит в `invalid_job` «Ссылка на аудио ведёт не на сервер instudy» | Хост `INSTUDY_API_URL` не совпадает с базовым адресом instudy (`API_BASE_URL` или `https://api.<DOMAIN>`). Исправьте `.env` или настройку instudy |
| `fail download_failed` «SHA-256 аудио не совпал» | Аудио дня пережали (`audio-reencode`) во время скачивания. Работа вернётся в очередь с новым хэшем |
| Контейнер `unhealthy` | Демон не обновлял `state.json` больше 3 минут, процесс завис или упал. Смотрите `docker compose logs --tail 200 transkrib`, перезапуск — `docker compose restart transkrib` |
| В логах предупреждение torchcodec про `libnppicc.so.12` | Безвредно (аудио передаётся в pyannote массивом, декодер torchcodec не используется); в `diarize.load()` оно заглушено |
