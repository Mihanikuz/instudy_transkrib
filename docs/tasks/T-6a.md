# T-6a · Демон: защита попыток дней от неисправностей сервера

```
Прочитай CLAUDE.md, docs/tasks/T-6.md (пункт про «сервер не должен сжигать попытки дней» и последние три критерия приёмки — они не реализованы в слитом PR #7), transkrib/daemon.py, transkrib/pipeline.py, transkrib/asr.py (ModelError), transkrib/state.py и договор instudy: https://github.com/Mihanikuz/instudy/blob/main/docs/transcription-api.md, раздел «fail» — shutdown не засчитывается в попытки, любой другой retryable-отказ засчитывается, третья неудача переводит день в failed. Ты — senior Python-разработчик.

Проблема: сейчас демон проверяет только файлы моделей (models_check). Если CUDA недоступна (на этом сервере уже был «Driver/library version mismatch»), модель не грузится внутри pipeline.run → PipelineError("model_error") → fail model_error retryable → попытка дня засчитана. За час сломанный сервер переведёт в failed весь архив.

Сделай:
1. Старт: после models_check — загрузить обе модели (diarizer.load(), recognizer.load()) до первой аренды. Исключение при загрузке → критический лог, статус error с причиной "models_load", повтор раз в MODELS_RETRY_SEC (600 с), работа не берётся. При KEEP_MODELS_LOADED=false после успешной проверки модели выгрузить (проверка = «грузятся», а не держать в памяти).
2. Посреди работы: отличать неисправность сервера от проблемы записи. pipeline.run пробрасывает ошибку загрузки модели отдельным исключением (например, ServerFault(PipelineError) с reason="model_error" и флагом server_fault=True, или ModelError как есть — выбери и опиши в PR) → демон шлёт fail shutdown (retryable, не засчитывается), ставит статус error "models_load" и больше не берёт работу, пока повторная загрузка (раз в 600 с) не пройдёт.
3. Предохранитель: счётчик работ подряд, закончившихся model_error или oom (не download_failed, не decode_failed, не no_speech). На третьей подряд — выдача на паузе: статус error "model_breaker", критический лог, повтор загрузки моделей раз в 600 с; успешная загрузка сбрасывает счётчик. Любой результат 202/409 сбрасывает счётчик.
4. state.json и cli status показывают причину (models_load / model_breaker) и время следующей проверки.
5. docs/architecture.md, «Ошибки и устойчивость» — две строки в таблицу (модели не грузятся; три сбоя модели подряд). docs/contract.md не трогать.

Критерии приёмки (pytest, фейковый instudy + фейковые модели, подменяемые часы):
- модели не загрузились на старте → next_job не вызывается ни разу, статус error/models_load, через 600 с повтор; успешная загрузка → работа берётся;
- ModelError при загрузке посреди работы → fail с reason=shutdown, retryable=true; следующий next_job — только после успешной повторной загрузки;
- три работы подряд с model_error (или oom) → четвёртая не берётся, статус error/model_breaker; успешная загрузка после повтора снимает блокировку;
- две работы с model_error, затем 202 → счётчик сброшен, третья model_error блокировку не включает;
- decode_failed и no_speech подряд не включают предохранитель;
- state.json валиден и без секретов на каждом шаге (существующий тест не ломается).
```
