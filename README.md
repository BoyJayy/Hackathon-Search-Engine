# Search Hackathon Solution

Репозиторий содержит решение для хакатона по индексации и поиску по сообщениям.

В проекте есть два сервиса:
- `index` — принимает чат и сообщения, строит чанки и sparse embeddings;
- `search` — принимает вопрос, делает retrieval и возвращает `message_ids`.

Контракты API менять нельзя. Внутреннюю логику можно менять.

## Карта документации

- [api.md](api.md) — контракт всех endpoint'ов и внешних API.
- [docs/index_service.md](docs/index_service.md) — устройство текущего `index` и его модули.
- [docs/process.md](docs/process.md) — полный процесс от входного JSON до финального поиска.
- [hackathon_roadmap.md](hackathon_roadmap.md) — roadmap по доработке решения.

## Структура репозитория

- `index/main.py` — FastAPI-приложение `index`.
- `index/config.py` — env, константы и логгер `index`.
- `index/schemas.py` — Pydantic-схемы `index`.
- `index/chunking.py` — нормализация, фильтрация и chunking.
- `index/sparse.py` — локальная sparse-модель и построение sparse vectors.
- `search/main.py` — текущий FastAPI-сервис поиска.
- `data/Go Nova.json` — анонимизированный пример реального чата.
- `docker-compose.yml` — локальный запуск `qdrant`, `index` и `search`.

## Что уже сделано

Текущий `index` уже не является заглушкой:
- нормализует `text`, `parts`, `file_snippets`, `mentions`;
- фильтрует скрытые и шумные сообщения;
- делает message-based chunking с overlap-контекстом;
- сжимает длинные технические логи;
- режет очень длинные обычные сообщения на большие смысловые части;
- формирует разные `page_content`, `dense_content`, `sparse_content`;
- сохраняет корректные `message_ids`.

## Что менять можно

Можно менять:
- chunking и нормализацию сообщений;
- формирование `dense_content`;
- формирование `sparse_content`;
- sparse-модель внутри `index`;
- retrieval, fusion и rerank в `search`;
- любые эвристики и фильтры.

Нельзя менять:
- `GET /health`
- `POST /index`
- `POST /sparse_embedding`
- `POST /search`

Если сломать request/response этих endpoint'ов, решение перестанет проходить проверку.

## Локальный запуск

Перед `docker compose up` надо задать учётные данные для внешнего dense/rerank API:

```bash
export OPEN_API_LOGIN=...
export OPEN_API_PASSWORD=...
```

Запуск:

```bash
docker compose up --build
```

Поднимутся:
- `qdrant` на `localhost:6333`
- `index` на `localhost:8001`
- `search` на `localhost:8002`

Важно: текущий репозиторий поднимает сервисы, но не содержит отдельного ingestion-orchestrator, который прогоняет `/index` -> dense/sparse embeddings -> upsert в Qdrant. Этот pipeline описан в [docs/process.md](docs/process.md).

## Быстрый тест `index`

Сборка и запуск только `index`:

```bash
cd index
make build
make run
```

Проверка health:

```bash
curl http://localhost:8000/health
```

Пример теста `POST /index`:

```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/index_payload.json
```

Что должно быть в ответе:
- реальные chunk'и, а не `"string"`;
- блоки `CONTEXT` и `MESSAGES` в `page_content`;
- разные `dense_content` и `sparse_content`;
- корректные `message_ids`;
- сжатие логов через `... [N lines omitted] ...`;
- split длинных сообщений через `part=1/2`, `part=2/2`.

## Внешний dense/rerank API

Хакатонный внешний API:
- `http://83.166.249.64:18001/embeddings`
- `http://83.166.249.64:18001/score`

Проверка моделей:

```bash
curl -u "$OPEN_API_LOGIN:$OPEN_API_PASSWORD" \
  "http://83.166.249.64:18001/embeddings/models"
```

```bash
curl -u "$OPEN_API_LOGIN:$OPEN_API_PASSWORD" \
  "http://83.166.249.64:18001/score/models"
```

## Важное архитектурное замечание

Сейчас `index` не строит dense vectors сам. Он только готовит `dense_content`.

Dense embeddings сейчас считаются во внешнем API. Для полной индексации нужен отдельный ingestion pipeline:

```text
POST /index
-> dense_content / sparse_content
-> dense embeddings API
-> POST /sparse_embedding
-> upsert в Qdrant
```

Этот процесс расписан подробно в [docs/process.md](docs/process.md).
