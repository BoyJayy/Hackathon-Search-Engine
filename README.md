# Search Hackathon Solution

Репозиторий содержит решение для хакатона по индексации и поиску по сообщениям.

В проекте есть два сервиса:
- `index` — принимает чат и сообщения, строит чанки и sparse embeddings;
- `search` — принимает вопрос, делает retrieval и возвращает `message_ids`.

Контракты API менять нельзя. Внутреннюю логику можно менять.

## Карта документации

- [api.md](api.md) — контракт всех endpoint'ов и внешних API.
- [docs/index_service.md](docs/index_service.md) — устройство текущего `index` и его модули.
- [docs/search_service.md](docs/search_service.md) — устройство текущего `search` и его retrieval pipeline.
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
- преобразует `member_event` системных сообщений в searchable text;
- размечает quoted-части сообщений отдельным блоком `Quoted message:`, чтобы ответ и цитата не смешивались в один сплошной текст;
- фильтрует скрытые и шумные сообщения;
- делает message-based chunking с thread-aware overlap-контекстом;
- не протягивает overlap через большие временные разрывы;
- сбрасывает overlap для явных новых top-level тем, чтобы новый вопрос не наследовал старое обсуждение только из-за близкого времени;
- сжимает длинные технические логи;
- режет очень длинные обычные сообщения на большие смысловые части;
- формирует разные `page_content`, `dense_content`, `sparse_content`;
- пишет расширенную диагностику чанков в логи;
- сохраняет корректные `message_ids`.

Текущий `search` уже использует enriched question заметно сильнее baseline:
- берёт `search_text` как primary query;
- использует несколько dense queries из `text`, `search_text`, `variants`, `hyde`;
- использует несколько sparse queries из `keywords`, `entities`, `date_mentions`, `asker`;
- объединяет retrieval через `RRF`;
- делает мягкий local rescoring по exact signals, `entities`, `participants` / `mentions` и `date_range`;
- сильнее доверяет совпадениям в `MESSAGES`, а не случайным совпадениям в `CONTEXT`, чтобы соседние чанки не обгоняли реальный ответ;
- при совпадениях в quoted-блоке доверяет своей реплике сильнее, чем процитированному тексту;
- делает ограниченный rerank top-кандидатов;
- отправляет во внешний reranker текст кандидата в формате `MESSAGES -> CONTEXT`, чтобы точный ответ внутри чанка не терялся за overlap-контекстом;
- умеет мягко переупорядочивать `message_ids` внутри найденного chunk;
- собирает финальную выдачу уже на уровне отдельных `message_id`, а не только на уровне chunk order;
- читает retrieval/rerank параметры из env, чтобы быстро тюнить качество без правки кода;
- смешивает внешний rerank score с исходным retrieval-порядком через `RERANK_ALPHA`, чтобы reranker не перетирал хороший prefetch слишком агрессивно;
- кэширует dense embeddings и reranker scores на время жизни контейнера, чтобы снижать риск `429` на повторяющихся вопросах;
- режет слишком длинные тексты перед rerank;
- при `429` и других сбоях внешнего dense API использует sparse-only fallback вместо падения `500`;
- при `429` и других сбоях внешнего reranker использует fallback на retrieval order вместо падения `500`;
- дедуплицирует `message_ids` и ограничивает выдачу top-50.

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

Для локальной разработки ingestion pipeline уже есть в виде eval-скрипта: [eval/ingest.py](/Users/boyjayy/Documents/Search/eval/ingest.py). Он прогоняет `/index` -> dense embeddings -> `/sparse_embedding` -> upsert в Qdrant. Полный процесс расписан в [docs/process.md](docs/process.md).

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

Для тюнинга chunking есть отдельный локальный инструмент:

```bash
python3 scripts/chunking_diagnostic.py
python3 scripts/chunking_diagnostic.py data/Go\ Nova.json
```

Он показывает:
- распределение размеров чанков;
- messages-per-chunk;
- coverage searchable сообщений;
- `dup_ratio`;
- короткий preview каждого chunk.

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

## Локальный ingestion и eval

После `docker compose up --build` можно локально наполнить Qdrant и проверить поиск:

```bash
python3 eval/ingest.py
```

Что делает скрипт:
- вызывает `POST /index`;
- считает dense embeddings через внешний API;
- считает sparse vectors через `POST /sparse_embedding`;
- удаляет старые точки этого же чата;
- делает upsert в Qdrant со стабильными id.

Для новых synthetic eval-наборов тоже поддерживается локальный ingest:

```bash
DATA_PATH=data/dataset_ts.jsonl RESET_COLLECTION=1 python3 eval/ingest.py
```

В этом режиме `eval/ingest.py`:
- читает JSONL с полями `question` / `answer`;
- строит synthetic corpus прямо из `answer.text`;
- сохраняет `answer.message_ids` как реальные `message_ids` в Qdrant;
- пересоздаёт коллекцию, чтобы старый локальный индекс не мешал новым eval-наборам.

После этого можно гонять поиск или локальный eval harness:

```bash
python3 eval/run.py --dataset /path/to/questions.jsonl --k 50 --verbose
```

## Первая отправка

По ТЗ образы нужно отправлять именно как Docker-образы под `linux/amd64`.

### Настройка Docker

Registry хакатона работает по адресу:
- `83.166.249.64:5000`

Так как он работает без TLS, его нужно добавить в `insecure-registries`.

Для Docker Desktop (`macOS` / `Windows`):
- открыть `Docker Desktop -> Settings -> Docker Engine`
- добавить в JSON:

```json
{
  "insecure-registries": ["83.166.249.64:5000"]
}
```

Если в конфиге уже есть другие поля, `insecure-registries` нужно добавить на верхний уровень существующего JSON, а не вставлять внутрь `builder`.

После этого:
- нажать `Apply & Restart`

Логин в registry:

```bash
docker login 83.166.249.64:5000 -u <login> -p <password>
```

Сборка и пуш `index`:

```bash
cd index
export TEAM_ID=...
export LOGIN=...
export PASSWORD=...
make push
```

Сборка и пуш `search`:

```bash
cd search
export TEAM_ID=...
export LOGIN=...
export PASSWORD=...
make push
```

Текущие `Makefile` уже собирают образы с `--platform linux/amd64`, как требует ТЗ.

### Готово для команды `31023`

Если используете выданные значения:
- `team_id = 31023`
- `VK login = 3e5bab13da3a6503`
- `VK password = 92117d7d10c11049cf178236144bddf2`

То полный путь такой:

```bash
cd /Users/boyjayy/Documents/Search

export OPEN_API_LOGIN='3e5bab13da3a6503'
export OPEN_API_PASSWORD='92117d7d10c11049cf178236144bddf2'

export LOGIN='3e5bab13da3a6503'
export PASSWORD='92117d7d10c11049cf178236144bddf2'
export TEAM_ID='31023'
```

Локальная проверка перед отправкой:

```bash
docker compose up --build -d
python3 eval/ingest.py
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Что писали про Go 1.18?"
    }
  }'
```

Логин в registry:

```bash
docker login 83.166.249.64:5000 -u "$LOGIN" -p "$PASSWORD"
```

Публикация образов:

```bash
cd /Users/boyjayy/Documents/Search/index
make push

cd /Users/boyjayy/Documents/Search/search
make push
```

В итоге в registry должны оказаться:
- `83.166.249.64:5000/31023/index-service:latest`
- `83.166.249.64:5000/31023/search-service:latest`

### Прямой вариант из инструкции хакатона

Если делать без `Makefile`, то команды такие:

Логин:

```bash
docker login 83.166.249.64:5000 -u "$LOGIN" -p "$PASSWORD"
```

Сборка:

```bash
docker build --platform linux/amd64 -t 83.166.249.64:5000/31023/index-service:latest /Users/boyjayy/Documents/Search/index
docker build --platform linux/amd64 -t 83.166.249.64:5000/31023/search-service:latest /Users/boyjayy/Documents/Search/search
```

Публикация:

```bash
docker push 83.166.249.64:5000/31023/index-service:latest
docker push 83.166.249.64:5000/31023/search-service:latest
```
