# Advanced hackathon solution from Samsung IC and VK tech

Проект состоит из двух сервисов:
- `index` — принимает чат, нормализует сообщения и возвращает чанки для индексации;
- `search` — принимает вопрос, выполняет hybrid retrieval и возвращает `message_ids`.

Контракты API фиксированы. Внутренняя реализация может меняться.

## Документация

- [docs/01_architecture.md](docs/01_architecture.md) — общая схема решения.
- [docs/api.md](docs/api.md) — API сервисов.
- [docs/index_service.md](docs/index_service.md) — устройство `index`.
- [docs/search_service.md](docs/search_service.md) — устройство `search`.
- [docs/process.md](docs/process.md) — полный путь от входного JSON до ответа поиска.

## Структура репозитория

- `index/`
  - `main.py` — FastAPI-приложение
  - `config.py` — настройки
  - `schemas.py` — входные и выходные схемы
  - `chunking.py` — нормализация, фильтрация, chunking
  - `sparse.py` — локальные sparse embeddings
- `search/`
  - `main.py` — FastAPI-приложение
  - `config.py` — настройки и env
  - `schemas.py` — схемы API
  - `querying.py` — подготовка query и query context
  - `pipeline.py` — retrieval, rescoring, rerank, сборка результата
- `eval/` — локальный ingest и offline-оценка
- `scripts/` — вспомогательные диагностические и sweep-скрипты
- `docker-compose.yml` — локальный запуск `qdrant`, `index`, `search`

## Быстрый старт

Нужны переменные окружения для внешнего dense/rerank API:

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

## Локальная индексация и оценка

Заполнить Qdrant:

```bash
python3 eval/ingest.py
```

Запустить оценку:

```bash
python3 eval/run.py --dataset data/Dataset_main_questions.jsonl --k 50
```

## Полезные команды

Проверка `index`:

```bash
curl http://localhost:8001/health
curl -X POST "http://localhost:8001/index" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/index_payload.json
```

Проверка `search`:

```bash
curl http://localhost:8002/health
```

Диагностика chunking:

```bash
python3 scripts/chunking_diagnostic.py
python3 scripts/chunking_diagnostic.py data/Dataset_main.json
```

Chunking sweep:

```bash
python3 scripts/sweep_chunking.py --phase smoke
```

## Деплой образов

Registry:
- `83.166.249.64:5000`

Образы должны собираться под `linux/amd64`.

```bash
docker build --platform linux/amd64 -t 83.166.249.64:5000/31023/index-service:latest ./index
docker build --platform linux/amd64 -t 83.166.249.64:5000/31023/search-service:latest ./search

docker push 83.166.249.64:5000/31023/index-service:latest
docker push 83.166.249.64:5000/31023/search-service:latest
```
