# API Reference

Это справочник по контрактам проекта. Здесь только API и обязательные поля, без roadmap и без внутренней реализации.

## Сервисы

В проекте есть два сервиса:
- `index-service`
- `search-service`

## Index Service

### `GET /health`

Назначение:
- healthcheck контейнера.

Ожидаемый ответ:

```json
{
  "status": "ok"
}
```

### `POST /index`

Назначение:
- принять новую пачку сообщений;
- построить чанки;
- вернуть тексты для payload, dense и sparse индексации.

Request body:

```json
{
  "data": {
    "chat": {
      "id": "string",
      "name": "string",
      "sn": "string",
      "type": "string",
      "is_public": true,
      "members_count": 0,
      "members": []
    },
    "overlap_messages": [
      {
        "id": "string",
        "thread_sn": "string",
        "time": 0,
        "text": "string",
        "sender_id": "string",
        "file_snippets": "string",
        "parts": [],
        "mentions": [],
        "member_event": null,
        "is_system": false,
        "is_hidden": false,
        "is_forward": false,
        "is_quote": false
      }
    ],
    "new_messages": [
      {
        "id": "string",
        "thread_sn": "string",
        "time": 0,
        "text": "string",
        "sender_id": "string",
        "file_snippets": "string",
        "parts": [],
        "mentions": [],
        "member_event": null,
        "is_system": false,
        "is_hidden": false,
        "is_forward": false,
        "is_quote": false
      }
    ]
  }
}
```

Что означают поля:
- `chat` — metadata чата.
- `overlap_messages` — предыдущий контекст, который надо учитывать при chunking.
- `new_messages` — новые сообщения, которые сейчас индексируются.

Response body:

```json
{
  "results": [
    {
      "page_content": "string",
      "dense_content": "string",
      "sparse_content": "string",
      "message_ids": ["string"]
    }
  ]
}
```

Поля ответа:
- `page_content` — читаемый текст чанка для payload и rerank.
- `dense_content` — текст для dense embedding.
- `sparse_content` — текст для sparse embedding.
- `message_ids` — сообщения, покрываемые чанком.

Требования:
- нельзя менять структуру ответа;
- нельзя терять `message_ids`;
- `results` может содержать несколько чанков на одну пачку сообщений.

### `POST /sparse_embedding`

Назначение:
- построить sparse vectors по batch текстов.

Request body:

```json
{
  "texts": [
    "string"
  ]
}
```

Response body:

```json
{
  "vectors": [
    {
      "indices": [1, 2, 3],
      "values": [0.1, 0.2, 0.3]
    }
  ]
}
```

Требования:
- формат должен быть совместим с Qdrant sparse vectors;
- sparse-модель должна быть доступна локально внутри контейнера.

## Search Service

### `GET /health`

Назначение:
- healthcheck контейнера.

Ожидаемый ответ:

```json
{
  "status": "ok"
}
```

### `POST /search`

Назначение:
- принять вопрос;
- выполнить retrieval по Qdrant;
- при необходимости сделать rerank;
- вернуть отсортированные `message_ids`.

Request body:

```json
{
  "question": {
    "text": "string",
    "asker": "string",
    "asked_on": "string",
    "variants": ["string"],
    "hyde": ["string"],
    "keywords": ["string"],
    "entities": {
      "people": ["string"],
      "emails": ["string"],
      "documents": ["string"],
      "names": ["string"],
      "links": ["string"]
    },
    "date_mentions": ["string"],
    "date_range": {
      "from": "string",
      "to": "string"
    },
    "search_text": "string"
  }
}
```

Response body:

```json
{
  "results": [
    {
      "message_ids": ["string"]
    }
  ]
}
```

Требования:
- нельзя менять структуру ответа;
- результаты должны быть отсортированы по релевантности;
- итоговая метрика считается по `message_ids`.

## Env-переменные

### `index`

Обязательные / используемые:
- `HOST`
- `PORT`

### `search`

Обязательные / используемые:
- `HOST`
- `PORT`
- `API_KEY` или `OPEN_API_LOGIN` + `OPEN_API_PASSWORD`
- `EMBEDDINGS_DENSE_URL`
- `RERANKER_URL`
- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`
- `QDRANT_DENSE_VECTOR_NAME`
- `QDRANT_SPARSE_VECTOR_NAME`

## Внешний API хакатона

Базовый URL:
- `http://83.166.249.64:18001`

Endpoint'ы:
- `POST /embeddings`
- `GET /embeddings/models`
- `POST /score`
- `GET /score/models`

Dense embeddings example:

```bash
curl -u "$OPEN_API_LOGIN:$OPEN_API_PASSWORD" \
  -X POST "http://83.166.249.64:18001/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["Пример поискового запроса"]
  }'
```

Rerank example:

```bash
curl -u "$OPEN_API_LOGIN:$OPEN_API_PASSWORD" \
  -X POST "http://83.166.249.64:18001/score" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-nemotron-rerank-1b-v2",
    "text_1": "Что обсуждали про релиз Go?",
    "text_2": ["Первый кандидат", "Второй кандидат"]
  }'
```

## Что относится к реализации, а не к контракту

Можно менять:
- chunking;
- dense / sparse text preparation;
- retrieval;
- fusion;
- rerank;
- любые эвристики и фильтры.

Нельзя менять:
- форму request/response у `POST /index`;
- форму request/response у `POST /sparse_embedding`;
- форму request/response у `POST /search`.
