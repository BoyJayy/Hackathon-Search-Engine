# API reference

This document describes the public API used by `Maple`.

## Services

- `index`
- `search`

Default local URLs:
- `http://localhost:8001` — index
- `http://localhost:8002` — search

## Index service

### GET /health

Response:

```json
{
  "status": "ok"
}
```

### POST /index

Builds searchable chunks from a chat payload.

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

Response fields:
- `page_content` — readable chunk text
- `dense_content` — text for dense embeddings
- `sparse_content` — text for sparse embeddings
- `message_ids` — message ids covered by the chunk

### POST /sparse_embedding

Builds sparse vectors for a batch of texts.

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

## Search service

### GET /health

Response:

```json
{
  "status": "ok"
}
```

### POST /search

Runs hybrid retrieval and returns ranked `message_ids`.

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

### POST /_debug/search

Runs the same search pipeline but also returns intermediate stage outputs.

Useful query parameters:
- `fusion=dbsf`
- `fusion=rrf`
- `max_dense=1`
- `max_sparse=2`
- `no_rescore=true`

Typical response:

```json
{
  "final": ["message-id-1", "message-id-2"],
  "stages": {
    "retrieval": ["message-id-1", "message-id-3"],
    "rescored": ["message-id-1", "message-id-2"]
  }
}
```

## Environment variables

### index

- `HOST`
- `PORT`
- `MAX_CHUNK_CHARS`
- `OVERLAP_MESSAGE_COUNT`
- `MAX_TIME_GAP_SECONDS`
- `LONG_MESSAGE_CHAR_THRESHOLD`
- `LONG_MESSAGE_LINE_THRESHOLD`
- `TECHNICAL_PREVIEW_LINES`
- `TECHNICAL_PREVIEW_CHARS`
- `SPLIT_MESSAGE_CHAR_THRESHOLD`
- `SPLIT_SEGMENT_TARGET_CHARS`

### search

- `HOST`
- `PORT`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`
- `QDRANT_DENSE_VECTOR_NAME`
- `QDRANT_SPARSE_VECTOR_NAME`
- `DENSE_MODEL_NAME`
- `DENSE_VECTOR_SIZE`
- `SPARSE_MODEL_NAME`
- `FUSION_MODE`
- `DENSE_PREFETCH_K`
- `SPARSE_PREFETCH_K`
- `RETRIEVE_K`
- `MAX_DENSE_QUERIES`
- `MAX_SPARSE_QUERIES`
- `FINAL_MESSAGE_LIMIT`

## Contract stability

Internal implementation may change, but these request and response shapes should remain stable:
- `POST /index`
- `POST /sparse_embedding`
- `POST /search`
