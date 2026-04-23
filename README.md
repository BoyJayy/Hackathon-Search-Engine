# Maple

`Maple` is a local search system for chat history. The project indexes chat messages into Qdrant and returns relevant `message_ids` for a user question.

The repository contains two services:
- `index` prepares searchable chunks from raw chat payloads.
- `search` runs hybrid retrieval over Qdrant and returns ranked `message_ids`.

The current version is local first:
- no external embedding API;
- no external reranker;
- all models run inside the project with `fastembed`;
- the default setup works on `localhost`.

## Project layout

- `index/` — indexing service
- `search/` — search service
- `eval/` — local ingestion and offline evaluation
- `scripts/` — helper utilities
- `data/` — sample datasets and payloads
- `docs/` — project documentation
- `docker-compose.yml` — local stack with Qdrant, index and search

## Documentation

- [docs/architecture.md](docs/architecture.md) — system overview
- [docs/api.md](docs/api.md) — API contracts
- [docs/index_service.md](docs/index_service.md) — indexing service
- [docs/search_service.md](docs/search_service.md) — search service
- [docs/local_development.md](docs/local_development.md) — local run, evaluation and troubleshooting

## Requirements

- Docker and Docker Compose
- Python 3.13 for local `eval` scripts

## Quick start

Start the stack:

```bash
docker compose up --build
```

Available services:
- `Qdrant` — `http://localhost:6333`
- `index` — `http://localhost:8001`
- `search` — `http://localhost:8002`

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:6333/collections
```

Index the sample dataset into Qdrant:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r eval/requirements.txt
python3 eval/ingest.py
```

Run offline evaluation:

```bash
python3 eval/run.py --dataset data/Dataset_main_questions.jsonl --k 50
python3 eval/run.py --dataset data/Dataset_main_questions.jsonl --k 50 --stages
```

## How to use the services

Build chunks from a chat payload:

```bash
curl -X POST "http://localhost:8001/index" \
  -H "Content-Type: application/json" \
  --data-binary @data/index_request_sample.json
```

Run search:

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Что обсуждали про Go 1.18?",
      "search_text": "Go 1.18",
      "keywords": ["Go", "1.18"]
    }
  }'
```

Debug search stages:

```bash
curl -X POST "http://localhost:8002/_debug/search?fusion=dbsf" \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Что обсуждали про Go 1.18?"
    }
  }'
```

## Configuration

The default setup is enough for local work, but the main settings can be overridden with environment variables.

Important variables:
- `QDRANT_COLLECTION_NAME`
- `DENSE_MODEL_NAME`
- `DENSE_VECTOR_SIZE`
- `SPARSE_MODEL_NAME`
- `FUSION_MODE`
- `DENSE_PREFETCH_K`
- `SPARSE_PREFETCH_K`
- `RETRIEVE_K`
- `MAX_DENSE_QUERIES`
- `MAX_SPARSE_QUERIES`

Chunking settings:
- `MAX_CHUNK_CHARS`
- `OVERLAP_MESSAGE_COUNT`
- `MAX_TIME_GAP_SECONDS`
- `LONG_MESSAGE_CHAR_THRESHOLD`
- `LONG_MESSAGE_LINE_THRESHOLD`
- `TECHNICAL_PREVIEW_LINES`
- `TECHNICAL_PREVIEW_CHARS`
- `SPLIT_MESSAGE_CHAR_THRESHOLD`
- `SPLIT_SEGMENT_TARGET_CHARS`

Example:

```bash
QDRANT_COLLECTION_NAME=my_messages \
FUSION_MODE=rrf \
RETRIEVE_K=80 \
docker compose up --build
```

## Useful commands

Run `index` image directly:

```bash
make -C index run
```

Run `search` image directly:

```bash
make -C search run
```

Chunking diagnostics:

```bash
python3 scripts/chunking_diagnostic.py
python3 scripts/chunking_diagnostic.py data/Dataset_main.json
```

Retrieval A B checks:

```bash
python3 scripts/ab_qdrant.py --help
```

Chunking sweep:

```bash
python3 scripts/sweep_chunking.py --help
```

## Typical workflow

1. Start the stack with `docker compose up --build`.
2. Ingest data with `python3 eval/ingest.py`.
3. Test the API with `curl` or Postman.
4. Run offline evaluation with `python3 eval/run.py`.
5. Tune chunking or retrieval parameters with environment variables.

## License

See [LICENSE](LICENSE).
