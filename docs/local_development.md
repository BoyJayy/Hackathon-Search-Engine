# Local development

This document covers the normal local workflow for `Maple`.

## 1. Start the services

```bash
docker compose up --build
```

The stack exposes:
- `http://localhost:6333` — Qdrant
- `http://localhost:8001` — index
- `http://localhost:8002` — search

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:6333/collections
```

## 2. Prepare Python environment for evaluation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r eval/requirements.txt
```

## 3. Ingest data into Qdrant

By default `eval/ingest.py` uses:
- `data/Dataset_main.json`
- `http://localhost:8001`
- `http://localhost:6333`
- collection `messages`

Run:

```bash
python3 eval/ingest.py
```

Common overrides:

```bash
DATA_PATH=data/Dataset_sweep.json python3 eval/ingest.py

QDRANT_COLLECTION_NAME=my_messages \
DATA_PATH=data/Dataset_main.json \
python3 eval/ingest.py
```

If you want to recreate the collection:

```bash
RESET_COLLECTION=1 python3 eval/ingest.py
```

## 4. Run search evaluation

Main dataset:

```bash
python3 eval/run.py --dataset data/Dataset_main_questions.jsonl --k 50
```

With stage output:

```bash
python3 eval/run.py --dataset data/Dataset_main_questions.jsonl --k 50 --stages
```

Sweep dataset:

```bash
python3 eval/run.py --dataset data/Dataset_sweep_questions.jsonl --k 50
```

## 5. Manual API checks

Index sample payload:

```bash
curl -X POST "http://localhost:8001/index" \
  -H "Content-Type: application/json" \
  --data-binary @data/index_request_sample.json
```

Sparse embedding endpoint:

```bash
curl -X POST "http://localhost:8001/sparse_embedding" \
  -H "Content-Type: application/json" \
  -d '{"texts":["Go 1.18 release notes"]}'
```

Search:

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Что обсуждали про Go 1.18?",
      "search_text": "Go 1.18"
    }
  }'
```

Debug search:

```bash
curl -X POST "http://localhost:8002/_debug/search?fusion=dbsf" \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Что обсуждали про Go 1.18?"
    }
  }'
```

## 6. Changing ports or URLs

If you want another local URL, override ports in `docker-compose.yml` or pass environment variables.

Example:

```bash
QDRANT_COLLECTION_NAME=demo \
RETRIEVE_K=80 \
docker compose up --build
```

For direct image runs:

```bash
make -C index run
make -C search run
```

`search` Makefile accepts:
- `PORT`
- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`

## 7. Tuning

Useful search variables:
- `FUSION_MODE=dbsf|rrf`
- `DENSE_PREFETCH_K`
- `SPARSE_PREFETCH_K`
- `RETRIEVE_K`
- `MAX_DENSE_QUERIES`
- `MAX_SPARSE_QUERIES`

Useful index variables:
- `MAX_CHUNK_CHARS`
- `OVERLAP_MESSAGE_COUNT`
- `MAX_TIME_GAP_SECONDS`
- `SPLIT_MESSAGE_CHAR_THRESHOLD`
- `SPLIT_SEGMENT_TARGET_CHARS`

Example:

```bash
MAX_CHUNK_CHARS=2200 \
OVERLAP_MESSAGE_COUNT=3 \
FUSION_MODE=rrf \
RETRIEVE_K=80 \
docker compose up --build
```

## 8. Helpful scripts

Chunking diagnostics:

```bash
python3 scripts/chunking_diagnostic.py
python3 scripts/chunking_diagnostic.py data/Dataset_main.json
```

Chunking sweep:

```bash
python3 scripts/sweep_chunking.py --help
```

Qdrant retrieval checks:

```bash
python3 scripts/ab_qdrant.py --help
```

## 9. Troubleshooting

If `search` fails to start:
- rebuild the image with `docker compose up --build`;
- check that Qdrant is available on `localhost:6333`;
- confirm that the collection name matches `QDRANT_COLLECTION_NAME`.

If evaluation returns no results:
- ingest the dataset again with `python3 eval/ingest.py`;
- check the collection in Qdrant;
- verify that `search` and `eval` use the same collection name.

If models download slowly on first start:
- wait for the first image build to finish;
- later runs will reuse the local model cache inside the image.
