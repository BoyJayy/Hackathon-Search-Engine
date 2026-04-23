# Search service

The `search` service accepts a question, runs hybrid retrieval against Qdrant and returns ranked `message_ids`.

The current implementation is local first:
- dense embeddings are computed locally with `fastembed`;
- sparse embeddings are computed locally with `fastembed`;
- there is no external reranker.

## Responsibilities

- prepare the primary search query;
- build dense and sparse query variants;
- compute query embeddings locally;
- fetch candidates from Qdrant;
- combine dense and sparse retrieval with fusion;
- apply lightweight local rescoring;
- assemble final `message_ids`.

## Module structure

- `search/main.py` — FastAPI app and endpoints
- `search/config.py` — runtime settings
- `search/schemas.py` — API models
- `search/querying.py` — query preparation
- `search/pipeline.py` — embeddings, retrieval, fusion, rescoring and assembly

## Endpoints

- `GET /health`
- `POST /search`
- `POST /_debug/search`

## Search flow

```text
question
  -> build search context
  -> build dense queries
  -> build sparse queries
  -> local embeddings
  -> Qdrant hybrid retrieval
  -> fusion
  -> rescoring
  -> message_id assembly
```

## Query preparation

The primary query is built from:
- `question.search_text`, if present;
- otherwise `question.text`.

Additional signal comes from:
- `keywords`
- `entities`
- `date_mentions`
- `variants`
- `asker`

The service extracts a compact set of exact terms and uses them in two places:
- as a compact sparse query;
- as local exact match signals during rescoring.

## Dense queries

Dense queries are built from a small set of normalized strings, typically:
- primary query;
- raw `question.text`;
- the first variant, if present;
- compact exact term query.

The number of dense queries is limited by `MAX_DENSE_QUERIES`.

## Sparse queries

Sparse queries are built from:
- exact terms;
- primary query;
- raw `question.text`;
- the first variant, if present.

The number of sparse queries is limited by `MAX_SPARSE_QUERIES`.

## Retrieval

The service embeds query variants locally, then sends them to Qdrant as multiple prefetch branches:
- dense branches use the `dense` vector field;
- sparse branches use the `sparse` vector field.

The result sets are merged with fusion:
- `dbsf`
- or `rrf`

Main parameters:
- `DENSE_PREFETCH_K`
- `SPARSE_PREFETCH_K`
- `RETRIEVE_K`
- `FUSION_MODE`

## Rescoring

After retrieval the service applies a lightweight local rescore.

Signals include:
- exact term hits in the message block;
- exact term hits in the context block;
- exact term hits in metadata such as participants and mentions;
- original point rank.

The goal is to slightly prefer candidates that contain stronger exact evidence without introducing a heavy rerank stage.

## Final assembly

The final response is built from payload `message_ids`.

If the number of rendered message blocks matches the number of `message_ids`, the service also uses per block term hits to improve local ordering inside the chunk.

The final list is:
- deduplicated;
- limited by `FINAL_MESSAGE_LIMIT`;
- returned as `results[].message_ids`.

## Debug endpoint

`POST /_debug/search` returns both the final output and intermediate stage outputs.

Useful query parameters:
- `fusion=dbsf`
- `fusion=rrf`
- `max_dense=1`
- `max_sparse=2`
- `no_rescore=true`

This endpoint is intended for local analysis and A B testing.

## Default models

- dense: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- sparse: `Qdrant/bm25`

Both can be overridden with environment variables.
