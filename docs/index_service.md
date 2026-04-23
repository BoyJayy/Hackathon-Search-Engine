# Index service

The `index` service converts raw chat payloads into searchable chunks. It does not write data to Qdrant directly. Instead, it returns chunk texts and `message_ids`, and exposes a local sparse embedding endpoint.

## Responsibilities

- normalize chat messages;
- filter empty or low value messages;
- split large messages into smaller segments;
- preserve limited overlap between batches;
- build chunk text for payload, dense search and sparse search;
- generate sparse embeddings for arbitrary text batches.

## Module structure

- `index/main.py` — FastAPI app and routes
- `index/config.py` — runtime and chunking settings
- `index/schemas.py` — request and response models
- `index/chunking.py` — normalization and chunk building
- `index/sparse.py` — sparse embedding wrapper

## Endpoints

- `GET /health`
- `POST /index`
- `POST /sparse_embedding`

## Request flow

```text
chat + overlap_messages + new_messages
  -> normalize messages
  -> render text from text, parts and events
  -> filter noise
  -> split long content
  -> build chunks
  -> return page_content, dense_content, sparse_content, message_ids
```

## Input data

`POST /index` accepts:
- chat metadata;
- `overlap_messages` from the previous batch;
- `new_messages` to index now.

This lets the service preserve short range context between adjacent indexing batches.

## Normalization

For each message the service may use:
- `text`
- text fragments from `parts`
- quoted fragments from `parts`
- `file_snippets`
- `member_event`

The output is converted to a normalized internal message representation with:
- `id`
- `time`
- `sender_id`
- `thread_sn`
- normalized text
- mentions
- message flags

## Filtering

The service drops:
- hidden messages;
- empty messages;
- short acknowledgment messages with no useful signal.

Examples of filtered noise:
- `ok`
- `thanks`
- `+`
- `понял`

## Long messages

Large messages are handled in two ways:
- technical traces and logs are compressed into a preview;
- long normal text is split into smaller semantic segments.

This keeps chunks readable and makes retrieval more stable.

## Chunk construction

Chunks are built incrementally from normalized messages. A chunk is flushed when:
- it grows beyond `MAX_CHUNK_CHARS`;
- a large time gap appears;
- message boundaries make the current chunk too large.

The service also keeps a short overlap context from previous messages.

## Output fields

### page_content

Human readable chunk text used as payload. It contains:
- chat header;
- optional context block;
- message block.

### dense_content

Compact semantic text used to compute dense embeddings.

### sparse_content

Token preserving text used to compute sparse embeddings.

### message_ids

The ordered list of message ids represented by the chunk.

## Sparse embeddings

`POST /sparse_embedding` accepts a list of texts and returns sparse vectors compatible with Qdrant. The service uses `fastembed` with `Qdrant/bm25` by default.

## What the service does not do

`index` does not:
- compute dense embeddings;
- create or update Qdrant collections;
- write points to Qdrant;
- execute retrieval or ranking.

These steps are handled by `eval/ingest.py` or by external callers.
