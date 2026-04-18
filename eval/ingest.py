"""Populate Qdrant from data/Go Nova.json by calling index service + dense API.

Usage:
    python eval/ingest.py

Env required:
    OPEN_API_LOGIN, OPEN_API_PASSWORD  -- creds for dense API (Basic Auth)

Optional env:
    INDEX_URL                (default http://localhost:8001)
    QDRANT_URL               (default http://localhost:6333)
    QDRANT_COLLECTION_NAME   (default evaluation)
    EMBEDDINGS_DENSE_URL     (default http://83.166.249.64:18001/embeddings)
    EMBEDDINGS_DENSE_MODEL   (default Qwen/Qwen3-Embedding-0.6B)
    DATA_PATH                (default data/Go Nova.json)
    BATCH_SIZE               (default 16)
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import httpx
from qdrant_client import QdrantClient, models

INDEX_URL = os.getenv("INDEX_URL", "http://localhost:8001")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL", "http://83.166.249.64:18001/embeddings")
DENSE_MODEL = os.getenv("EMBEDDINGS_DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
LOGIN = os.environ["OPEN_API_LOGIN"]
PASSWORD = os.environ["OPEN_API_PASSWORD"]
DATA_PATH = Path(os.getenv("DATA_PATH", "data/Go Nova.json"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))


def embed_dense_batch(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    r = client.post(
        DENSE_URL,
        json={"model": DENSE_MODEL, "input": texts},
        auth=(LOGIN, PASSWORD),
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


def build_metadata(chat: dict, chunk: dict, messages_by_id: dict) -> dict:
    msg_ids = chunk["message_ids"]
    msgs = [messages_by_id[mid] for mid in msg_ids if mid in messages_by_id]
    times = [m["time"] for m in msgs] or [0]
    participants = sorted({m["sender_id"] for m in msgs})
    mentions = sorted({m for msg in msgs for m in (msg.get("mentions") or [])})
    return {
        "chat_name": chat["name"],
        "chat_type": chat["type"],
        "chat_id": chat["id"],
        "chat_sn": chat["sn"],
        "message_ids": msg_ids,
        "start": str(min(times)),
        "end": str(max(times)),
        "participants": participants,
        "mentions": mentions,
        "contains_forward": any(m.get("is_forward") for m in msgs),
        "contains_quote": any(m.get("is_quote") for m in msgs),
    }


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    chat = data["chat"]
    messages = data["messages"]
    messages_by_id = {m["id"]: m for m in messages}

    http = httpx.Client()

    print(f"[1/4] POST /index  ({len(messages)} msgs)")
    r = http.post(
        f"{INDEX_URL}/index",
        json={"data": {"chat": chat, "overlap_messages": [], "new_messages": messages}},
        timeout=300.0,
    )
    r.raise_for_status()
    chunks = r.json()["results"]
    print(f"      -> {len(chunks)} chunks")

    print(f"[2/4] POST /sparse_embedding  (batch)")
    r = http.post(
        f"{INDEX_URL}/sparse_embedding",
        json={"texts": [c["sparse_content"] for c in chunks]},
        timeout=300.0,
    )
    r.raise_for_status()
    sparse_vectors = r.json()["vectors"]
    assert len(sparse_vectors) == len(chunks)

    print(f"[3/4] dense embed  ({len(chunks)} in batches of {BATCH_SIZE})")
    dense_vectors: list[list[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = [c["dense_content"] for c in chunks[i : i + BATCH_SIZE]]
        dense_vectors.extend(embed_dense_batch(http, batch))
        print(f"      batch {i // BATCH_SIZE + 1} done")

    print(f"[4/4] Qdrant upsert  -> {COLLECTION}")
    qc = QdrantClient(url=QDRANT_URL)
    points = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                },
                payload={
                    "page_content": chunk["page_content"],
                    "metadata": build_metadata(chat, chunk, messages_by_id),
                },
            )
        )
    qc.upsert(collection_name=COLLECTION, points=points)
    print(f"done. {len(points)} points")


if __name__ == "__main__":
    main()
