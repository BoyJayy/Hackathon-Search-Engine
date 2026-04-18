"""Populate Qdrant from a local corpus into Qdrant.

Usage:
    python eval/ingest.py

Env required:
    OPEN_API_LOGIN, OPEN_API_PASSWORD  -- creds for dense API (Basic Auth)

Optional env:
    INDEX_URL                    (default http://localhost:8001)
    QDRANT_URL                   (default http://localhost:6333)
    QDRANT_COLLECTION_NAME       (default evaluation)
    EMBEDDINGS_DENSE_URL         (default http://83.166.249.64:18001/embeddings)
    EMBEDDINGS_DENSE_MODEL       (default Qwen/Qwen3-Embedding-0.6B)
    DATA_PATH                    (default data/Go Nova.json)
    BATCH_SIZE                   (default 16)
    DELETE_EXISTING_CHAT_POINTS  (default 1)
    RESET_COLLECTION             (default 0; useful for synthetic JSONL corpora)
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

INDEX_URL = os.getenv("INDEX_URL", "http://localhost:8001")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL", "http://83.166.249.64:18001/embeddings")
DENSE_MODEL = os.getenv("EMBEDDINGS_DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DENSE_SIZE = int(os.getenv("EMBEDDINGS_DENSE_SIZE", "1024"))
LOGIN = os.environ["OPEN_API_LOGIN"]
PASSWORD = os.environ["OPEN_API_PASSWORD"]
DATA_PATH = Path(os.getenv("DATA_PATH", "data/Go Nova.json"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
DELETE_EXISTING_CHAT_POINTS = os.getenv("DELETE_EXISTING_CHAT_POINTS", "1") == "1"
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0") == "1"

_CHUNK_ID_NAMESPACE = uuid.UUID("6f8c3a1e-0000-0000-0000-000000000001")


def stable_chunk_id(chat_id: str, chunk: dict) -> str:
    page_hash = hashlib.sha1(chunk["page_content"].encode("utf-8"), usedforsecurity=False).hexdigest()
    key = f"{chat_id}:" + ",".join(sorted(chunk["message_ids"])) + f":{page_hash}"
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, key))


def ensure_collection(qc: QdrantClient, name: str, dense_size: int) -> None:
    if qc.collection_exists(name):
        return
    qc.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    print(f"      created collection {name} (dense={dense_size}, sparse=bm25+IDF)")


def recreate_collection(qc: QdrantClient, name: str, dense_size: int) -> None:
    if qc.collection_exists(name):
        qc.delete_collection(name)
    ensure_collection(qc, name, dense_size)


def delete_existing_chat_points(qc: QdrantClient, collection_name: str, chat_id: str) -> None:
    qc.delete(
        collection_name=collection_name,
        wait=True,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.chat_id",
                        match=models.MatchValue(value=chat_id),
                    ),
                ],
            ),
        ),
    )


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
    if not msgs:
        raise ValueError(f"chunk has no resolvable messages: {msg_ids[:3]}")
    times = [m["time"] for m in msgs]
    participants = sorted({m["sender_id"] for m in msgs})
    mentions = sorted({m for msg in msgs for m in (msg.get("mentions") or [])})
    return {
        "chat_name": chat["name"],
        "chat_type": chat["type"],
        "chat_id": chat["id"],
        "chat_sn": chat["sn"],
        "thread_sn": next((m.get("thread_sn") for m in msgs if m.get("thread_sn")), None),
        "message_ids": msg_ids,
        "start": str(min(times)),
        "end": str(max(times)),
        "participants": participants,
        "mentions": mentions,
        "contains_forward": any(m.get("is_forward") for m in msgs),
        "contains_quote": any(m.get("is_quote") for m in msgs),
    }


def load_index_payload(data_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(data_path.read_text())
    chat = data["chat"]
    messages = data["messages"]
    messages_by_id = {m["id"]: m for m in messages}
    return chat, messages, messages_by_id


def load_synthetic_eval_chunks(data_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    chat = {
        "id": f"synthetic://{data_path.stem}",
        "name": f"Synthetic Eval Corpus {data_path.stem}",
        "sn": f"synthetic://{data_path.stem}",
        "type": "group",
        "is_public": False,
        "members_count": 0,
        "members": [],
    }

    chunks: list[dict[str, Any]] = []
    messages_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    base_time = 1_700_000_000

    with data_path.open() as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            answer = entry.get("answer") or {}
            message_ids = [str(message_id) for message_id in (answer.get("message_ids") or []) if message_id]
            answer_text = str(answer.get("text") or "").strip()
            if not message_ids or not answer_text:
                continue

            for message_id in message_ids:
                if message_id in seen_ids:
                    continue
                seen_ids.add(message_id)
                timestamp = base_time + len(seen_ids)
                messages_by_id[message_id] = {
                    "id": message_id,
                    "time": timestamp,
                    "sender_id": "synthetic@eval.local",
                    "mentions": [],
                    "thread_sn": None,
                    "is_forward": False,
                    "is_quote": False,
                }
                chunks.append(
                    {
                        "page_content": (
                            f"CHAT: {chat['name']}\n\n"
                            f"CHAT_TYPE: {chat['type']}\n\n"
                            f"CHAT_ID: {chat['id']}\n\n"
                            "MESSAGES:\n\n"
                            f"[2023-11-14 22:13:{timestamp % 60:02d} UTC | synthetic@eval.local]\n"
                            f"{answer_text}"
                        ),
                        "dense_content": f"chat {chat['name']}\n\nchat_type {chat['type']}\n\n{answer_text}",
                        "sparse_content": (
                            f"{chat['name']}\n{chat['type']}\n{chat['id']}\n"
                            f"{answer_text}\n\nsender: synthetic@eval.local"
                        ),
                        "message_ids": [message_id],
                    }
                )

    return chat, chunks, messages_by_id


def is_synthetic_eval_jsonl(data_path: Path) -> bool:
    if data_path.suffix.lower() != ".jsonl":
        return False
    with data_path.open() as fh:
        first_line = fh.readline().strip()
    if not first_line:
        return False
    try:
        first = json.loads(first_line)
    except json.JSONDecodeError:
        return False
    return "question" in first and "answer" in first


def main() -> None:
    http = httpx.Client()

    synthetic_mode = is_synthetic_eval_jsonl(DATA_PATH)
    if synthetic_mode:
        print(f"[1/4] build synthetic corpus from {DATA_PATH}")
        chat, chunks, messages_by_id = load_synthetic_eval_chunks(DATA_PATH)
        print(f"      -> {len(chunks)} synthetic chunks")
    else:
        chat, messages, messages_by_id = load_index_payload(DATA_PATH)
        print(f"[1/4] POST /index  ({len(messages)} msgs)")
        r = http.post(
            f"{INDEX_URL}/index",
            json={"data": {"chat": chat, "overlap_messages": [], "new_messages": messages}},
            timeout=300.0,
        )
        r.raise_for_status()
        chunks = r.json()["results"]
        print(f"      -> {len(chunks)} chunks")

    print("[2/4] POST /sparse_embedding  (batch)")
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
    if RESET_COLLECTION:
        print(f"      recreating collection {COLLECTION}")
        recreate_collection(qc, COLLECTION, DENSE_SIZE)
    else:
        ensure_collection(qc, COLLECTION, DENSE_SIZE)
    if DELETE_EXISTING_CHAT_POINTS and not RESET_COLLECTION:
        print(f"      deleting existing points for chat {chat['id']}")
        delete_existing_chat_points(qc, COLLECTION, chat["id"])
    points = []
    skipped = 0
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
        if not chunk["message_ids"]:
            skipped += 1
            continue
        points.append(
            models.PointStruct(
                id=stable_chunk_id(chat["id"], chunk),
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
    print(f"done. {len(points)} points (skipped {skipped} empty chunks)")


if __name__ == "__main__":
    main()
