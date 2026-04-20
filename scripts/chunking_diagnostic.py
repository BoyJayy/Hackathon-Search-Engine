"""Quick chunking diagnostics for local datasets.

Usage:
    python3 scripts/chunking_diagnostic.py
    python3 scripts/chunking_diagnostic.py data/Dataset_main.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "index"))

from chunking import build_chunks, is_message_searchable, normalize_message
from schemas import Chat, Message


def histogram(values: list[int], bins: list[tuple[int, int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for value in values:
        for index, (lo, hi) in enumerate(bins):
            if lo <= value < hi:
                counts[index] += 1
                break
    return counts


def load_dataset(path: Path) -> tuple[Chat, list[Message]]:
    raw = json.loads(path.read_text())
    if "chat" not in raw or "messages" not in raw:
        data = raw["data"]
        raw = {"chat": data["chat"], "messages": data["new_messages"]}
    chat_payload = raw["chat"]
    messages_payload = raw["messages"]
    chat = Chat(**chat_payload)
    messages = [Message(**item) for item in messages_payload]
    return chat, messages


def printable_thread_count(messages: list[Message]) -> int:
    return len({message.thread_sn or "__root__" for message in messages})


def main() -> int:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / "Dataset_main.json"
    chat, messages = load_dataset(dataset_path)

    normalized_messages = [normalize_message(message, is_overlap=False) for message in messages]
    kept_messages = [message for message in normalized_messages if is_message_searchable(message)]
    chunks = build_chunks(chat, [], messages)

    print(f"dataset path    : {dataset_path}")
    print(f"chat            : {chat.name} ({chat.id})")
    print(f"messages total  : {len(messages)}")
    print(f"messages kept   : {len(kept_messages)}")
    print(f"thread count    : {printable_thread_count(messages)}")
    print(f"chunks          : {len(chunks)}")
    print()

    if not chunks:
        print("No chunks produced.")
        return 0

    page_sizes = [len(chunk.page_content) for chunk in chunks]
    dense_sizes = [len(chunk.dense_content) for chunk in chunks]
    sparse_sizes = [len(chunk.sparse_content) for chunk in chunks]
    chunk_message_counts = [len(chunk.message_ids) for chunk in chunks]

    print("=== Chunk sizes ===")
    print(
        "page  : min=%s max=%s mean=%.0f median=%.0f"
        % (min(page_sizes), max(page_sizes), statistics.mean(page_sizes), statistics.median(page_sizes))
    )
    print(
        "dense : min=%s max=%s mean=%.0f median=%.0f"
        % (min(dense_sizes), max(dense_sizes), statistics.mean(dense_sizes), statistics.median(dense_sizes))
    )
    print(
        "sparse: min=%s max=%s mean=%.0f median=%.0f"
        % (min(sparse_sizes), max(sparse_sizes), statistics.mean(sparse_sizes), statistics.median(sparse_sizes))
    )
    print()

    print("=== Messages per chunk ===")
    print(
        "count : min=%s max=%s mean=%.1f median=%.1f"
        % (
            min(chunk_message_counts),
            max(chunk_message_counts),
            statistics.mean(chunk_message_counts),
            statistics.median(chunk_message_counts),
        )
    )
    print()

    size_bins = [(0, 400), (400, 800), (800, 1200), (1200, 1600), (1600, 2000), (2000, 10000)]
    count_bins = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 100)]

    print("=== Page size histogram ===")
    size_hist = histogram(page_sizes, size_bins)
    for index, (lo, hi) in enumerate(size_bins):
        bar = "#" * size_hist.get(index, 0)
        print(f"[{lo:>5}..{hi:<5}) {size_hist.get(index, 0):>2} {bar}")
    print()

    print("=== Message-count histogram ===")
    count_hist = histogram(chunk_message_counts, count_bins)
    for index, (lo, hi) in enumerate(count_bins):
        bar = "#" * count_hist.get(index, 0)
        print(f"[{lo:>2}..{hi:<3}) {count_hist.get(index, 0):>2} {bar}")
    print()

    message_occurrences: Counter[str] = Counter()
    for chunk in chunks:
        for message_id in chunk.message_ids:
            message_occurrences[message_id] += 1

    kept_ids = {message.id for message in kept_messages}
    covered = sum(1 for message_id in kept_ids if message_occurrences[message_id] > 0)
    dup_ratio = (
        sum(message_occurrences.values()) / len(message_occurrences)
        if message_occurrences
        else 0.0
    )

    print("=== Coverage ===")
    print(f"kept msg covered : {covered}/{len(kept_ids)} ({(100 * covered / len(kept_ids)):.0f}%)")
    print(f"dup ratio        : {dup_ratio:.2f}x")
    uncovered = sorted(kept_ids - set(message_occurrences))
    if uncovered:
        print(f"uncovered sample : {uncovered[:10]}")
    print()

    print("=== Chunk preview ===")
    for index, chunk in enumerate(chunks):
        first_id = chunk.message_ids[0][-6:]
        last_id = chunk.message_ids[-1][-6:]
        preview = chunk.page_content[:80].replace("\n", " ")
        print(
            f"{index:>2} msgs={len(chunk.message_ids):>2} page={len(chunk.page_content):>4} "
            f"dense={len(chunk.dense_content):>4} sparse={len(chunk.sparse_content):>4} "
            f"[{first_id}..{last_id}] {preview!r}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
