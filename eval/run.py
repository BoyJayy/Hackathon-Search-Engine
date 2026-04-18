"""Run eval dataset through search service, compute Recall@K and nDCG@K.

Usage:
    python eval/run.py
    python eval/run.py --k 50 --dataset eval/dataset.jsonl

Assumes Qdrant already populated (see eval/ingest.py).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import httpx

from metrics import ndcg_at_k, recall_at_k, score

SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8002")


def run(dataset_path: Path, k: int, verbose: bool) -> None:
    entries = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    http = httpx.Client(timeout=120.0)

    recalls: list[float] = []
    ndcgs: list[float] = []
    misses: list[dict] = []

    for entry in entries:
        qid = entry["id"]
        question = entry["question"]
        gt = set(entry["answer"]["message_ids"])

        r = http.post(f"{SEARCH_URL}/search", json={"question": question})
        r.raise_for_status()
        results = r.json().get("results") or []
        predicted: list[str] = results[0]["message_ids"] if results else []

        r_k = recall_at_k(predicted, gt, k)
        n_k = ndcg_at_k(predicted, gt, k)
        recalls.append(r_k)
        ndcgs.append(n_k)

        if r_k < 1.0:
            missed = gt - set(predicted[:k])
            misses.append({"id": qid, "recall": r_k, "missed": sorted(missed)})

        if verbose:
            print(f"  {qid}  R@{k}={r_k:.3f}  nDCG@{k}={n_k:.3f}  '{question['text'][:60]}'")

    recall_avg = statistics.mean(recalls) if recalls else 0.0
    ndcg_avg = statistics.mean(ndcgs) if ndcgs else 0.0
    s = score(recall_avg, ndcg_avg)

    print()
    print(f"N         = {len(entries)}")
    print(f"Recall@{k} = {recall_avg:.4f}")
    print(f"nDCG@{k}   = {ndcg_avg:.4f}")
    print(f"score     = {s:.4f}   (0.8 * recall + 0.2 * nDCG)")
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses:
            print(f"  {m['id']}  R={m['recall']:.3f}  missed={m['missed']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("eval/dataset.jsonl"))
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    run(args.dataset, args.k, args.verbose)


if __name__ == "__main__":
    main()
