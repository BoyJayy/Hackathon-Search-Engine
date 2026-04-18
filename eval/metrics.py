"""Retrieval metrics: Recall@K and nDCG@K."""
from __future__ import annotations

import math


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for msg_id in top_k if msg_id in relevant)
    return hits / len(relevant)


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, msg_id in enumerate(predicted[:k]):
        if msg_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def score(recall_avg: float, ndcg_avg: float) -> float:
    return recall_avg * 0.8 + ndcg_avg * 0.2
