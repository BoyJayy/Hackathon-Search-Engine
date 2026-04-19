import asyncio
import os
import re
from functools import lru_cache
from typing import Any

import httpx
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models

from config import (
    DENSE_PREFETCH_K,
    EMBEDDINGS_DENSE_MODEL,
    EMBEDDINGS_DENSE_URL,
    FINAL_MESSAGE_LIMIT,
    QDRANT_COLLECTION_NAME,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_SPARSE_VECTOR_NAME,
    RERANK_ALPHA,
    RERANK_LIMIT,
    RERANK_MAX_TEXT_CHARS,
    RERANKER_MODEL,
    RERANKER_URL,
    RETRIEVE_K,
    SPARSE_MODEL_NAME,
    SPARSE_PREFETCH_K,
    UPSTREAM_CACHE_MAX_ITEMS,
    UPSTREAM_MAX_RETRIES,
    UPSTREAM_RETRY_DELAY_SECONDS,
    get_upstream_request_kwargs,
    logger,
)
from querying import (
    QueryContext,
    build_dense_queries,
    build_phrase_terms,
    build_primary_query,
    build_query_context,
    build_query_signal_tokens,
    build_rerank_query,
    build_sparse_queries,
    dedupe_message_ids,
    normalize_query_text,
    parse_timestamp,
    score_intent_alignment,
    trim_rerank_text,
)
from schemas import DenseEmbeddingResponse, SearchAPIRequest, SparseVector


MESSAGE_BLOCK_SPLIT_RE = re.compile(r"\n\n(?=\[\d{4}-\d{2}-\d{2} )")
PART_FLAG_RE = re.compile(r"part=(\d+)/(\d+)")
FLAG_RE = re.compile(r"\|\s*([^\]]+)\]")
QUOTE_MARKER = "Quoted message:"
DENSE_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}
RERANK_SCORE_CACHE: dict[tuple[str, str, str], float] = {}


def cache_set(cache: dict[Any, Any], key: Any, value: Any) -> None:
    if UPSTREAM_CACHE_MAX_ITEMS <= 0:
        return
    if len(cache) >= UPSTREAM_CACHE_MAX_ITEMS:
        cache.pop(next(iter(cache)))
    cache[key] = value


async def post_upstream_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    purpose: str,
) -> httpx.Response:
    for attempt in range(UPSTREAM_MAX_RETRIES + 1):
        response = await client.post(
            url,
            **get_upstream_request_kwargs(),
            json=payload,
        )
        if response.status_code != 429 or attempt >= UPSTREAM_MAX_RETRIES:
            response.raise_for_status()
            return response

        delay = UPSTREAM_RETRY_DELAY_SECONDS * (attempt + 1)
        logger.warning("%s API returned 429; retrying in %.2fs", purpose, delay)
        if delay > 0:
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable upstream retry state")


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def get_point_payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", None) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def get_point_metadata(point: Any) -> dict[str, Any]:
    metadata = get_point_payload(point).get("metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    return metadata


def get_point_text(point: Any) -> str:
    return normalize_query_text(str(get_point_payload(point).get("page_content") or "")).lower()


def split_page_sections(page_content: str) -> tuple[str, str]:
    if not page_content:
        return "", ""

    if "MESSAGES:" not in page_content:
        return "", page_content

    before_messages, messages = page_content.split("MESSAGES:", 1)
    context = ""
    if "CONTEXT:" in before_messages:
        _, context = before_messages.split("CONTEXT:", 1)

    return context.strip(), messages.strip()


def split_block_header_body(block: str) -> tuple[str, str]:
    lines = block.splitlines()
    if not lines:
        return "", ""
    return lines[0].strip(), "\n".join(lines[1:]).strip()


def block_has_flag(block: str, flag: str) -> bool:
    header, _ = split_block_header_body(block)
    match = FLAG_RE.search(header)
    if not match:
        return False
    flags = {item.strip().lower() for item in match.group(1).split(",")}
    return flag.lower() in flags


def split_quoted_text(text: str) -> tuple[str, str]:
    if QUOTE_MARKER not in text:
        return text.strip(), ""

    own_text, quoted_text = text.split(QUOTE_MARKER, 1)
    return own_text.strip(), quoted_text.strip()


def get_point_context_text(point: Any) -> str:
    context_text, _ = split_page_sections(str(get_point_payload(point).get("page_content") or ""))
    return normalize_query_text(context_text).lower()


def get_point_message_text(point: Any) -> str:
    _, message_text = split_page_sections(str(get_point_payload(point).get("page_content") or ""))
    return normalize_query_text(message_text).lower()


def build_rerank_target(point: Any) -> str:
    page_content = str(get_point_payload(point).get("page_content") or "")
    context_text, message_text = split_page_sections(page_content)
    if not message_text:
        return normalize_query_text(page_content)

    sections = [f"MESSAGES:\n{message_text}"]
    if context_text:
        sections.append(f"CONTEXT:\n{context_text}")
    return normalize_query_text("\n\n".join(sections))


def score_text_signals(
    text: str,
    *,
    phrase_terms: list[str] | tuple[str, ...],
    token_terms: list[str] | tuple[str, ...],
) -> float:
    phrase_boost = 0.0
    for term in phrase_terms:
        if term and term in text:
            if any(ch.isdigit() for ch in term) or any(ch in term for ch in "@./:+-_"):
                phrase_boost += 0.07
            else:
                phrase_boost += 0.04

    token_boost = 0.0
    for token in token_terms:
        if token and token in text:
            token_boost += 0.01

    return min(phrase_boost, 0.24) + min(token_boost, 0.08)


def score_best_message_block(ctx: QueryContext, point: Any) -> float:
    page_content = str(get_point_payload(point).get("page_content") or "")
    blocks = collapse_message_blocks(extract_message_blocks(page_content))
    if not blocks:
        return 0.0

    best_score = 0.0
    for block in blocks:
        header, body = split_block_header_body(block)
        own_body, quoted_body = split_quoted_text(body)
        quote_flag = block_has_flag(block, "quote")

        own_text = normalize_query_text("\n".join(part for part in [header, own_body] if part)).lower()
        quoted_text = normalize_query_text(quoted_body).lower()

        own_score = score_text_signals(
            own_text,
            phrase_terms=ctx.phrase_terms,
            token_terms=ctx.token_terms,
        )
        quoted_score = 0.0
        if quote_flag and quoted_text:
            quoted_score = min(
                score_text_signals(
                    quoted_text,
                    phrase_terms=ctx.phrase_terms,
                    token_terms=ctx.token_terms,
                )
                * 0.2,
                0.05,
            )

        block_score = own_score + quoted_score
        if len(own_text) <= 220 and block_score > 0:
            block_score += 0.03
        best_score = max(best_score, block_score)

    return min(best_score, 0.28)


def score_metadata_signals(
    metadata: dict[str, Any],
    *,
    identity_terms: set[str] | frozenset[str],
) -> float:
    if not identity_terms:
        return 0.0

    participants = {normalize_query_text(str(item)).lower() for item in (metadata.get("participants") or []) if item}
    mentions = {normalize_query_text(str(item)).lower() for item in (metadata.get("mentions") or []) if item}

    participant_hits = len(participants & identity_terms)
    mention_hits = len(mentions & identity_terms)
    return min(participant_hits * 0.03, 0.06) + min(mention_hits * 0.04, 0.08)


def score_temporal_signal(ctx: QueryContext, metadata: dict[str, Any]) -> float:
    if ctx.query_start is None or ctx.query_end is None:
        return 0.0

    point_start = parse_timestamp(metadata.get("start"))
    point_end = parse_timestamp(metadata.get("end"))

    if point_start is None or point_end is None:
        return 0.0

    if point_end < ctx.query_start or point_start > ctx.query_end:
        return 0.0

    return 0.06


def compute_local_boost(ctx: QueryContext, point: Any) -> float:
    message_text = get_point_message_text(point)
    context_text = get_point_context_text(point)
    metadata = get_point_metadata(point)
    message_score = score_text_signals(
        message_text,
        phrase_terms=ctx.phrase_terms,
        token_terms=ctx.token_terms,
    )
    context_score = score_text_signals(
        context_text,
        phrase_terms=ctx.phrase_terms,
        token_terms=ctx.token_terms,
    )
    best_block_score = score_best_message_block(ctx, point)
    context_penalty = 0.0
    if context_score > 0 and message_score == 0 and best_block_score == 0:
        context_penalty = 0.08

    return (
        message_score
        + min(context_score * 0.2, 0.05)
        + best_block_score
        + score_intent_alignment(ctx.question, message_text)
        + score_metadata_signals(metadata, identity_terms=ctx.identity_terms)
        + score_temporal_signal(ctx, metadata)
        - context_penalty
    )


def rescore_points(ctx: QueryContext, points: list[Any]) -> tuple[list[Any], dict[Any, float]]:
    if not points:
        return [], {}

    boost_map: dict[Any, float] = {}
    rescored: list[tuple[float, float, int, Any]] = []
    for index, point in enumerate(points):
        local_boost = compute_local_boost(ctx, point)
        boost_map[point.id] = local_boost
        base_score = float(getattr(point, "score", 0.0) or 0.0)
        rescored.append((base_score + local_boost, local_boost, -index, point))

    rescored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [point for _, _, _, point in rescored], boost_map


async def embed_dense_many(
    client: httpx.AsyncClient,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []

    model = os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL)
    results: list[list[float] | None] = [None] * len(texts)
    missing_texts: list[str] = []
    missing_indexes: list[int] = []

    for index, text in enumerate(texts):
        cache_key = (model, text)
        cached = DENSE_EMBED_CACHE.get(cache_key)
        if cached is not None:
            results[index] = cached
            continue
        missing_indexes.append(index)
        missing_texts.append(text)

    if not missing_texts:
        return [result for result in results if result is not None]

    response = await post_upstream_json(
        client,
        EMBEDDINGS_DENSE_URL,
        {
            "model": model,
            "input": missing_texts,
        },
        purpose="Dense embedding",
    )

    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")

    sorted_items = sorted(payload.data, key=lambda item: item.index)
    if len(sorted_items) != len(missing_indexes):
        raise ValueError("Dense embedding response length mismatch")

    for missing_index, item in zip(missing_indexes, sorted_items, strict=True):
        results[missing_index] = item.embedding
        cache_set(DENSE_EMBED_CACHE, (model, texts[missing_index]), item.embedding)

    if any(result is None for result in results):
        raise ValueError("Dense embedding result assembly failed")

    return [result for result in results if result is not None]


async def embed_dense_many_safe(
    client: httpx.AsyncClient,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []

    try:
        return await embed_dense_many(client, texts)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            logger.warning(
                "Dense embedding API rate-limited the request; using sparse-only retrieval for %s queries",
                len(texts),
            )
            return []
        logger.warning(
            "Dense embedding API HTTP error %s; using sparse-only retrieval for %s queries",
            exc.response.status_code,
            len(texts),
        )
        return []
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Dense embedding request failed (%s); using sparse-only retrieval for %s queries",
            exc.__class__.__name__,
            len(texts),
        )
        return []


async def embed_sparse_many(texts: list[str]) -> list[SparseVector]:
    if not texts:
        return []

    vectors = await asyncio.to_thread(lambda: list(get_sparse_model().embed(texts)))
    if not vectors:
        raise ValueError("Sparse embedding response is empty")

    return [
        SparseVector(
            indices=[int(index) for index in item.indices.tolist()],
            values=[float(value) for value in item.values.tolist()],
        )
        for item in vectors
    ]


async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    *,
    fusion: str = "dbsf",
) -> Any | None:
    prefetch: list[models.Prefetch] = [
        models.Prefetch(
            query=dense_vector,
            using=QDRANT_DENSE_VECTOR_NAME,
            limit=DENSE_PREFETCH_K,
        )
        for dense_vector in dense_vectors
    ]
    prefetch.extend(
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            ),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_PREFETCH_K,
        )
        for sparse_vector in sparse_vectors
    )

    if not prefetch:
        return None

    fusion_mode = {
        "rrf": models.Fusion.RRF,
        "dbsf": models.Fusion.DBSF,
    }.get(fusion.lower(), models.Fusion.RRF)

    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=fusion_mode),
        limit=RETRIEVE_K,
        with_payload=True,
    )

    if not response.points:
        return None

    return response.points


def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    message_ids = metadata.get("message_ids") or []

    return [str(message_id) for message_id in message_ids]


def extract_message_blocks(page_content: str) -> list[str]:
    if "MESSAGES:" not in page_content:
        return []

    messages_section = page_content.split("MESSAGES:", 1)[1].strip()
    if not messages_section:
        return []

    return [block.strip() for block in MESSAGE_BLOCK_SPLIT_RE.split(messages_section) if block.strip()]


def collapse_message_blocks(blocks: list[str]) -> list[str]:
    grouped: list[str] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        header = block.splitlines()[0] if block else ""
        part_match = PART_FLAG_RE.search(header)

        if part_match and part_match.group(1) == "1":
            fragment_total = int(part_match.group(2))
            group = blocks[index : index + fragment_total]
            grouped.append("\n\n".join(group).strip())
            index += fragment_total
            continue

        grouped.append(block)
        index += 1

    return grouped


def reorder_message_ids_for_point(ctx: QueryContext, point: Any) -> list[str]:
    message_ids = extract_message_ids(point)
    if len(message_ids) <= 1:
        return message_ids

    page_content = str(get_point_payload(point).get("page_content") or "")
    blocks = collapse_message_blocks(extract_message_blocks(page_content))
    if len(blocks) != len(message_ids):
        return message_ids

    scored = []
    for index, (message_id, block) in enumerate(zip(message_ids, blocks, strict=True)):
        block_text = normalize_query_text(block).lower()
        block_score = score_text_signals(
            block_text,
            phrase_terms=ctx.phrase_terms,
            token_terms=ctx.token_terms,
        )
        if ctx.prefers_earliest:
            block_score += max(0.0, 0.04 - index * 0.01)
        scored.append((block_score, -index, message_id))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [message_id for _, _, message_id in scored]


def score_message_block(ctx: QueryContext, block: str, *, message_index: int) -> float:
    header, body = split_block_header_body(block)
    own_body, quoted_body = split_quoted_text(body)
    quote_flag = block_has_flag(block, "quote")

    own_text = normalize_query_text("\n".join(part for part in [header, own_body] if part)).lower()
    quoted_text = normalize_query_text(quoted_body).lower()

    own_score = score_text_signals(
        own_text,
        phrase_terms=ctx.phrase_terms,
        token_terms=ctx.token_terms,
    )
    quoted_score = 0.0
    if quote_flag and quoted_text:
        quoted_score = min(
            score_text_signals(
                quoted_text,
                phrase_terms=ctx.phrase_terms,
                token_terms=ctx.token_terms,
            )
            * 0.2,
            0.05,
        )

    block_score = own_score + quoted_score
    block_score += score_intent_alignment(ctx.question, own_text)
    if len(own_text) <= 220 and block_score > 0:
        block_score += 0.03
    if ctx.prefers_earliest and quote_flag and own_score == 0 and quoted_score > 0:
        block_score -= 0.05
    if ctx.prefers_earliest:
        block_score += max(0.0, 0.04 - message_index * 0.01)
    return block_score


def assemble_message_ids(ctx: QueryContext, points: list[Any], *, limit: int) -> list[str]:
    scored_messages: list[tuple[float, int, int, str]] = []

    for point_index, point in enumerate(points):
        page_content = str(get_point_payload(point).get("page_content") or "")
        message_ids = extract_message_ids(point)
        blocks = collapse_message_blocks(extract_message_blocks(page_content))
        point_bonus = max(0.0, 0.18 - point_index * 0.004)

        if len(blocks) == len(message_ids) and blocks:
            for message_index, (message_id, block) in enumerate(zip(message_ids, blocks, strict=True)):
                block_score = score_message_block(ctx, block, message_index=message_index)
                scored_messages.append((point_bonus + block_score, -point_index, -message_index, message_id))
            continue

        for message_index, message_id in enumerate(reorder_message_ids_for_point(ctx, point)):
            fallback_bonus = max(0.0, point_bonus - message_index * 0.01)
            scored_messages.append((fallback_bonus, -point_index, -message_index, message_id))

    scored_messages.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    ordered_message_ids = [message_id for _, _, _, message_id in scored_messages]
    return dedupe_message_ids(ordered_message_ids, limit=limit)


async def get_rerank_scores(
    client: httpx.AsyncClient,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []

    results: list[float | None] = [None] * len(targets)
    missing_targets: list[str] = []
    missing_indexes: list[int] = []
    for index, target in enumerate(targets):
        cache_key = (RERANKER_MODEL, label, target)
        cached = RERANK_SCORE_CACHE.get(cache_key)
        if cached is not None:
            results[index] = cached
            continue
        missing_indexes.append(index)
        missing_targets.append(target)

    if not missing_targets:
        return [score for score in results if score is not None]

    # Rerank endpoint возвращает score для пары query -> candidate text.
    response = await post_upstream_json(
        client,
        RERANKER_URL,
        {
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": label,
            "text_2": missing_targets,
        },
        purpose="Reranker",
    )

    payload = response.json()
    data = payload.get("data") or []
    if len(data) != len(missing_indexes):
        raise ValueError("Reranker response length mismatch")

    for missing_index, sample in zip(missing_indexes, data, strict=True):
        score = float(sample["score"])
        results[missing_index] = score
        cache_set(RERANK_SCORE_CACHE, (RERANKER_MODEL, label, targets[missing_index]), score)

    if any(score is None for score in results):
        raise ValueError("Reranker score assembly failed")

    return [score for score in results if score is not None]


async def rerank_points(
    client: httpx.AsyncClient,
    ctx: QueryContext,
    query: str,
    points: list[Any],
    boost_map: dict[Any, float],
) -> list[Any]:
    if not points:
        return []
    if RERANK_LIMIT <= 0 or RERANK_ALPHA <= 0:
        return points

    rerank_candidates = points[:RERANK_LIMIT]
    untouched_tail = points[RERANK_LIMIT:]
    rerank_targets = [
        trim_rerank_text(build_rerank_target(point), limit=RERANK_MAX_TEXT_CHARS)
        for point in rerank_candidates
    ]

    try:
        scores = await get_rerank_scores(client, query, rerank_targets)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            logger.warning(
                "Reranker rate-limited the request; using retrieval order fallback for %s candidates",
                len(rerank_candidates),
            )
            return points
        logger.warning(
            "Reranker HTTP error %s; using retrieval order fallback for %s candidates",
            exc.response.status_code,
            len(rerank_candidates),
        )
        return points
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Reranker request failed (%s); using retrieval order fallback for %s candidates",
            exc.__class__.__name__,
            len(rerank_candidates),
        )
        return points

    if len(scores) != len(rerank_candidates):
        logger.warning(
            "Reranker returned %s scores for %s candidates; using retrieval order fallback",
            len(scores),
            len(rerank_candidates),
        )
        return points

    reranked_candidates: list[Any] = []
    scored_candidates: list[tuple[float, float, float, int, Any]] = []
    total_candidates = max(len(rerank_candidates), 1)
    for index, (score, point) in enumerate(zip(scores, rerank_candidates, strict=True)):
        retrieval_rank_score = 1.0 - (index / total_candidates)
        local_boost = boost_map.get(point.id)
        if local_boost is None:
            local_boost = compute_local_boost(ctx, point)
        rerank_score = score + local_boost
        blended_score = (RERANK_ALPHA * rerank_score) + ((1.0 - RERANK_ALPHA) * retrieval_rank_score)
        scored_candidates.append((blended_score, rerank_score, retrieval_rank_score, -index, point))

    reranked_candidates = [
        point
        for _, _, _, _, point in sorted(
            scored_candidates,
            key=lambda item: (item[0], item[1], item[2], item[3]),
            reverse=True,
        )
    ]

    return reranked_candidates + untouched_tail


async def run_search_pipeline(
    http_client: httpx.AsyncClient,
    qdrant_client: AsyncQdrantClient,
    payload: SearchAPIRequest,
    *,
    skip_rescore: bool = False,
    skip_rerank: bool = False,
    collect_stages: bool = False,
    fusion: str = "dbsf",
    max_dense: int | None = None,
    max_sparse: int | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    primary_query = build_primary_query(payload.question)
    if not primary_query:
        raise ValueError("question.text is required")

    ctx = build_query_context(payload.question)
    dense_queries = build_dense_queries(payload.question)
    sparse_queries = build_sparse_queries(payload.question)
    if max_dense is not None:
        dense_queries = dense_queries[: max(0, max_dense)]
    if max_sparse is not None:
        sparse_queries = sparse_queries[: max(0, max_sparse)]

    dense_vectors, sparse_vectors = await asyncio.gather(
        embed_dense_many_safe(http_client, dense_queries),
        embed_sparse_many(sparse_queries),
    )
    if not dense_vectors and not sparse_vectors:
        return [], {}

    best_points = await qdrant_search(qdrant_client, dense_vectors, sparse_vectors, fusion=fusion)

    if best_points is None:
        return [], {}

    stages: dict[str, list[str]] = {}
    best_points = list(best_points)
    if collect_stages:
        stages["retrieval"] = assemble_message_ids(ctx, best_points, limit=FINAL_MESSAGE_LIMIT)

    if skip_rescore:
        boost_map: dict[Any, float] = {}
    else:
        best_points, boost_map = rescore_points(ctx, best_points)
        if collect_stages:
            stages["rescored"] = assemble_message_ids(ctx, best_points, limit=FINAL_MESSAGE_LIMIT)

    if not skip_rerank:
        rerank_query = build_rerank_query(payload.question)
        best_points = await rerank_points(http_client, ctx, rerank_query, best_points, boost_map)
        if collect_stages:
            stages["reranked"] = assemble_message_ids(ctx, best_points, limit=FINAL_MESSAGE_LIMIT)

    final_message_ids = assemble_message_ids(ctx, best_points, limit=FINAL_MESSAGE_LIMIT)
    return final_message_ids, stages
