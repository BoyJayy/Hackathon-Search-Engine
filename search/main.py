import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
REQUIRED_ENV_VARS = [
    "EMBEDDINGS_DENSE_URL",
    "RERANKER_URL",
    "QDRANT_URL",
]
 
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def validate_required_env() -> None:
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")

    missing_env_vars = [
        name for name in REQUIRED_ENV_VARS if os.getenv(name) is None or os.getenv(name) == ""
    ]
    if not missing_env_vars:
        return

    logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
    raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]

# Метадата чанков в Qdrant'e, по которой вы можете фильтровать
class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str # channel, group, private, thread
    chat_id: str
    chat_sn: str
    thread_sn: str | None = None
    message_ids: list[str]
    start: str
    end: str
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    contains_forward: bool = False
    contains_quote: bool = False


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.1.0", lifespan=lifespan)


def getenv_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid integer env %s=%r; using default %s", name, raw, default)
            value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def getenv_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Invalid float env %s=%r; using default %s", name, raw, default)
            value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


DENSE_PREFETCH_K = getenv_int("DENSE_PREFETCH_K", 45, min_value=1)
SPARSE_PREFETCH_K = getenv_int("SPARSE_PREFETCH_K", 45, min_value=1)
RETRIEVE_K = getenv_int("RETRIEVE_K", 60, min_value=1)
RERANK_LIMIT = getenv_int("RERANK_LIMIT", 8, min_value=0)
RERANK_ALPHA = getenv_float("RERANK_ALPHA", 0.5, min_value=0.0, max_value=1.0)
RERANK_MAX_TEXT_CHARS = getenv_int("RERANK_MAX_TEXT_CHARS", 1200, min_value=200)
FINAL_MESSAGE_LIMIT = getenv_int("FINAL_MESSAGE_LIMIT", 50, min_value=1, max_value=50)
MAX_DENSE_QUERIES = getenv_int("MAX_DENSE_QUERIES", 8, min_value=1, max_value=8)
MAX_SPARSE_QUERIES = getenv_int("MAX_SPARSE_QUERIES", 6, min_value=1, max_value=8)
UPSTREAM_CACHE_MAX_ITEMS = getenv_int("UPSTREAM_CACHE_MAX_ITEMS", 20_000, min_value=0)
UPSTREAM_MAX_RETRIES = getenv_int("UPSTREAM_MAX_RETRIES", 1, min_value=0, max_value=3)
UPSTREAM_RETRY_DELAY_SECONDS = getenv_float("UPSTREAM_RETRY_DELAY_SECONDS", 0.25, min_value=0.0, max_value=3.0)
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[\w@./:+-]+", re.UNICODE)
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


def normalize_query_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def unique_texts(texts: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for text in texts:
        normalized = normalize_query_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        items.append(normalized)
        if limit is not None and len(items) >= limit:
            break
    return items


def collect_entity_terms(entities: Entities | None) -> list[str]:
    if entities is None:
        return []

    values = [
        *(entities.people or []),
        *(entities.emails or []),
        *(entities.documents or []),
        *(entities.names or []),
        *(entities.links or []),
    ]
    return unique_texts(values)


def build_primary_query(question: Question) -> str:
    return normalize_query_text(question.search_text or question.text)


def build_dense_queries(question: Question) -> list[str]:
    candidates = [
        question.search_text,
        question.text,
        *((question.variants or [])[:4]),
        *((question.hyde or [])[:3]),
    ]
    return unique_texts(candidates, limit=MAX_DENSE_QUERIES)


def build_sparse_queries(question: Question) -> list[str]:
    primary = build_primary_query(question)
    entity_terms = collect_entity_terms(question.entities)
    focus_terms = unique_texts(
        [
            *(question.keywords or []),
            *entity_terms,
            *(question.date_mentions or []),
            question.asker,
        ]
    )
    exact_focus = " ".join(focus_terms)
    combined = "\n".join(part for part in [primary, exact_focus] if part)
    candidates = [
        combined,
        exact_focus,
        " ".join(entity_terms),
        primary,
        question.text,
        *((question.variants or [])[:2]),
        *((question.hyde or [])[:1]),
    ]
    return unique_texts(candidates, limit=MAX_SPARSE_QUERIES)


def build_rerank_query(question: Question) -> str:
    base_query = build_primary_query(question) or normalize_query_text(question.text)
    clarifiers: list[str] = []
    for term in build_phrase_terms(question):
        if term and term not in base_query.lower():
            clarifiers.append(term)
        if len(clarifiers) >= 2:
            break

    if not clarifiers:
        return base_query

    return "\n".join([base_query, " ".join(clarifiers)])


def dedupe_message_ids(message_ids: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    for message_id in message_ids:
        if message_id in seen:
            continue
        seen.add(message_id)
        unique_ids.append(message_id)
        if len(unique_ids) >= limit:
            break
    return unique_ids


def trim_rerank_text(text: str, *, limit: int) -> str:
    normalized = normalize_query_text(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()} ..."


def normalize_terms(values: list[str]) -> list[str]:
    return unique_texts([value.lower() for value in values if value])


def extract_signal_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(normalize_query_text(text).lower()):
        has_special_signal = any(ch.isdigit() for ch in token) or any(ch in token for ch in "@./:+-_")
        if len(token) >= 4 or has_special_signal:
            tokens.append(token)
    return unique_texts(tokens)


def build_phrase_terms(question: Question) -> list[str]:
    return normalize_terms(
        [
            *(question.keywords or []),
            *collect_entity_terms(question.entities),
            *(question.date_mentions or []),
            question.asker,
        ]
    )


def build_query_signal_tokens(question: Question) -> list[str]:
    candidates = [
        build_primary_query(question),
        question.text,
        *(question.keywords or []),
        *collect_entity_terms(question.entities),
        *((question.variants or [])[:2]),
        *((question.hyde or [])[:2]),
    ]
    tokens: list[str] = []
    for candidate in candidates:
        tokens.extend(extract_signal_tokens(candidate))
    return unique_texts(tokens)


def query_prefers_earliest_message(question: Question) -> bool:
    lowered = " ".join(
        part.lower()
        for part in [
            question.text,
            question.search_text,
            *(question.keywords or []),
            *((question.variants or [])[:2]),
        ]
        if part
    )
    return any(
        marker in lowered
        for marker in (
            "перв",
            "исходн",
            "самое ран",
            "начал",
            "first",
            "earliest",
        )
    )


def parse_timestamp(value: Any, *, end_of_day: bool = False) -> int | None:
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return int(raw)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            dt = datetime.fromisoformat(raw).replace(tzinfo=UTC)
            if end_of_day:
                dt += timedelta(days=1) - timedelta(seconds=1)
            return int(dt.timestamp())
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())

    return None


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
    phrase_terms: list[str],
    token_terms: list[str],
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


def score_best_message_block(
    question: Question,
    point: Any,
    *,
    phrase_terms: list[str],
    token_terms: list[str],
) -> float:
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
            phrase_terms=phrase_terms,
            token_terms=token_terms,
        )
        quoted_score = 0.0
        if quote_flag and quoted_text:
            quoted_score = min(
                score_text_signals(
                    quoted_text,
                    phrase_terms=phrase_terms,
                    token_terms=token_terms,
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
    identity_terms: set[str],
) -> float:
    if not identity_terms:
        return 0.0

    participants = {normalize_query_text(str(item)).lower() for item in (metadata.get("participants") or []) if item}
    mentions = {normalize_query_text(str(item)).lower() for item in (metadata.get("mentions") or []) if item}

    participant_hits = len(participants & identity_terms)
    mention_hits = len(mentions & identity_terms)
    return min(participant_hits * 0.03, 0.06) + min(mention_hits * 0.04, 0.08)


def score_temporal_signal(question: Question, metadata: dict[str, Any]) -> float:
    if question.date_range is None:
        return 0.0

    query_start = parse_timestamp(question.date_range.from_)
    query_end = parse_timestamp(question.date_range.to, end_of_day=True)
    point_start = parse_timestamp(metadata.get("start"))
    point_end = parse_timestamp(metadata.get("end"))

    if None in {query_start, query_end, point_start, point_end}:
        return 0.0

    if point_end < query_start or point_start > query_end:
        return 0.0

    return 0.06


def compute_local_boost(question: Question, point: Any) -> float:
    phrase_terms = build_phrase_terms(question)
    token_terms = build_query_signal_tokens(question)
    identity_terms = set(normalize_terms([question.asker, *collect_entity_terms(question.entities)]))

    message_text = get_point_message_text(point)
    context_text = get_point_context_text(point)
    metadata = get_point_metadata(point)
    message_score = score_text_signals(message_text, phrase_terms=phrase_terms, token_terms=token_terms)
    context_score = score_text_signals(context_text, phrase_terms=phrase_terms, token_terms=token_terms)
    best_block_score = score_best_message_block(
        question,
        point,
        phrase_terms=phrase_terms,
        token_terms=token_terms,
    )
    context_penalty = 0.0
    if context_score > 0 and message_score == 0 and best_block_score == 0:
        context_penalty = 0.08

    return (
        message_score
        + min(context_score * 0.2, 0.05)
        + best_block_score
        + score_metadata_signals(metadata, identity_terms=identity_terms)
        + score_temporal_signal(question, metadata)
        - context_penalty
    )


def rescore_points(question: Question, points: list[Any]) -> list[Any]:
    if not points:
        return []

    rescored: list[tuple[float, float, int, Any]] = []
    for index, point in enumerate(points):
        local_boost = compute_local_boost(question, point)
        base_score = float(getattr(point, "score", 0.0) or 0.0)
        rescored.append((base_score + local_boost, local_boost, -index, point))

    rescored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [point for _, _, _, point in rescored]


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

    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
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


def reorder_message_ids_for_point(question: Question, point: Any) -> list[str]:
    message_ids = extract_message_ids(point)
    if len(message_ids) <= 1:
        return message_ids

    page_content = str(get_point_payload(point).get("page_content") or "")
    blocks = collapse_message_blocks(extract_message_blocks(page_content))
    if len(blocks) != len(message_ids):
        return message_ids

    phrase_terms = build_phrase_terms(question)
    token_terms = build_query_signal_tokens(question)
    prefer_earliest = query_prefers_earliest_message(question)

    scored = []
    for index, (message_id, block) in enumerate(zip(message_ids, blocks, strict=True)):
        block_text = normalize_query_text(block).lower()
        block_score = score_text_signals(
            block_text,
            phrase_terms=phrase_terms,
            token_terms=token_terms,
        )
        if prefer_earliest:
            block_score += max(0.0, 0.04 - index * 0.01)
        scored.append((block_score, -index, message_id))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [message_id for _, _, message_id in scored]


def score_message_block(
    question: Question,
    block: str,
    *,
    message_index: int,
) -> float:
    phrase_terms = build_phrase_terms(question)
    token_terms = build_query_signal_tokens(question)
    header, body = split_block_header_body(block)
    own_body, quoted_body = split_quoted_text(body)
    quote_flag = block_has_flag(block, "quote")

    own_text = normalize_query_text("\n".join(part for part in [header, own_body] if part)).lower()
    quoted_text = normalize_query_text(quoted_body).lower()

    own_score = score_text_signals(
        own_text,
        phrase_terms=phrase_terms,
        token_terms=token_terms,
    )
    quoted_score = 0.0
    if quote_flag and quoted_text:
        quoted_score = min(
            score_text_signals(
                quoted_text,
                phrase_terms=phrase_terms,
                token_terms=token_terms,
            )
            * 0.2,
            0.05,
        )

    block_score = own_score + quoted_score
    if len(own_text) <= 220 and block_score > 0:
        block_score += 0.03
    if query_prefers_earliest_message(question) and quote_flag and own_score == 0 and quoted_score > 0:
        block_score -= 0.05
    if query_prefers_earliest_message(question):
        block_score += max(0.0, 0.04 - message_index * 0.01)
    return block_score


def assemble_message_ids(question: Question, points: list[Any], *, limit: int) -> list[str]:
    scored_messages: list[tuple[float, int, int, str]] = []

    for point_index, point in enumerate(points):
        page_content = str(get_point_payload(point).get("page_content") or "")
        message_ids = extract_message_ids(point)
        blocks = collapse_message_blocks(extract_message_blocks(page_content))
        point_bonus = max(0.0, 0.18 - point_index * 0.004)

        if len(blocks) == len(message_ids) and blocks:
            for message_index, (message_id, block) in enumerate(zip(message_ids, blocks, strict=True)):
                block_score = score_message_block(question, block, message_index=message_index)
                scored_messages.append((point_bonus + block_score, -point_index, -message_index, message_id))
            continue

        for message_index, message_id in enumerate(reorder_message_ids_for_point(question, point)):
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
    question: Question,
    query: str,
    points: list[Any],
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
        rerank_score = score + compute_local_boost(question, point)
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


# Ваш сервис должен имплементировать оба этих метода
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    primary_query = build_primary_query(payload.question)
    if not primary_query:
        raise HTTPException(status_code=400, detail="question.text is required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    dense_queries = build_dense_queries(payload.question)
    sparse_queries = build_sparse_queries(payload.question)

    dense_vectors, sparse_vectors = await asyncio.gather(
        embed_dense_many_safe(client, dense_queries),
        embed_sparse_many(sparse_queries),
    )
    if not dense_vectors and not sparse_vectors:
        return SearchAPIResponse(results=[])

    best_points = await qdrant_search(qdrant, dense_vectors, sparse_vectors)

    if best_points is None:
        return SearchAPIResponse(results=[])

    best_points = rescore_points(payload.question, list(best_points))
    rerank_query = build_rerank_query(payload.question)
    best_points = await rerank_points(client, payload.question, rerank_query, best_points)

    final_message_ids = assemble_message_ids(payload.question, best_points, limit=FINAL_MESSAGE_LIMIT)
    if not final_message_ids:
        return SearchAPIResponse(results=[])

    return SearchAPIResponse(
        results=[SearchAPIItem(message_ids=final_message_ids)]
    )


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
