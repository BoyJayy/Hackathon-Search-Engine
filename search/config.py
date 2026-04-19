import logging
import os
from typing import Any


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


def getenv_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
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


def getenv_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
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


DENSE_PREFETCH_K = getenv_int("DENSE_PREFETCH_K", 70, min_value=1)
SPARSE_PREFETCH_K = getenv_int("SPARSE_PREFETCH_K", 45, min_value=1)
RETRIEVE_K = getenv_int("RETRIEVE_K", 150, min_value=1)
RERANK_LIMIT = getenv_int("RERANK_LIMIT", 20, min_value=0)
RERANK_ALPHA = getenv_float("RERANK_ALPHA", 0.3, min_value=0.0, max_value=1.0)
RERANK_MAX_TEXT_CHARS = getenv_int("RERANK_MAX_TEXT_CHARS", 1200, min_value=200)
FINAL_MESSAGE_LIMIT = getenv_int("FINAL_MESSAGE_LIMIT", 50, min_value=1, max_value=50)
MAX_DENSE_QUERIES = getenv_int("MAX_DENSE_QUERIES", 8, min_value=1, max_value=8)
MAX_SPARSE_QUERIES = getenv_int("MAX_SPARSE_QUERIES", 8, min_value=1, max_value=8)
UPSTREAM_CACHE_MAX_ITEMS = getenv_int("UPSTREAM_CACHE_MAX_ITEMS", 20_000, min_value=0)
UPSTREAM_MAX_RETRIES = getenv_int("UPSTREAM_MAX_RETRIES", 1, min_value=0, max_value=3)
UPSTREAM_RETRY_DELAY_SECONDS = getenv_float(
    "UPSTREAM_RETRY_DELAY_SECONDS",
    0.25,
    min_value=0.0,
    max_value=3.0,
)
INTENT_ALIGNMENT_WEIGHT = getenv_float(
    "INTENT_ALIGNMENT_WEIGHT",
    0.0,
    min_value=0.0,
    max_value=1.0,
)
