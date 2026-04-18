import logging
import os
from functools import lru_cache
from typing import Any
import asyncio
import hashlib

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str  # group, channel, private
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None


class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool


class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]


class IndexAPIRequest(BaseModel):
    data: ChatData


# dense_content будет передан в dense embedding модель для построения семантического вектора.
# sparse_content будет передан в sparse модель для построения разреженного индекса "по словам".
# Можно оставить dense_content и sparse_content равными page_content,
# а можно формировать для них разные версии текста.
class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]


class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]


class SparseEmbeddingRequest(BaseModel):
    texts: list[str]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


app = FastAPI(title="Index Service", version="0.1.0")

# Message-boundary chunking — см. docs/ml-roadmap.md Этап 1

LOWER_CHARS = 400         # ниже порога — не эмитим на size-boundary, продолжаем копить
UPPER_CHARS = 1600        # достигли — режем
TIME_GAP_SECONDS = 3600   # > 1ч между сообщениями → разрыв
OVERLAP_MESSAGES = 2      # хвост переносится в следующий чанк (только внутри thread/time window)

SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"

# Важная переманная, которая позволяет вычислять sparse вектор в несколько ядер. Не рекомендуется изменять.
UVICORN_WORKERS=8


def render_message(message: Message) -> str:
    parts: list[str] = []
    if message.text:
        parts.append(message.text.strip())
    if message.parts:
        for part in message.parts:
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                parts.append(part_text.strip())
    if message.file_snippets:
        parts.append(message.file_snippets.strip())
    return "\n".join(p for p in parts if p)


def keep_message(m: Message) -> bool:
    if m.is_hidden:
        return False
    if m.is_system:
        return False
    return True


def build_chunks(
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    all_messages = overlap_messages + new_messages
    rendered: list[tuple[Message, str]] = []
    for m in all_messages:
        if not keep_message(m):
            continue
        text = render_message(m)
        if not text:
            continue
        rendered.append((m, text))
    rendered.sort(key=lambda pair: (pair[0].time, pair[0].id))

    new_ids = {m.id for m in new_messages}

    groups: list[list[tuple[Message, str]]] = []
    current: list[tuple[Message, str]] = []
    current_len = 0

    for m, text in rendered:
        if current:
            prev = current[-1][0]
            same_thread = (m.thread_sn or "") == (prev.thread_sn or "")
            gap = (m.time - prev.time) if (m.time and prev.time) else 0
            hard_boundary = (not same_thread) or gap > TIME_GAP_SECONDS
            size_boundary = current_len + len(text) > UPPER_CHARS

            if hard_boundary or (size_boundary and current_len >= LOWER_CHARS):
                groups.append(current)
                if hard_boundary or OVERLAP_MESSAGES <= 0:
                    tail: list[tuple[Message, str]] = []
                else:
                    tail = current[-OVERLAP_MESSAGES:]
                current = list(tail)
                current_len = sum(len(t) for _, t in current)
        current.append((m, text))
        current_len += len(text)

    if current:
        groups.append(current)

    chunks: list[IndexAPIItem] = []
    for group in groups:
        if not any(m.id in new_ids for m, _ in group):
            continue
        body = "\n".join(t for _, t in group)
        chunks.append(
            IndexAPIItem(
                page_content=body,
                dense_content=body,
                sparse_content=body,
                message_ids=[m.id for m, _ in group],
            )
        )
    return chunks

# Ваш сервис должен имплементировать оба этих метода
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    return IndexAPIResponse(
        results=build_chunks(
            payload.data.overlap_messages,
            payload.data.new_messages,
        )
    )


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding

    # можете делать любой вектор, который будет совместим с вашим поиском в Qdrant
    # помните об ограничении времени выполнения вашей работы в тестирующей системе
    logger.info(
        "Loading sparse model %s from cache %s",
        SPARSE_MODEL_NAME,
        FASTEMBED_CACHE_PATH,
    )
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    vectors: list[dict[str, list[int] | list[float]]] = []

    for item in model.embed(texts):
        vectors.append(
            {
                "indices": item.indices.tolist(),
                "values": item.values.tolist(),
            }
        )

    return vectors


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:
    # Проверяющая система вызывает этот endpoint при создании коллекции
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}

# красивая обработка ошибок
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=UVICORN_WORKERS,
    )


if __name__ == "__main__":
    main()
