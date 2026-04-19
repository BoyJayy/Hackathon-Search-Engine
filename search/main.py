from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from config import API_KEY, HOST, PORT, QDRANT_URL, logger
from pipeline import run_search_pipeline
from schemas import SearchAPIItem, SearchAPIRequest, SearchAPIResponse


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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    try:
        final_message_ids, _ = await run_search_pipeline(
            app.state.http,
            app.state.qdrant,
            payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not final_message_ids:
        return SearchAPIResponse(results=[])

    return SearchAPIResponse(
        results=[SearchAPIItem(message_ids=final_message_ids)]
    )


@app.post("/_debug/search")
async def search_debug(
    payload: SearchAPIRequest,
    no_rescore: bool = False,
    no_rerank: bool = False,
    fusion: str = "dbsf",
    max_dense: int | None = None,
    max_sparse: int | None = None,
) -> dict[str, Any]:
    try:
        final_message_ids, stages = await run_search_pipeline(
            app.state.http,
            app.state.qdrant,
            payload,
            skip_rescore=no_rescore,
            skip_rerank=no_rerank,
            collect_stages=True,
            fusion=fusion,
            max_dense=max_dense,
            max_sparse=max_sparse,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not final_message_ids:
        return {"final": [], "stages": stages}

    return {"final": final_message_ids, "stages": stages}


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
