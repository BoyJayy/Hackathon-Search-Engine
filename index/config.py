import logging
import os

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")

MAX_CHUNK_CHARS = 1800
OVERLAP_MESSAGE_COUNT = 2
OVERLAP_CONTEXT_CHARS = 500
MAX_TIME_GAP_SECONDS = 3 * 60 * 60
LONG_MESSAGE_CHAR_THRESHOLD = 1600
LONG_MESSAGE_LINE_THRESHOLD = 35
PAGE_TECHNICAL_MAX_LINES = 24
PAGE_TECHNICAL_MAX_CHARS = 2200
DENSE_TECHNICAL_MAX_LINES = 10
DENSE_TECHNICAL_MAX_CHARS = 900
SPARSE_TECHNICAL_MAX_LINES = 14
SPARSE_TECHNICAL_MAX_CHARS = 1200
SPLIT_MESSAGE_CHAR_THRESHOLD = 1200
SPLIT_SEGMENT_TARGET_CHARS = 700

SHORT_ACK_MESSAGES = {
    "+",
    "++",
    "ага",
    "да",
    "нет",
    "ок",
    "окей",
    "понял",
    "спасибо",
    "ясно",
    "yes",
    "no",
    "ok",
    "thanks",
    "thx",
}

SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"
UVICORN_WORKERS = 8

TECHNICAL_TRACE_MARKERS = (
    "traceback",
    "exception",
    "stack trace",
    "goroutine ",
    "runtime.",
    "pc=",
    "sigabrt",
    "panic:",
    " at ",
    ".go:",
    ".py:",
)
