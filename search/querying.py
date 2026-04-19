import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from config import INTENT_ALIGNMENT_WEIGHT, MAX_DENSE_QUERIES, MAX_SPARSE_QUERIES
from schemas import Entities, Question


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[\w@./:+-]+", re.UNICODE)
SUMMARY_QUERY_STARTS = (
    "где ",
    "в каком документ",
    "в каком файле",
    "where ",
    "which document",
)
DETAIL_QUERY_STARTS = (
    "какие ",
    "какой ",
    "какую ",
    "как ",
    "что ",
    "сколько ",
    "при каком ",
    "до какого ",
    "на каком ",
    "which ",
    "what ",
    "how ",
)
SUMMARY_TEXT_MARKERS = (
    "зафиксировал итог по теме",
    "в документе '",
    "ссылка http",
    "ссылка https",
)


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
        has_special_signal = any(ch.isdigit() for ch in token) or any(
            ch in token for ch in "@./:+-_"
        )
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


def detect_query_intent(question: Question) -> str:
    lowered = normalize_query_text(
        " ".join(part for part in [question.text, question.search_text] if part)
    ).lower()

    if any(lowered.startswith(marker) or marker in lowered for marker in SUMMARY_QUERY_STARTS):
        return "summary"

    if any(lowered.startswith(marker) or marker in lowered for marker in DETAIL_QUERY_STARTS):
        return "detail"

    return "neutral"


def is_summary_like_text(text: str) -> bool:
    lowered = normalize_query_text(text).lower()
    if any(marker in lowered for marker in SUMMARY_TEXT_MARKERS):
        return True
    return "документ" in lowered and ("http" in lowered or "https" in lowered)


@dataclass(frozen=True)
class QueryContext:
    question: Question
    phrase_terms: tuple[str, ...]
    token_terms: tuple[str, ...]
    identity_terms: frozenset[str]
    intent: str
    prefers_earliest: bool
    query_start: int | None
    query_end: int | None


def build_query_context(question: Question) -> QueryContext:
    query_start = query_end = None
    if question.date_range is not None:
        query_start = parse_timestamp(question.date_range.from_)
        query_end = parse_timestamp(question.date_range.to, end_of_day=True)

    return QueryContext(
        question=question,
        phrase_terms=tuple(build_phrase_terms(question)),
        token_terms=tuple(build_query_signal_tokens(question)),
        identity_terms=frozenset(normalize_terms([question.asker, *collect_entity_terms(question.entities)])),
        intent=detect_query_intent(question),
        prefers_earliest=query_prefers_earliest_message(question),
        query_start=query_start,
        query_end=query_end,
    )


def score_intent_alignment(question: Question, text: str) -> float:
    if INTENT_ALIGNMENT_WEIGHT <= 0:
        return 0.0

    intent = detect_query_intent(question)
    lowered = normalize_query_text(text).lower()
    summary_like = is_summary_like_text(lowered)

    if intent == "summary":
        score = 0.12 if summary_like else -0.04
        if "документ" in lowered or "http" in lowered:
            score += 0.02
        return score * INTENT_ALIGNMENT_WEIGHT

    if intent == "detail":
        score = -0.08 if summary_like else 0.08
        if not summary_like and len(lowered) <= 220:
            score += 0.02
        return score * INTENT_ALIGNMENT_WEIGHT

    return 0.0


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
