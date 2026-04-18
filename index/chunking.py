from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

ROOT_THREAD_KEY = "__root__"


class MessageLike(Protocol):
    id: str
    thread_sn: str | None
    text: str
    parts: list[dict[str, Any]] | None
    is_system: bool
    is_hidden: bool


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    overlap_size: int
    long_message_threshold: int


@dataclass(frozen=True)
class ChunkResult:
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]


def render_part(part: dict[str, Any]) -> str:
    part_text = part.get("text")
    if not isinstance(part_text, str):
        return ""

    part_text = part_text.strip()
    if not part_text:
        return ""

    media_type = str(part.get("mediaType") or "").strip().lower()
    prefix_map = {
        "forward": "[forward]",
        "quote": "[quote]",
        "text": "",
    }
    prefix = prefix_map.get(media_type, f"[{media_type}]") if media_type else ""

    if not prefix:
        return part_text

    return f"{prefix}\n{part_text}"


def render_message(message: MessageLike) -> str:
    chunks: list[str] = []

    message_text = message.text.strip()
    if message_text:
        chunks.append(message_text)

    if message.parts:
        for part in message.parts:
            rendered_part = render_part(part)
            if rendered_part:
                chunks.append(rendered_part)

    return "\n\n".join(chunks).strip()


def get_thread_key(message: MessageLike) -> str:
    return message.thread_sn or ROOT_THREAD_KEY


def should_index_message(message: MessageLike) -> bool:
    return not message.is_hidden and not message.is_system


def collect_rendered_messages(
    messages: list[MessageLike],
) -> dict[str, list[tuple[MessageLike, str]]]:
    rendered_by_thread: dict[str, list[tuple[MessageLike, str]]] = defaultdict(list)

    for message in messages:
        if not should_index_message(message):
            continue

        rendered_text = render_message(message)
        if not rendered_text:
            continue

        rendered_by_thread[get_thread_key(message)].append((message, rendered_text))

    return dict(rendered_by_thread)


def render_chunk(messages: list[tuple[MessageLike, str]]) -> str:
    return "\n\n".join(text for _, text in messages)


def is_long_message(message_text: str, threshold: int) -> bool:
    return len(message_text) >= threshold


def build_thread_chunks(
    overlap_messages: list[tuple[MessageLike, str]],
    new_messages: list[tuple[MessageLike, str]],
    config: ChunkingConfig,
) -> list[ChunkResult]:
    thread_chunks: list[ChunkResult] = []
    chunk_context = overlap_messages[-config.overlap_size :] if config.overlap_size else []
    pending_messages = list(new_messages)

    while pending_messages:
        chunk_body: list[tuple[MessageLike, str]] = []

        while pending_messages and len(chunk_body) < config.chunk_size:
            next_message = pending_messages[0]
            _, next_text = next_message

            if chunk_body and is_long_message(next_text, config.long_message_threshold):
                break

            chunk_body.append(pending_messages.pop(0))

            if is_long_message(next_text, config.long_message_threshold):
                break

        chunk_messages = [*chunk_context, *chunk_body]
        chunk_text = render_chunk(chunk_messages)

        thread_chunks.append(
            ChunkResult(
                page_content=chunk_text,
                dense_content=chunk_text,
                sparse_content=chunk_text,
                message_ids=[message.id for message, _ in chunk_body],
            )
        )

        if config.overlap_size:
            chunk_context = chunk_messages[-config.overlap_size :]

    return thread_chunks


def build_chunks(
    overlap_messages: list[MessageLike],
    new_messages: list[MessageLike],
    config: ChunkingConfig,
) -> list[ChunkResult]:
    result: list[ChunkResult] = []
    overlap_by_thread = collect_rendered_messages(overlap_messages)
    new_by_thread = collect_rendered_messages(new_messages)

    for thread_key, thread_new_messages in new_by_thread.items():
        thread_overlap_messages = overlap_by_thread.get(thread_key, [])
        result.extend(build_thread_chunks(thread_overlap_messages, thread_new_messages, config))

    return result
