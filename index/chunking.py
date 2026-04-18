from dataclasses import dataclass, replace
from datetime import UTC, datetime
import re

from config import (
    DENSE_TECHNICAL_MAX_CHARS,
    DENSE_TECHNICAL_MAX_LINES,
    LONG_MESSAGE_CHAR_THRESHOLD,
    LONG_MESSAGE_LINE_THRESHOLD,
    MAX_CHUNK_CHARS,
    MAX_TIME_GAP_SECONDS,
    OVERLAP_CONTEXT_CHARS,
    OVERLAP_MESSAGE_COUNT,
    PAGE_TECHNICAL_MAX_CHARS,
    PAGE_TECHNICAL_MAX_LINES,
    SHORT_ACK_MESSAGES,
    SPARSE_TECHNICAL_MAX_CHARS,
    SPARSE_TECHNICAL_MAX_LINES,
    SPLIT_MESSAGE_CHAR_THRESHOLD,
    SPLIT_SEGMENT_TARGET_CHARS,
    TECHNICAL_TRACE_MARKERS,
    logger,
)
from schemas import Chat, IndexAPIItem, Message


@dataclass(slots=True)
class NormalizedMessage:
    id: str
    time: int
    sender_id: str
    thread_sn: str | None
    text: str
    file_snippets: str
    mentions: list[str]
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool
    is_overlap: bool
    fragment_index: int = 1
    fragment_count: int = 1


def normalize_text(text: str) -> str:
    stripped_lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in stripped_lines if line]
    return "\n".join(non_empty_lines).strip()


def join_text_parts(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part).strip()


def extract_part_texts(message: Message) -> list[str]:
    parts_text: list[str] = []
    for part in message.parts or []:
        part_text = part.get("text")
        if isinstance(part_text, str):
            normalized = normalize_text(part_text)
            if normalized:
                parts_text.append(normalized)
    return parts_text


def render_message(message: Message) -> str:
    content_parts: list[str] = []

    normalized_text = normalize_text(message.text)
    if normalized_text:
        content_parts.append(normalized_text)

    part_texts = extract_part_texts(message)
    if part_texts:
        content_parts.extend(part_texts)

    normalized_snippets = normalize_text(message.file_snippets)
    if normalized_snippets:
        content_parts.append(normalized_snippets)

    return join_text_parts(content_parts)


def is_technical_message(text: str) -> bool:
    lowered = text.lower()
    line_count = len(text.splitlines())
    if len(text) >= LONG_MESSAGE_CHAR_THRESHOLD or line_count >= LONG_MESSAGE_LINE_THRESHOLD:
        return True
    return any(marker in lowered for marker in TECHNICAL_TRACE_MARKERS)


def compress_text_for_index(
    text: str,
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    lines = normalized.splitlines()
    original_line_count = len(lines)
    if len(lines) > max_lines:
        head_lines = max(1, int(max_lines * 0.7))
        tail_lines = max(1, max_lines - head_lines - 1)
        shortened_lines = lines[:head_lines]
        omitted = max(0, original_line_count - head_lines - tail_lines)
        if omitted:
            shortened_lines.append(f"... [{omitted} lines omitted] ...")
        if tail_lines > 0:
            shortened_lines.extend(lines[-tail_lines:])
        lines = shortened_lines

    compressed = "\n".join(lines)
    if len(compressed) > max_chars:
        suffix = f"\n... [truncated {len(compressed) - max_chars} chars]"
        compressed = compressed[: max_chars - len(suffix)].rstrip()
        compressed += suffix

    return compressed.strip()


def prepare_text_variant(
    text: str,
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    if not is_technical_message(normalized):
        return normalized
    return compress_text_for_index(
        normalized,
        max_lines=max_lines,
        max_chars=max_chars,
    )


def split_long_text(text: str, *, target_chars: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    primary_segments = [part.strip() for part in normalized.split("\n") if part.strip()]
    atomic_parts: list[str] = []
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    for segment in primary_segments:
        if len(segment) <= target_chars:
            atomic_parts.append(segment)
            continue

        sentence_parts = [part.strip() for part in sentence_pattern.split(segment) if part.strip()]
        if len(sentence_parts) <= 1:
            sentence_parts = [segment[i : i + target_chars] for i in range(0, len(segment), target_chars)]

        current_parts: list[str] = []
        current_len = 0
        for sentence in sentence_parts:
            sentence_len = len(sentence)
            if current_parts and current_len + sentence_len + 1 > target_chars:
                atomic_parts.append(" ".join(current_parts).strip())
                current_parts = [sentence]
                current_len = sentence_len
            else:
                current_parts.append(sentence)
                current_len += sentence_len + (1 if current_parts[:-1] else 0)

        if current_parts:
            atomic_parts.append(" ".join(current_parts).strip())

    passages: list[str] = []
    current_parts = []
    current_len = 0
    for part in atomic_parts:
        part_len = len(part)
        separator_len = 2 if current_parts else 0
        if current_parts and current_len + separator_len + part_len > target_chars:
            passages.append("\n".join(current_parts).strip())
            current_parts = [part]
            current_len = part_len
        else:
            current_parts.append(part)
            current_len += part_len + separator_len

    if current_parts:
        passages.append("\n".join(current_parts).strip())

    return [passage for passage in passages if passage]


def normalize_message(message: Message, *, is_overlap: bool) -> NormalizedMessage:
    return NormalizedMessage(
        id=message.id,
        time=message.time,
        sender_id=message.sender_id,
        thread_sn=message.thread_sn,
        text=render_message(message),
        file_snippets=normalize_text(message.file_snippets),
        mentions=[mention.strip() for mention in (message.mentions or []) if mention.strip()],
        is_system=message.is_system,
        is_hidden=message.is_hidden,
        is_forward=message.is_forward,
        is_quote=message.is_quote,
        is_overlap=is_overlap,
    )


def split_message_for_chunking(message: NormalizedMessage) -> list[NormalizedMessage]:
    if is_technical_message(message.text):
        return [message]

    if len(message.text) <= SPLIT_MESSAGE_CHAR_THRESHOLD:
        return [message]

    segments = split_long_text(message.text, target_chars=SPLIT_SEGMENT_TARGET_CHARS)
    if len(segments) <= 1:
        return [message]

    return [
        replace(
            message,
            text=segment,
            fragment_index=index,
            fragment_count=len(segments),
        )
        for index, segment in enumerate(segments, start=1)
    ]


def is_message_searchable(message: NormalizedMessage) -> bool:
    if message.is_hidden:
        return False

    has_signal = bool(message.text or message.file_snippets or message.mentions)
    if not has_signal:
        return False

    if message.is_system and not (message.file_snippets or message.mentions):
        return False

    compact_text = " ".join(message.text.lower().split())
    if (
        compact_text in SHORT_ACK_MESSAGES
        and not message.file_snippets
        and not message.mentions
        and not message.is_forward
        and not message.is_quote
    ):
        return False

    return True


def format_timestamp(unix_time: int) -> str:
    return datetime.fromtimestamp(unix_time, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_page_message(message: NormalizedMessage) -> str:
    flags: list[str] = []
    if message.thread_sn:
        flags.append(f"thread={message.thread_sn}")
    if message.is_forward:
        flags.append("forward")
    if message.is_quote:
        flags.append("quote")
    if message.is_system:
        flags.append("system")
    if message.fragment_count > 1:
        flags.append(f"part={message.fragment_index}/{message.fragment_count}")

    header = f"[{format_timestamp(message.time)} | {message.sender_id}"
    if flags:
        header += f" | {', '.join(flags)}"
    header += "]"

    body_text = prepare_text_variant(
        message.text,
        max_lines=PAGE_TECHNICAL_MAX_LINES,
        max_chars=PAGE_TECHNICAL_MAX_CHARS,
    )
    body_parts = [body_text] if body_text else []
    if message.mentions:
        body_parts.append(f"Mentions: {', '.join(message.mentions)}")

    return f"{header}\n{join_text_parts(body_parts)}".strip()


def format_dense_message(message: NormalizedMessage) -> str:
    dense_text = prepare_text_variant(
        message.text,
        max_lines=DENSE_TECHNICAL_MAX_LINES,
        max_chars=DENSE_TECHNICAL_MAX_CHARS,
    )
    parts = [dense_text] if dense_text else []
    if message.mentions:
        parts.append(f"mentions: {', '.join(message.mentions)}")
    return join_text_parts(parts)


def format_sparse_message(message: NormalizedMessage) -> str:
    sparse_text = prepare_text_variant(
        message.text,
        max_lines=SPARSE_TECHNICAL_MAX_LINES,
        max_chars=SPARSE_TECHNICAL_MAX_CHARS,
    )
    parts = [sparse_text] if sparse_text else []
    parts.append(f"sender: {message.sender_id}")
    if message.mentions:
        parts.append(" ".join(message.mentions))
    if message.is_forward:
        parts.append("forwarded")
    if message.is_quote:
        parts.append("quoted")
    return join_text_parts(parts)


def estimate_page_message_size(message: NormalizedMessage) -> int:
    return len(format_page_message(message)) + 2


def build_page_content(
    chat: Chat,
    context_messages: list[NormalizedMessage],
    chunk_messages: list[NormalizedMessage],
) -> str:
    sections = [
        f"CHAT: {chat.name}",
        f"CHAT_TYPE: {chat.type}",
        f"CHAT_ID: {chat.id}",
    ]

    if context_messages:
        sections.append("CONTEXT:")
        sections.extend(format_page_message(message) for message in context_messages)

    sections.append("MESSAGES:")
    sections.extend(format_page_message(message) for message in chunk_messages)
    return "\n\n".join(section for section in sections if section).strip()


def build_dense_content(
    chat: Chat,
    context_messages: list[NormalizedMessage],
    chunk_messages: list[NormalizedMessage],
) -> str:
    formatted_messages = [
        formatted
        for message in [*context_messages, *chunk_messages]
        if (formatted := format_dense_message(message))
    ]
    return "\n\n".join([f"chat {chat.name}", f"chat_type {chat.type}", *formatted_messages]).strip()


def build_sparse_content(
    chat: Chat,
    context_messages: list[NormalizedMessage],
    chunk_messages: list[NormalizedMessage],
) -> str:
    parts = [chat.name, chat.type, chat.id]
    for message in [*context_messages, *chunk_messages]:
        sparse_message = format_sparse_message(message)
        if sparse_message:
            parts.append(sparse_message)
    return "\n".join(parts).strip()


def should_flush_chunk(
    current_chunk: list[NormalizedMessage],
    next_message: NormalizedMessage,
    current_size: int,
) -> bool:
    if not current_chunk:
        return False

    previous_message = current_chunk[-1]
    if (
        previous_message.thread_sn
        and next_message.thread_sn
        and previous_message.thread_sn != next_message.thread_sn
    ):
        return True

    if next_message.time - previous_message.time > MAX_TIME_GAP_SECONDS:
        return True

    return current_size + estimate_page_message_size(next_message) > MAX_CHUNK_CHARS


def select_overlap_context(messages: list[NormalizedMessage]) -> list[NormalizedMessage]:
    context: list[NormalizedMessage] = []
    total_chars = 0

    for message in reversed(messages):
        message_size = estimate_page_message_size(message)
        if context and (
            len(context) >= OVERLAP_MESSAGE_COUNT
            or total_chars + message_size > OVERLAP_CONTEXT_CHARS
        ):
            break
        context.append(message)
        total_chars += message_size

    context.reverse()
    return context


def build_chunk_item(
    chat: Chat,
    context_messages: list[NormalizedMessage],
    chunk_messages: list[NormalizedMessage],
) -> IndexAPIItem:
    ordered_message_ids: list[str] = []
    for message in chunk_messages:
        if message.id not in ordered_message_ids:
            ordered_message_ids.append(message.id)

    return IndexAPIItem(
        page_content=build_page_content(chat, context_messages, chunk_messages),
        dense_content=build_dense_content(chat, context_messages, chunk_messages),
        sparse_content=build_sparse_content(chat, context_messages, chunk_messages),
        message_ids=ordered_message_ids,
    )


def log_chunk_diagnostics(chunks: list[IndexAPIItem]) -> None:
    if not chunks:
        logger.info("Chunk diagnostics: no chunks produced")
        return

    page_lengths = [len(chunk.page_content) for chunk in chunks]
    dense_lengths = [len(chunk.dense_content) for chunk in chunks]
    sparse_lengths = [len(chunk.sparse_content) for chunk in chunks]
    covered_messages = sum(len(chunk.message_ids) for chunk in chunks)
    sample_ids = chunks[0].message_ids[:5]

    logger.info(
        (
            "Chunk diagnostics: count=%s, covered_messages=%s, "
            "avg_page=%s, max_page=%s, avg_dense=%s, avg_sparse=%s, sample_ids=%s"
        ),
        len(chunks),
        covered_messages,
        sum(page_lengths) // len(page_lengths),
        max(page_lengths),
        sum(dense_lengths) // len(dense_lengths),
        sum(sparse_lengths) // len(sparse_lengths),
        sample_ids,
    )


def build_chunks(
    chat: Chat,
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    normalized_overlap = [
        normalize_message(message, is_overlap=True)
        for message in overlap_messages
    ]
    normalized_new = [
        normalize_message(message, is_overlap=False)
        for message in new_messages
    ]

    searchable_overlap = [
        message
        for message in sorted(normalized_overlap, key=lambda item: (item.time, item.id))
        if is_message_searchable(message)
    ]
    searchable_new = [
        message
        for message in sorted(normalized_new, key=lambda item: (item.time, item.id))
        if is_message_searchable(message)
    ]

    if not searchable_new:
        logger.info("Index build produced no searchable new messages")
        return []

    chunkable_overlap = [
        segment
        for message in searchable_overlap
        for segment in split_message_for_chunking(message)
    ]
    chunkable_new = [
        segment
        for message in searchable_new
        for segment in split_message_for_chunking(message)
    ]

    result: list[IndexAPIItem] = []
    current_context = select_overlap_context(chunkable_overlap)
    current_chunk: list[NormalizedMessage] = []
    current_size = sum(estimate_page_message_size(message) for message in current_context)

    for message in chunkable_new:
        if should_flush_chunk(current_chunk, message, current_size):
            result.append(build_chunk_item(chat, current_context, current_chunk))
            current_context = select_overlap_context([*current_context, *current_chunk])
            current_chunk = []
            current_size = sum(estimate_page_message_size(item) for item in current_context)

        current_chunk.append(message)
        current_size += estimate_page_message_size(message)

    if current_chunk:
        result.append(build_chunk_item(chat, current_context, current_chunk))

    logger.info(
        "Built %s chunks from %s overlap and %s new messages (%s searchable new)",
        len(result),
        len(overlap_messages),
        len(new_messages),
        len(searchable_new),
    )
    log_chunk_diagnostics(result)

    return result
