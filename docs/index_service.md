# Index Service

Эта doc описывает текущую реализацию `index` после рефактора на модули.

## Назначение

`index` принимает чат и сообщения, строит чанки и подготавливает три текстовых представления:
- `page_content`
- `dense_content`
- `sparse_content`

Также сервис умеет локально считать sparse vectors через `POST /sparse_embedding`.

Важно:
- `index` не строит dense vectors сам;
- `index` только подготавливает `dense_content` для внешней dense embedding модели.

## Текущая структура файлов

- [`index/main.py`](/Users/boyjayy/Documents/Search/index/main.py) — FastAPI app, роуты, обработка ошибок.
- [`index/config.py`](/Users/boyjayy/Documents/Search/index/config.py) — env, константы chunking и логирования.
- [`index/schemas.py`](/Users/boyjayy/Documents/Search/index/schemas.py) — Pydantic-схемы запросов и ответов.
- [`index/chunking.py`](/Users/boyjayy/Documents/Search/index/chunking.py) — вся основная логика индексатора.
- [`index/sparse.py`](/Users/boyjayy/Documents/Search/index/sparse.py) — загрузка sparse-модели и построение sparse vectors.

## Что происходит в `POST /index`

Pipeline:

```text
chat + overlap_messages + new_messages
-> normalize
-> filter
-> split long messages
-> build chunks
-> format page_content / dense_content / sparse_content
-> return results[]
```

## Основные шаги

### 1. Нормализация сообщения

На этом шаге сервис:
- собирает текст из `text`;
- вытаскивает `parts[*].text`;
- добавляет `file_snippets`;
- нормализует переносы строк и пустые строки;
- сохраняет `mentions`, `thread_sn`, `sender_id`, флаги `is_forward`, `is_quote`, `is_system`, `is_hidden`.

Ключевые функции:
- `normalize_text`
- `extract_part_texts`
- `render_message`
- `normalize_message`

### 2. Фильтрация

Сервис отбрасывает:
- `is_hidden=true`;
- пустые сообщения;
- часть системных сообщений без полезного сигнала;
- короткие ack-сообщения вроде чистого `ок`, если у них нет других сигналов.

Ключевая функция:
- `is_message_searchable`

### 3. Обработка длинных сообщений

Текущая логика различает два класса:

- технические простыни и логи;
- длинные обычные сообщения.

#### Технические сообщения

Определяются по:
- длине;
- количеству строк;
- trace-маркерам вроде `sigabrt`, `goroutine`, `runtime.`, `.go:`, `.py:`.

Такие сообщения:
- не дробятся на маленькие части;
- сжимаются через `... [N lines omitted] ...`;
- режутся по разным лимитам для `page`, `dense` и `sparse`.

Ключевые функции:
- `is_technical_message`
- `compress_text_for_index`
- `prepare_text_variant`

#### Длинные обычные сообщения

Если сообщение длинное, но не техническое, оно режется на большие смысловые части:
- сначала по абзацам;
- потом по предложениям;
- потом при необходимости по target size.

В ответе такие части маркируются как `part=1/2`, `part=2/2`.

Ключевые функции:
- `split_long_text`
- `split_message_for_chunking`

### 4. Chunking

Chunking делается не по символам всей склейки, а по сообщениям.

Учитывается:
- контекст из `overlap_messages`;
- временной разрыв между сообщениями;
- лимит на общий размер чанка;
- ограниченный overlap из предыдущих сообщений.

Это позволяет:
- не терять контекст между батчами;
- не смешивать слишком далёкие сообщения;
- собирать связные диалоги в один chunk.

### 5. Формирование трёх текстовых вариантов

#### `page_content`

Назначение:
- payload в Qdrant;
- читаемый текст;
- rerank;
- локальная отладка.

Особенности:
- содержит `CHAT`, `CHAT_TYPE`, `CHAT_ID`;
- содержит блоки `CONTEXT` и `MESSAGES`;
- хранит timestamps, sender и flags;
- может оставлять больше контекста, чем dense.

#### `dense_content`

Назначение:
- текст для dense embedding модели.

Особенности:
- более гладкий и “смысловой” текст;
- меньше служебной структуры;
- без принудительного lower-case;
- сжатые технические сообщения;
- сохраняет смысл разговора, а не только keyword-сигналы.

#### `sparse_content`

Назначение:
- текст для sparse embedding.

Особенности:
- сохраняет точные токены;
- добавляет `sender: ...`;
- включает `mentions`;
- отмечает `forwarded` и `quoted`;
- лучше подходит для keyword retrieval.

## Почему `dense_content` и `sparse_content` отличаются

Dense нужен для поиска по смыслу:
- похожие формулировки;
- семантическая близость;
- общий смысл разговора.

Sparse нужен для точных совпадений:
- имена;
- email;
- ссылки;
- ошибки;
- версии;
- технические токены.

Именно поэтому:
- в dense не хочется лишней служебки;
- в sparse наоборот полезны дополнительные keyword-сигналы.

## Что делает `POST /sparse_embedding`

Endpoint:
- принимает `texts: string[]`;
- загружает локальную sparse-модель `fastembed`;
- возвращает `indices` и `values` в формате, совместимом с Qdrant.

Это используется:
- при индексации chunk'ов;
- при построении sparse query vector в поиске.

## Что `index` сейчас не делает

`index` не:
- считает dense vectors;
- пишет точки в Qdrant;
- выполняет retrieval;
- делает rerank.

То есть сервис отвечает только за подготовку chunk'ов и sparse embeddings.

## Как проверять `index`

Минимальная ручная проверка:

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/index_payload.json
```

Что смотреть глазами:
- есть ли `CONTEXT`;
- различаются ли `page_content`, `dense_content`, `sparse_content`;
- сжимаются ли логи;
- режутся ли длинные посты на `part=...`;
- корректны ли `message_ids`.
