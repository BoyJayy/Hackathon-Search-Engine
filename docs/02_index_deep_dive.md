# 02. Index Service - разобраться до последней функции

---

## 2.1 Зачем нужен index-service

Chat не влезет в LLM-контекст одним куском. Нужно нарезать его на чанки -
смысловые группы сообщений. Затем для каждого чанка построить:

- `page_content` - «человеческий» текст с шапкой чата + CONTEXT + MESSAGES, для rerank.
- `dense_content` - гладкий текст для embedding-модели.
- `sparse_content` - keyword-rich текст для BM25.
- `message_ids` - какие сообщения покрыл этот чанк (важно для метрики!).

Также index-service отдаёт `POST /sparse_embedding` - отдельный endpoint,
который локально (без интернета) считает BM25-векторы через `fastembed`.

## 2.2 Контракт (нельзя менять)

Подробно в [api.md](../api.md). Если в двух словах:

- `POST /index` ← `{data: {chat, overlap_messages, new_messages}}` →
  `{results: [{page_content, dense_content, sparse_content, message_ids}]}`
- `POST /sparse_embedding` ← `{texts: [...]}` →
  `{vectors: [{indices: [...], values: [...]}]}` (Qdrant-совместимый формат).

## 2.3 Структура модулей

| Файл              | Что делает                                                       |
| ----------------- | ---------------------------------------------------------------- |
| `main.py`         | FastAPI app, роуты, exception handler                            |
| `schemas.py`      | Pydantic-модели запросов/ответов                                 |
| `config.py`       | Все env-параметры chunking'a                                     |
| `chunking.py`     | Вся смысловая логика: normalize → filter → split → chunk → format |
| `sparse.py`       | Ленивая загрузка `SparseTextEmbedding(model_name="Qdrant/bm25")` |

## 2.4 Pipeline внутри `/index` - пошагово

```
POST /index payload
  │
  ▼
normalize_message()          # собираем text + parts + file_snippets + member_event
  │
  ▼
is_message_searchable()      # фильтр: hidden, ack, empty, часть system-сообщений
  │
  ▼
split_message_for_chunking() # длинные не-технические сообщения → несколько fragments
  │
  ▼
should_flush_chunk()         # решение: включить msg в текущий chunk или начать новый
  │ (thread_sn / time_gap / size)
  ▼
build_chunk_item()           # формируем page / dense / sparse + message_ids
  │
  ▼
results: list[IndexAPIItem]
```

## 2.5 Шаг 1 - `normalize_message` (строки 229-244)

Сборка **единого текста** из разрозненных полей сообщения:

- `message.text` - основной текст
- `message.parts[].text` - rich-части (mediaType=`quote` маркируется отдельно)
- `render_member_event()` - `"User X added members: A, B"`
- `message.file_snippets` - содержимое вложений

Нюанс: `mediaType == "quote"` превращается в блок:

```
Quoted message:
<текст цитаты>
```

Зачем: чтобы при ранжировании search-service мог отличать свою реплику
от процитированной (иначе ответ с цитатой обгоняет оригинал).

## 2.6 Шаг 2 - `is_message_searchable` (строки 268-289)

Выкидываем шум:

1. `is_hidden=True` → вон.
2. Нет ни `text`, ни `file_snippets`, ни `mentions` → вон.
3. Системное и нет полезного сигнала → вон (но `addMembers` сохраняется).
4. `"ок"`, `"+"`, `"спасибо"` и пр. ack без `file_snippets`/`mentions`/forward/quote → вон.

Список ack в [index/config.py `SHORT_ACK_MESSAGES`](../index/config.py#L25-L41).

## 2.7 Шаг 3 - `split_long_text` + `split_message_for_chunking` (строки 176-265)

Зачем два класса обработки длинных сообщений:

- Технические (traceback, 35+ строк, маркеры `sigabrt`, `.ts:`, `.py:`):
  не дробим, сжимаем через `compress_text_for_index` с `... [N lines omitted] ...`.
- Длинные обычные (≥ `SPLIT_MESSAGE_CHAR_THRESHOLD=1200` chars):
  режем на фрагменты `part=1/N`, `part=2/N` по абзацам → предложениям →
  по target size. Все фрагменты получают одинаковый `message.id`,
  чтобы метрика работала корректно.

Разные лимиты для разных вариантов текста в [config.py](../index/config.py):

| Variant  | `MAX_LINES`                   | `MAX_CHARS`                  |
| -------- | ----------------------------- | ---------------------------- |
| page     | `PAGE_TECHNICAL_MAX_LINES=24` | `PAGE_TECHNICAL_MAX_CHARS=2200` |
| dense    | `DENSE_TECHNICAL_MAX_LINES=10`| `DENSE_TECHNICAL_MAX_CHARS=900`  |
| sparse   | `SPARSE_TECHNICAL_MAX_LINES=14`| `SPARSE_TECHNICAL_MAX_CHARS=1200` |

Почему sparse строже, чем page, но мягче dense: sparse должен сохранить токены
(имена, версии, ошибки), но слишком длинный текст размазывает BM25.

## 2.8 Шаг 4 - Chunking (`build_chunks`, строки 551-630)

Сердце сервиса. Алгоритм:

1. Сортируем `overlap_messages` и `new_messages` по `(time, id)`.
2. Отбрасываем нефильтрующиеся.
3. Раскладываем длинные non-technical сообщения на фрагменты.
4. Для первого чанка:
   - если сообщение выглядит как новая top-level тема (`starts_new_topic`,
     строки 427-450) - сбрасываем overlap полностью.
   - иначе - берём `select_thread_aware_overlap_context` (последние сообщения
     того же `thread_sn`, что и первое new-сообщение).
5. Итерируемся по new-сообщениям; для каждого проверяем `should_flush_chunk`:
   - другой `thread_sn` у соседних сообщений → flush.
   - time_gap > `MAX_TIME_GAP_SECONDS` (3h) → flush.
   - size + msg > `MAX_CHUNK_CHARS` (1800) → flush.
6. На flush: сохраняем `build_chunk_item(...)`, пересчитываем overlap
   для следующего чанка (на основе `current_context + current_chunk`).

### Почему так, а не просто character splitter

- По сообщениям, потому что `message_ids` - первичная единица оценки.
- Overlap thread-aware, потому что нет смысла тянуть overlap из другого треда.
- Time-gap cut-off, потому что 3 часа тишины - это почти всегда новая тема.
- Topic reset, потому что «Всем привет, у нас вопрос по X...» - явный сигнал
  начать чистый чанк.

## 2.9 Шаг 5 - Три форматера

### `build_page_content` (строки 359-376)

Человекочитаемый payload для Qdrant и rerank.

```
CHAT: <name>
CHAT_TYPE: <type>
CHAT_ID: <id>

CONTEXT:
[2024-05-03 12:00:00 UTC | user@x | thread=T1]
Hello, previous discussion...

[2024-05-03 12:01:00 UTC | user@y | thread=T1]
...

MESSAGES:
[2024-05-03 12:05:00 UTC | user@z | thread=T1, quote]
Current answer text
Mentions: @person
```

`MESSAGES:` идёт ПОСЛЕ `CONTEXT:`. При rerank search-service перекладывает
эти секции в порядке `MESSAGES → CONTEXT`, чтобы reranker сначала увидел ответ,
а не overlap.

### `build_dense_content` (строки 379-389)

```
chat <name>
chat_type <type>

<dense-сжатый текст msg 1>

<dense-сжатый текст msg 2>
...
```

Меньше служебки, без timestamp'ов, без `[...]`-заголовков - smooth текст для
embedding-модели.

### `build_sparse_content` (строки 392-402)

```
<chat.name>
<chat.type>
<chat.id>
<sparse-сжатый text msg 1>
sender: <user_id>
<mentions tokens>
forwarded
quoted
...
```

Одной строкой на сообщение, с явными sender/mentions и флагами - всё, что
поможет BM25 зацепить редкие токены.

## 2.10 `log_chunk_diagnostics` (строки 515-548)

На каждый `/index` в лог пишется:

```
Chunk diagnostics: count=N, unique_messages=M, dup_ratio=1.12x,
avg_page=..., max_page=..., avg_dense=..., avg_sparse=..., sample_ids=[...]
```

Зачем:
- `dup_ratio` показывает, насколько overlap удваивает покрытие. Норма 1.1-1.3x.
- `max_page` - контроль, что `MAX_CHUNK_CHARS` реально соблюдается.
- `sample_ids` - можно найти конкретный чанк в payload для отладки.

## 2.11 Sparse embeddings (`index/sparse.py`)

```python
@lru_cache(maxsize=1)
def get_sparse_model():
    return SparseTextEmbedding(model_name="Qdrant/bm25")
```

Модель загружается один раз, лежит в `/models/fastembed` (pre-baked в Dockerfile
через `RUN python -c "from fastembed import ...; SparseTextEmbedding(...)"`).

На `/sparse_embedding`:
- батч текстов → `model.embed(texts)` → `[{indices, values}, ...]`.
- Используется как на ingestion (чанки), так и на query-side в search-service.

Формат индексов/значений напрямую совместим с `qdrant_client.models.SparseVector`.

## 2.12 Env-параметры (все в `index/config.py`)

| Env                              | Default | Что контролирует                                    |
| -------------------------------- | ------- | --------------------------------------------------- |
| `HOST`/`PORT`                    | 0.0.0.0 / 8004 | сетевые параметры                            |
| `MAX_CHUNK_CHARS`                | 1800    | верхний предел на `page_content`                   |
| `OVERLAP_MESSAGE_COUNT`          | 2       | сколько сообщений брать в CONTEXT максимум         |
| `OVERLAP_CONTEXT_CHARS`          | 500     | макс. размер CONTEXT-секции                        |
| `MAX_TIME_GAP_SECONDS`           | 10800   | 3 часа - граница «новая тема»                      |
| `LONG_MESSAGE_CHAR_THRESHOLD`    | 1600    | когда сообщение считается «техническим» по длине   |
| `SPLIT_MESSAGE_CHAR_THRESHOLD`   | 1200    | длиннее этого - режем на `part=1/N`                |
| `SPLIT_SEGMENT_TARGET_CHARS`     | 700     | target размер фрагмента                            |
| `SHORT_ACK_MESSAGES`             | hardcoded | «да/нет/ок» - не индексируем                      |

Эти параметры можно гонять через `scripts/sweep_chunking.py` без пересборки
образа - они читаются через `os.getenv` на старте.

## 2.13 Как вручную продиагностировать чанкер

```bash
python3 scripts/chunking_diagnostic.py
python3 scripts/chunking_diagnostic.py data/dataset_ts_chat.json
```

Покажет:
- распределение размеров `page/dense/sparse`;
- histogram по messages-per-chunk;
- coverage (все ли searchable сообщения попали в индекс);
- `dup_ratio`;
- preview первых 80 символов каждого чанка.

## 2.14 Что может пойти не так - и как это заметить

| Симптом                                     | Что смотреть                              |
| ------------------------------------------- | ----------------------------------------- |
| `Recall@50` низкий → сообщений нет в индексе | `log_chunk_diagnostics`, `chunking_diagnostic.py` - coverage, uncovered sample |
| `dup_ratio > 2.0x`                          | overlap слишком длинный, понизить `OVERLAP_MESSAGE_COUNT` |
| Reranker падает с timeout                   | `max_page` больше 3000 → уменьшить `MAX_CHUNK_CHARS` или `PAGE_TECHNICAL_MAX_CHARS` |
| Сообщение разделено на фрагменты, но не находится | Проверить, что все фрагменты получают один `message.id` в `split_message_for_chunking` |
| Сообщение из другого треда подтянулось в CONTEXT | `select_thread_aware_overlap_context` - там стоит barrier по `thread_sn` |

---

## 2.15 Worked example - проследим 3 сообщения от JSON до chunk'а

Проследим один маленький input от начала до конца.

### Input (`POST /index`)

```json
{
  "data": {
    "chat": {"chat_id": "42", "name": "Atlas Hub"},
    "overlap_messages": [
      {"id": "10", "thread_sn": "t1", "sender_id": "alice",
       "timestamp": 1713460000, "text": "Кто-нибудь пробовал desktop client 3.4 на M1?"}
    ],
    "new_messages": [
      {"id": "11", "thread_sn": "t1", "sender_id": "bob",
       "timestamp": 1713460060,
       "text": "Да, у меня MacBook Air (M1, 2020) - desktop client падает с SIGABRT при старте."},
      {"id": "12", "thread_sn": "t1", "sender_id": "alice",
       "timestamp": 1713460120,
       "text": "А можешь показать stacktrace?",
       "quoted": {"sender_id": "bob", "text": "падает с SIGABRT при старте"}},
      {"id": "13", "thread_sn": "t1", "sender_id": "bob",
       "timestamp": 1713460180,
       "text": "Traceback (most recent call last):\nFile \"/app/main.py\", line 42\nRuntimeError: init failed"}
    ]
  }
}
```

### Что происходит пошагово

Шаг 1. `normalize_text` - strip whitespace, удаление пустых строк.
Сообщение #13 (stacktrace) проходит `compress_text_for_index` →
детектируется `Traceback`, `RuntimeError`, помечается как technical,
не будет дробиться на фрагменты.

Шаг 2. `filter_searchable` - все четыре сообщения имеют текст и не являются
системными → пропускаются дальше.

Шаг 3. `split_message_for_chunking` - сообщения короткие (<800 символов),
дробление не нужно. Длинное сообщение #13 (technical) тоже не дробится.

Шаг 4. `build_chunks` - все 3 new_messages + 1 overlap помещаются в один chunk
(суммарно ~250 символов, лимит `MAX_CHUNK_CHARS=1800`). Thread_sn у всех `t1` -
barrier не срабатывает.

Шаг 5. `format_page_content` формирует:

```text
CHAT: Atlas Hub
CONTEXT:
[2024-04-18 10:26:40] alice: Кто-нибудь пробовал desktop client 3.4 на M1?

MESSAGES:
[2024-04-18 10:27:40] bob: Да, у меня MacBook Air (M1, 2020) - desktop client падает с SIGABRT при старте.
[2024-04-18 10:28:40] alice: А можешь показать stacktrace?
  Quoted message: падает с SIGABRT при старте
[2024-04-18 10:29:40] bob: Traceback (most recent call last): File "/app/main.py", line 42...
```

Шаг 6. `format_dense_content` - гладкий текст для embedding:

```text
Кто-нибудь пробовал desktop client 3.4 на M1? Да, у меня MacBook Air (M1, 2020) -
desktop client падает с SIGABRT при старте. А можешь показать stacktrace? Traceback
(most recent call last)...
```

Шаг 7. `format_sparse_content` - keyword-rich с `sender:` префиксами:

```text
alice: Кто-нибудь пробовал desktop client 3.4 на M1?
bob: Да у меня MacBook Air M1 2020 desktop client падает SIGABRT при старте
alice: А можешь показать stacktrace? quoted: падает SIGABRT при старте
bob: traceback file app main py runtimeerror init failed
```

### Output (`POST /index` response)

```json
{
  "results": [
    {
      "page_content": "CHAT: Atlas Hub\nCONTEXT:\n[...]alice: Кто-нибудь пробовал desktop client 3.4 на M1?\n\nMESSAGES:\n[...]",
      "dense_content": "Кто-нибудь пробовал desktop client 3.4 на M1? Да, у меня MacBook Air (M1, 2020)...",
      "sparse_content": "alice: Кто-нибудь пробовал desktop client 3.4 на M1?\nbob: ...",
      "message_ids": ["11", "12", "13"],
      "metadata": {
        "chat_id": "42", "chat_name": "Atlas Hub",
        "thread_sn": "t1",
        "start": 1713460060, "end": 1713460180,
        "participants": ["alice", "bob"],
        "mentions": [],
        "contains_quote": true, "contains_forward": false
      }
    }
  ]
}
```

Наблюдения:

- `message_ids = ["11","12","13"]` - только `new_messages`, id=`10` из overlap НЕ включён (он идёт в CONTEXT, но не в ground truth).
- `contains_quote=true` - rerank/rescore учтут, что в chunk'е есть цитата.
- Stacktrace не разбился на фрагменты - technical detection сохранил его целым.
- Три текста, один `message_ids` - один chunk → одна точка в Qdrant (uuid5 от `chat_id + message_ids[0] + start`), с dense+sparse вектором.

---

## 2.16 Приложение: API reference - все функции index-сервиса

Карта функций, которые реально исполняются в рантайме. Helpers, обёртки и все функции chunking pipeline'а - по одной строке.

### `main.py` - FastAPI app и роуты

| Функция                  | Сигнатура                                    | Что делает |
| ------------------------ | -------------------------------------------- | --------- |
| `health`                 | `GET /health → {"status": "ok"}`             | Health-check endpoint, возвращает 200 OK для liveness-проверок тестовой системы. |
| `index`                  | `POST /index(payload: IndexAPIRequest)`      | Основной endpoint: принимает `{chat, overlap_messages, new_messages}`, вызывает `build_chunks`, возвращает `IndexAPIResponse` со списком chunks. |
| `sparse_embedding`       | `POST /sparse_embedding(payload)`            | Отдаёт `embed_sparse_texts(payload.texts)` - локальные BM25-векторы для query и index. Вызывается тестовой системой на обеих фазах. |
| `exception_handler`      | `FastAPI exception_handler`                  | Глобальный хендлер: ловит любое необработанное исключение, логирует с traceback, возвращает 500 с `{"detail": "..."}` вместо HTML-страницы. |
| `main`                   | `if __name__ == "__main__"`                  | Точка входа uvicorn: читает `HOST`/`PORT` из env и запускает FastAPI. |

### `chunking.py` - нормализация, фильтрация, сборка chunk'ов

| Функция                                | Что делает |
| -------------------------------------- | ---------- |
| `normalize_text(text)`                 | `strip()` по каждой строке + убирает пустые строки. Используется везде, где нужен чистый текст для сравнения/записи. |
| `join_text_parts(parts)`               | Склеивает непустые куски через `\n\n` и стрипит результат. Общий helper для `format_page/dense/sparse_message`. |
| `extract_part_texts(message)`          | Разбирает `message.parts` на `(direct_parts, quoted_parts)` по `mediaType=="quote"`. Источник данных для `is_quote` флага и quote-aware разметки. |
| `render_member_event(message)`         | Превращает `member_event` (join/leave/rename) в одну читаемую строку, если сообщение системное. |
| `render_message(message)`              | Собирает итоговый `text` из `message.text`, parts, file_snippets, member_event - т.е. полностью текстовое представление сообщения. |
| `normalize_message(message, chat)`     | Превращает сырой `Message` из `IndexAPIRequest` в `NormalizedMessage` со всеми флагами, готовым `text`, `time`, `sender_id`. Это вход для всего pipeline'а дальше. |
| `is_technical_message(text)`           | Эвристика: есть ли в тексте stacktrace / JSON / URL с path / SQL / длинные code-like токены. Если да - не разбиваем на фрагменты и не режем агрессивно по длине. |
| `is_message_searchable(message)`       | Фильтр: выкидываем hidden/empty/ack-реакции/служебные system-сообщения. Только оставшиеся идут в chunks. |
| `starts_new_topic(message)`            | Эвристика «новое сообщение явно стартует новую тему» (greeting + вопрос, длинный первый блок). Влияет на `should_flush_chunk`. |
| `compress_text_for_index(text, ...)`   | Обрезает длинные сообщения по `max_lines/max_chars` с сохранением головы и хвоста. Для dense/sparse варианта, чтобы не раздувать embedding. |
| `prepare_text_variant(text, ...)`      | Обёртка над `compress_text_for_index` с технической детекцией - technical-сообщения сжимаются мягче. |
| `format_timestamp(unix_time)`          | Форматирует `int` timestamp в строку `YYYY-MM-DD HH:MM:SS UTC`. Используется в page-header для человекочитаемости. |
| `format_page_message(message)`         | Собирает page-вариант сообщения: `[timestamp \| sender \| flags]\nbody\nMentions: ...`. Это то, что попадает в `page_content` → rerank читает именно это. |
| `format_dense_message(message)`        | Собирает dense-вариант: гладкий текст без timestamp-шапок - embedding-модель не должна тратить токены на метаданные. |
| `format_sparse_message(message)`       | Собирает sparse-вариант: keyword-rich, с `sender:`, mentions, флагами `forwarded/quoted` - BM25 ловит эти токены как сигнал. |
| `estimate_page_message_size(message)`  | Возвращает `len(format_page_message(message)) + 2`. Используется в `select_overlap_context` и `should_flush_chunk`, чтобы заранее оценить размер без двойного формирования. |
| `build_page_content(chat, ctx, msgs)`  | Собирает финальный `page_content`: `CHAT:`, `CHAT_TYPE:`, `CHAT_ID:`, затем `CONTEXT:` + `MESSAGES:`. Эта структура потом парсится `split_page_sections` в search. |
| `build_dense_content(chat, ctx, msgs)` | Собирает `dense_content`: шапка чата + гладко склеенные dense-варианты, без секций CONTEXT/MESSAGES. |
| `build_sparse_content(chat, ctx, msgs)`| Собирает `sparse_content`: керн-рич текст с тегами, шапкой чата, флагами. |
| `split_message_for_chunking(message)`  | Если техническое сообщение слишком длинное - режет на `NormalizedMessage`-фрагменты с `fragment_index/fragment_count`. Non-technical - оставляет целым. |
| `split_long_text(text, limit)`         | Helper для `split_message_for_chunking`: режет текст по `\n` / пробелам так, чтобы не разорвать слово. |
| `should_flush_chunk(current, next_msg)`| Решение «закрывать ли текущий chunk»: по общему размеру, по `MAX_TIME_GAP`, по `thread_sn` смене, по `starts_new_topic`. |
| `select_overlap_context(messages)`     | Выбирает **последние** `OVERLAP_MESSAGE_COUNT` сообщений из предыдущих `overlap_messages` для CONTEXT - но с учётом `MAX_TIME_GAP_SECONDS` и лимита символов. |
| `select_thread_aware_overlap_context(...)` | То же, но дополнительно бустит сообщения из того же `thread_sn`, что и первый new_message - не теряем thread-контекст при смене темы. |
| `build_chunk_item(chat, ctx, msgs)`    | Собирает один `IndexAPIResultItem`: вызывает все три `build_*_content` + считает `message_ids` + metadata. |
| `build_chunks(payload)`                | Итерирует по `new_messages`, накапливает в `current_chunk`, вызывает `should_flush_chunk` - и на каждом flush'е зовёт `build_chunk_item`. Выдаёт финальный список chunks. |
| `log_chunk_diagnostics(chunks)`        | Дебаг-лог: сколько чанков, средний размер, распределение `thread_sn`. Включается через `LOG_DIAGNOSTICS`. |

### `sparse.py` - локальная BM25-модель

| Функция                      | Что делает |
| ---------------------------- | ---------- |
| `get_sparse_model()`         | `@lru_cache(maxsize=1)` - ленивая инициализация `SparseTextEmbedding(model_name="Qdrant/bm25")`. Грузит модель один раз на процесс из `FASTEMBED_CACHE_PATH`. |
| `embed_sparse_texts(texts)`  | Вызывает `model.embed(texts)`, конвертирует `item.indices/values` в Python `list`, возвращает `[{indices: [...], values: [...]}]` - ровно контракт `POST /sparse_embedding`. |

### Как читать эту таблицу

- Любая строка = одна функция в рантайме.
- «Используется в …» опущен только там, где функция - helper для своего же модуля.
- Полная цепочка вызовов `/index`: `index → build_chunks → (для каждого flush) build_chunk_item → build_page_content / build_dense_content / build_sparse_content → format_{page,dense,sparse}_message`.
- Полная цепочка `/sparse_embedding`: `sparse_embedding → embed_sparse_texts → get_sparse_model().embed()`.
