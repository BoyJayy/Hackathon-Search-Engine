# Chunking Input Spec

Короткая памятка для разработчика `index`-сервиса: что должно приходить в модуль chunking и в каком виде.

Основание:
- [hackathon_roadmap.md](/Users/nikita/code/Hackathon-Search-Engine/hackathon_roadmap.md)
- [ml-roadmap.md](/Users/nikita/code/Hackathon-Search-Engine/ml-roadmap.md)

## 1. Что приходит во внешний API `/index`

Во входном request приходит:

```json
{
  "data": {
    "chat": {
      "id": "string",
      "name": "string",
      "sn": "string",
      "type": "string"
    },
    "overlap_messages": [
      {
        "id": "string",
        "thread_sn": "string|null",
        "time": 0,
        "text": "string",
        "sender_id": "string",
        "file_snippets": "string",
        "parts": [{}],
        "mentions": ["string"],
        "member_event": {},
        "is_system": false,
        "is_hidden": false,
        "is_forward": false,
        "is_quote": false
      }
    ],
    "new_messages": [
      {
        "id": "string",
        "thread_sn": "string|null",
        "time": 0,
        "text": "string",
        "sender_id": "string",
        "file_snippets": "string",
        "parts": [{}],
        "mentions": ["string"],
        "member_event": {},
        "is_system": false,
        "is_hidden": false,
        "is_forward": false,
        "is_quote": false
      }
    ]
  }
}
```

Важно:
- `overlap_messages` — это контекст на границе батча
- `new_messages` — это новые сообщения, которые индексируем сейчас
- chunking должен учитывать оба массива вместе

## 2. Что НЕ должно приходить напрямую в chunking

В chunking не стоит передавать сырой request целиком.

Не надо смешивать в одном модуле:
- HTTP/Pydantic-валидацию
- очистку текста
- фильтрацию мусора
- логику разбиения на чанки

До chunking сообщения должны быть уже нормализованы.

## 3. Что должно приходить в chunking внутри сервиса

Рекомендуемый вход chunking-модуля:

```python
NormalizedMessage(
    id: str,
    source: str,            # "overlap" | "new"
    thread_sn: str | None,
    thread_key: str,        # thread_sn or "__root__"
    time: int,
    sender_id: str,
    body_text: str,         # основной текст после нормализации
    dense_text: str,        # текст для dense-представления
    sparse_text: str,       # текст для sparse-представления
    mentions: tuple[str, ...],
    file_snippets: str,
    is_forward: bool,
    is_quote: bool,
)
```

Минимально обязательные поля для chunking:
- `id`
- `source`
- `thread_sn` / `thread_key`
- `time`
- `body_text`

Практически полезные поля:
- `sender_id`
- `mentions`
- `file_snippets`
- `is_forward`
- `is_quote`

## 4. Что должно быть сделано ДО chunking

Перед тем как вызывать chunking, нужен этап `message_processing`:

1. Отфильтровать:
- `is_hidden == true`
- `is_system == true`
- полностью пустые сообщения

2. Собрать текст сообщения из:
- `text`
- `parts[*].text`
- `file_snippets`
- `mentions` при необходимости для sparse

3. Стабильно отсортировать сообщения:
- внутри thread по `(time, id)`

4. Сохранить происхождение сообщения:
- `source="overlap"`
- `source="new"`

## 5. Как chunking должен мыслить вход

Chunking работает не с "батчем JSON", а с потоком нормализованных сообщений:

- сначала сообщения делятся по `thread_key`
- затем внутри каждого потока идут по времени
- `overlap` используется как контекст
- `new` образует body текущих чанков

Хорошая ментальная модель:

```text
raw request
-> schemas
-> message_processing
-> list[NormalizedMessage]
-> chunking
-> list[ChunkWindow]
-> content_builders
-> API response
```

## 6. Что должен возвращать chunking

Chunking не должен сразу строить финальный API response.

Лучше возвращать промежуточную структуру окна:

```python
ChunkWindow(
    thread_key: str,
    context_messages: tuple[NormalizedMessage, ...],
    body_messages: tuple[NormalizedMessage, ...],
    all_messages: tuple[NormalizedMessage, ...],
)
```

Где:
- `context_messages` — overlap-контекст
- `body_messages` — новые сообщения, ради которых создан chunk
- `all_messages` — `context + body`

## 7. Что важно для соответствия roadmap

Chunking по roadmap должен быть:
- message-boundary
- thread-aware
- позже time-aware
- с controlled overlap

Именно поэтому вход в chunking должен быть уже очищенным и нормализованным.

Если подавать туда сырой JSON, модуль быстро станет перегруженным и нестабильным.

