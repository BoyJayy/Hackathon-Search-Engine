
# API-спецификация и roadmap по хакатону

## 1. Полный список API

В проекте есть два сервиса:

- `index-service`
- `search-service`

## 2. API index-service

### 2.1 `GET /health`

#### Назначение
Проверка, что сервис жив и отвечает.

#### Ответ
HTTP `200 OK`

Пример:

---
```json
{
  "status": "ok"
}
```
---

#### Что важно
- этот endpoint нужен только для healthcheck
- на качество поиска напрямую не влияет
- должен стабильно работать в контейнере

### 2.2 `POST /index`

#### Назначение
Принимает чат и пачку сообщений, возвращает набор чанков для последующей индексации.

#### Request body

---
```json
{
  "data": {
    "chat": {
      "id": "string",
      "name": "string",
      "sn": "string",
      "type": "string",
      "is_public": true,
      "members_count": 0,
      "members": [
        {
          "additionalProp1": {}
        }
      ]
    },
    "overlap_messages": [
      {
        "id": "string",
        "thread_sn": "string",
        "time": 0,
        "text": "string",
        "sender_id": "string",
        "file_snippets": "string",
        "parts": [
          {}
        ],
        "mentions": [
          "string"
        ],
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
        "thread_sn": "string",
        "time": 0,
        "text": "string",
        "sender_id": "string",
        "file_snippets": "string",
        "parts": [
          {}
        ],
        "mentions": [
          "string"
        ],
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
---

#### Поля `chat`

- `id: string` — идентификатор чата
- `name: string` — название чата
- `sn: string` — внутренний идентификатор / short name
- `type: string` — тип чата
- `is_public: boolean | null` — публичный ли чат
- `members_count: integer | null` — количество участников
- `members: array<object> | null` — список участников и их данные

#### Поля `message`

- `id: string` — id сообщения
- `thread_sn: string | null` — id треда, если сообщение внутри треда
- `time: integer` — timestamp сообщения
- `text: string` — текст сообщения
- `sender_id: string` — id автора сообщения
- `file_snippets: string` — текстовые фрагменты из файлов / вложений
- `parts: array<object> | null` — структурные части сообщения
- `mentions: array<string> | null` — упоминания
- `member_event: object | null` — события, связанные с участниками
- `is_system: boolean` — системное ли сообщение
- `is_hidden: boolean` — скрытое ли сообщение
- `is_forward: boolean` — переслано ли сообщение
- `is_quote: boolean` — является ли цитатой

#### Логика входных массивов

- `overlap_messages` — контекст, который надо учитывать при chunking
- `new_messages` — новые сообщения, которые приходят на индексацию
- правильный indexer почти всегда должен смотреть на оба массива вместе

#### Response body

---
```json
{
  "results": [
    {
      "page_content": "string",
      "dense_content": "string",
      "sparse_content": "string",
      "message_ids": [
        "string"
      ]
    }
  ]
}
```
---

#### Поля ответа

- `page_content: string` — полный текст чанка, который будет храниться в payload
- `dense_content: string` — текст, по которому строится dense-вектор
- `sparse_content: string` — текст, по которому строится sparse-вектор
- `message_ids: array<string>` — список id сообщений, покрываемых чанком

#### Что важно для качества

- `page_content` должен содержать нормальный читаемый контекст
- `dense_content` должен быть очищен от мусора и полезен для semantic retrieval
- `sparse_content` должен быть насыщен keyword-сигналами
- `message_ids` нельзя терять, потому что метрика завязана именно на них

#### Возможный ответ текущего baseline
По скриншотам у текущей базы ответ выглядит очень примитивно, примерно так:

---
```json
{
  "results": [
    {
      "page_content": "string\nstring",
      "dense_content": "string\nstring",
      "sparse_content": "string\nstring",
      "message_ids": [
        "string"
      ]
    }
  ]
}
```
---

Это один из признаков, что baseline chunking очень слабый и почти заглушечный.

### 2.3 `POST /sparse_embedding`

#### Назначение
Строит sparse-вектора по текстам. Этот endpoint используется и на индексации, и на поиске.

#### Request body

---
```json
{
  "texts": [
    "string"
  ]
}
```
---

#### Поля
- `texts: array<string>` — список текстов, для которых нужно построить sparse embeddings

#### Ожидаемый смысл ответа
По ТЗ endpoint должен вернуть sparse-вектор совместимого формата:

- `indices`
- `values`

Swagger на скриншотах показывает только общую response-схему, но по задаче фактическая структура должна быть совместима с Qdrant sparse vectors.

#### Что важно
- sparse модель должна жить внутри контейнера
- нельзя тянуть ее из интернета во время проверки
- реализация должна быть быстрой и стабильной

### 2.4 Схемы index-service из swagger

#### `Chat`
- `id*: string`
- `name*: string`
- `sn*: string`
- `type*: string`
- `is_public: boolean | null`
- `members_count: integer | null`
- `members: array<object> | null`

#### `ChatData`
- `chat*: object`
- `overlap_messages*: array<object>`
- `new_messages*: array<object>`

#### `IndexAPIItem`
- `page_content*: string`
- `dense_content*: string`
- `sparse_content*: string`
- `message_ids*: array<string>`

#### `IndexAPIRequest`
- `data*: object`

#### `IndexAPIResponse`
- `results*: array<object>`

#### `Message`
- `id*: string`
- `thread_sn: string | null`
- `time*: integer`
- `text*: string`
- `sender_id*: string`
- `file_snippets*: string`
- `parts: array<object> | null`
- `mentions: array<string> | null`
- `member_event: object | null`
- `is_system*: boolean`
- `is_hidden*: boolean`
- `is_forward*: boolean`
- `is_quote*: boolean`

#### `SparseEmbeddingRequest`
- `texts*: array<string>`

#### `ValidationError`
- `loc*: array<(string | integer)>`
- `msg*: string`
- `type*: string`
- `input: any`
- `ctx: object`

## 3. API search-service

### 3.1 `GET /health`

#### Назначение
Проверка, что search-service жив.

#### Ответ
HTTP `200 OK`

Swagger показывает generic object response, то есть endpoint живой, но конкретный json может зависеть от реализации.

#### Что важно
- нужен для healthcheck
- должен стабильно подниматься в контейнере

### 3.2 `POST /search`

#### Назначение
Принимает обогащенный вопрос и возвращает отсортированный список релевантных `message_ids`.

#### Request body

---
```json
{
  "question": {
    "text": "string",
    "asker": "",
    "asked_on": "",
    "variants": [
      "string"
    ],
    "hyde": [
      "string"
    ],
    "keywords": [
      "string"
    ],
    "entities": {
      "people": [
        "string"
      ],
      "emails": [
        "string"
      ],
      "documents": [
        "string"
      ],
      "names": [
        "string"
      ],
      "links": [
        "string"
      ]
    },
    "date_mentions": [
      "string"
    ],
    "date_range": {
      "from": "string",
      "to": "string"
    },
    "search_text": "string"
  }
}
```
---

#### Поля `question`

- `text: string` — исходный текст вопроса
- `asker: string` — кто задал вопрос
- `asked_on: string` — когда вопрос был задан
- `variants: array<string> | null` — варианты / перефразировки вопроса
- `hyde: array<string> | null` — HyDE-подсказки
- `keywords: array<string> | null` — ключевые слова
- `entities: object | null` — извлеченные сущности
- `date_mentions: array<string> | null` — найденные упоминания дат
- `date_range: object | null` — нормализованный интервал дат
- `search_text: string` — поисковая нормализованная форма вопроса

#### Поля `entities`

- `people: array<string> | null`
- `emails: array<string> | null`
- `documents: array<string> | null`
- `names: array<string> | null`
- `links: array<string> | null`

#### Поля `date_range`

- `from: string`
- `to: string`

#### Response body

---
```json
{
  "results": [
    {
      "message_ids": [
        "string"
      ]
    }
  ]
}
```
---

#### Поля ответа

- `results: array<object>` — отсортированный список результатов
- `message_ids: array<string>` — id сообщений, покрываемых найденным чанком

#### Что важно для качества
Именно этот endpoint дает основной буст при грамотном использовании:
- `search_text`
- `variants`
- `hyde`
- `keywords`
- `entities`
- `date_mentions`
- `date_range`
- `asker`

Если искать только по `text`, то вы почти точно теряете качество.

### 3.3 Схемы search-service из swagger

#### `DateRange`
- `from*: string`
- `to*: string`

#### `Entities`
- `people: array<string> | null`
- `emails: array<string> | null`
- `documents: array<string> | null`
- `names: array<string> | null`
- `links: array<string> | null`

#### `Question`
- `text*: string`
- `asker: string`
- `asked_on: string`
- `variants: array<string> | null`
- `hyde: array<string> | null`
- `keywords: array<string> | null`
- `entities: object | null`
- `date_mentions: array<string> | null`
- `date_range: object | null`
- `search_text: string`

#### `SearchAPIItem`
- `message_ids*: array<string>`

#### `SearchAPIRequest`
- `question*: object`

#### `SearchAPIResponse`
- `results*: array<object>`

#### `ValidationError`
- `loc*: array<(string | integer)>`
- `msg*: string`
- `type*: string`
- `input: any`
- `ctx: object`

## 4. Что означают API для архитектуры решения

### `POST /index`
Это не просто «endpoint для индексации». Это главный контрольный пункт качества recall.

Именно здесь решается:
- как будут строиться chunk’и
- какой контекст попадет внутрь chunk’а
- как будут сформированы `dense_content`
- как будут сформированы `sparse_content`
- какие `message_ids` будут покрываться

Если этот endpoint слабый, сильный `search` уже не вытащит качество полностью.

### `POST /sparse_embedding`
Это технический, но важный endpoint.

Он нужен, чтобы:
- при индексации строить sparse vectors
- при поиске строить sparse query vectors
- поддерживать lexical / keyword retrieval

Если здесь слабая реализация, вы будете плохо ловить:
- exact match по именам
- email
- документам
- ссылкам
- датам
- коротким точным ответам

### `POST /search`
Это главный ranking endpoint.

Именно здесь решается:
- какой query реально строится
- используется ли весь enriched payload
- как dense retrieval комбинируется со sparse
- как merge’ятся кандидаты
- как применяется rerank
- как учитываются даты, entities и asker

Если этот endpoint использует только `question.text`, то большая часть полезного сигнала из API просто теряется.

## 5. Полная roadmap от текущего состояния до сдачи хакатона

Ниже уже нормальная, полная версия с учетом всего, что у нас теперь есть:

- ТЗ
- текущее состояние baseline
- Swagger по `index` сервису
- Swagger по `search` сервису
- то, что сейчас baseline уже поднимается, `GET /health` живой
- и то, что текущий `POST /index` по сути отдает почти заглушечный результат, а поиск у тебя ранее возвращал пустоту

## 5.1 Где мы находимся сейчас по факту

### Что уже есть

У нас уже есть два сервиса:

- `index-service`
- `search-service`

И по API сейчас видны такие контракты.

#### Index service
- `GET /health`
- `POST /index`
- `POST /sparse_embedding`

#### Search service
- `GET /health`
- `POST /search`

### Что видно по текущему baseline

#### `GET /health`
Работает. Это хорошо, значит контейнер и FastAPI в целом поднимаются.

#### `POST /index`
По текущему execute / example видно, что baseline сейчас формирует очень примитивный результат вида:

---
```json
{
  "results": [
    {
      "page_content": "string\nstring",
      "dense_content": "string\nstring",
      "sparse_content": "string\nstring",
      "message_ids": ["string"]
    }
  ]
}
```
---

Это почти наверняка означает не реально умный chunking, а очень базовую схему:
- берет сообщения
- как-то склеивает текст
- отдает один или несколько примитивных чанков без нормальной логики контекста

#### `POST /sparse_embedding`
Контракт есть, но по swagger видно только форму запроса/ответа. Значит sparse-часть пока рассматриваем как технически рабочую, но не оптимизированную.

#### `POST /search`
Контракт богатый, и это очень важно. По search API видно, что на вход приходят не только:
- `question.text`

но и дополнительные сигналы:
- `asker`
- `asked_on`
- `variants`
- `hyde`
- `keywords`
- `entities.people`
- `entities.emails`
- `entities.documents`
- `entities.names`
- `entities.links`
- `date_mentions`
- `date_range`
- `search_text`

Это значит, что baseline, который ищет только по `text`, почти гарантированно слабый. Основной буст качества здесь будет не просто от подключения dense+sparse, а от правильного использования всех этих полей.

### Главный вывод по текущей точке

Сейчас состояние такое:

- инфраструктура есть
- API контракты есть
- сервисы поднимаются
- baseline индексатор слишком примитивный
- baseline поиск почти точно недоиспользует входной enriched question
- значит главная задача — не написать сервисы с нуля, а заменить слабую retrieval-логику на сильную, не ломая контракт

## 5.2 Контракты, которые менять нельзя

Это фундамент. Вся roadmap ниже строится строго вокруг них.

### Index service
- `GET /health` должен отвечать `200 OK`
- `POST /index` должен принимать `data.chat`, `overlap_messages`, `new_messages`
- `POST /index` должен возвращать `results[]` с:
  - `page_content`
  - `dense_content`
  - `sparse_content`
  - `message_ids`
- `POST /sparse_embedding` должен принимать `texts`

### Search service
- `GET /health` должен отвечать `200 OK`
- `POST /search` должен принимать `question`
- `POST /search` должен возвращать `results[]` с `message_ids`

## 5.3 Главная стратегия решения

Если совсем по сути, то задача распадается на две части.

### `/index` отвечает за recall
Именно здесь ты решаешь:
- как разбить сообщения на чанки
- какой контекст попадет в один chunk
- какие тексты уйдут в dense
- какие тексты уйдут в sparse
- какие `message_ids` будет покрывать один результат

Если `/index` слабый, то даже сильный `/search` уже не спасет.

### `/search` отвечает за ranking и добор recall
Здесь ты решаешь:
- как построить query из enriched question
- как использовать dense retrieval
- как использовать sparse retrieval
- как объединять кандидатов
- как применять reranker
- как учитывать даты, автора вопроса, entities и прочие сигналы

Если `/search` слабый, правильные сообщения могут находиться, но будут стоять слишком низко.

## 5.4 Целевое состояние решения

### Index service должен
- стабильно принимать батчи сообщений
- корректно собирать чанки с учетом thread и времени
- формировать осмысленные `page_content`, `dense_content`, `sparse_content`
- возвращать хорошие `message_ids` покрытия

### Search service должен
- использовать все полезные поля `question`
- искать hybrid-способом: dense + sparse
- делать merge кандидатов
- rerank-ить top-N
- выдавать итоговый top-50 по релевантности

### Система в целом должна
- укладываться в лимиты по времени
- работать без интернета
- не ломать API контракты
- проходить полную проверку

