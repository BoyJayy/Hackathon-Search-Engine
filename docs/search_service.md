# Search Service

Эта doc описывает текущую реализацию `search`.

## Назначение

`search` принимает enriched question, делает hybrid retrieval по Qdrant и возвращает ранжированный список `message_ids`.

Сервис отвечает за:
- подготовку query;
- dense и sparse query embeddings;
- hybrid retrieval;
- fusion;
- rerank;
- финальную сборку выдачи.

## Текущая структура

Сейчас логика живёт в одном файле:
- [`search/main.py`](/Users/boyjayy/Documents/Search/search/main.py)

## Что приходит на `POST /search`

На вход приходит `question` со следующими полезными полями:
- `text`
- `search_text`
- `variants`
- `hyde`
- `keywords`
- `entities`
- `date_mentions`
- `date_range`
- `asker`
- `asked_on`

Контракт не меняется, но внутренняя логика может по-разному использовать эти поля.

## Текущий pipeline

```text
question
-> build primary query
-> build dense queries
-> build sparse queries
-> dense embed batch
-> sparse embed batch
-> Qdrant hybrid retrieval
-> RRF fusion
-> local rescoring
-> rerank top candidates
-> dedupe message_ids
-> top-50 results
```

## 1. Query preparation

### Primary query

Primary query сейчас строится как:
- `question.search_text`, если он есть;
- иначе `question.text`.

Это даёт более стабильный retrieval, чем поиск только по сырому `text`.

### Dense queries

Для dense retrieval сервис строит несколько запросов:
- `search_text`
- `text`
- до 4 `variants`
- до 3 `hyde`

После этого запросы:
- нормализуются;
- дедуплицируются;
- режутся по лимиту.

Цель:
- поймать смысловые перефразировки;
- использовать enriched question как источник recall.

### Sparse queries

Для sparse retrieval сервис строит keyword-heavy запросы из:
- primary query;
- `keywords`;
- `entities`;
- `date_mentions`;
- `asker`.

Это помогает лучше ловить:
- имена;
- email;
- ссылки;
- названия продуктов и документов;
- точные технические токены.

Дополнительно сервис может брать часть `variants` и `hyde` как дешёвые sparse-ветки, чтобы лучше ловить paraphrase-heavy кейсы.

## 2. Embeddings

### Dense

Dense embeddings считаются батчем через внешний API:
- `POST /embeddings`

Это экономит запросы и снижает latency по сравнению с вызовом на каждый query отдельно.

Если внешний dense API отвечает `429 Too Many Requests` или другим сетевым/HTTP-сбоем, сервис не падает:
- dense-ветка временно отключается для этого запроса;
- retrieval продолжается как sparse-only.

### Sparse

Sparse embeddings считаются локально через `fastembed`:
- модель `Qdrant/bm25`

Sparse queries тоже считаются батчем.

## 3. Retrieval и fusion

Сервис отправляет в Qdrant несколько prefetch-веток:
- несколько dense prefetch;
- несколько sparse prefetch.

После этого используется:
- `Fusion.RRF`

Это даёт более безопасный baseline, чем ручное смешивание dense/sparse score.

## 4. Rerank

Перед внешним rerank сервис делает мягкий local rescoring кандидатов.

Он использует:
- точные phrase hits из `keywords`, `entities`, `date_mentions`, `asker`;
- token hits из `search_text` / `text` / части `variants` / `hyde`;
- лучший message-block внутри chunk, чтобы короткие точные ответы не проигрывали длинным обсуждениям;
- для quoted message-block сильнее доверяет собственной реплике, чем процитированному тексту;
- metadata сигналы по `participants` и `mentions`;
- мягкий temporal boost, если есть `date_range`.

При этом rescoring сейчас осознанно сильнее смотрит на секцию `MESSAGES`, чем на `CONTEXT`.
Это важно, потому что overlap-контекст часто полезен для retrieval, но может мешать ранжированию:
- запрос совпадает с предыдущим сообщением в `CONTEXT`;
- а сервису нужно поднять текущий message-block, который действительно отвечает на вопрос.

Аналогичная проблема бывает у quoted-сообщений:
- ответ с цитатой может тащить в себя весь текст исходного вопроса;
- live-update может цитировать старый анонс и выглядеть релевантнее самого анонса.

Поэтому quote-aware rescoring старается не давать quoted-тексту перетягивать весь score на себя.

Если candidate совпал только по `CONTEXT`, а сам `MESSAGES`-блок не содержит нужных сигналов, сервис дополнительно штрафует такой chunk.

Этот слой:
- почти ничего не стоит по latency;
- помогает подтянуть exact matches выше ещё до внешнего reranker;
- улучшает fallback-поведение, если внешний reranker недоступен или rate-limited.

Сам rerank query тоже делается не совсем сырым:
- база = `search_text`, иначе `text`;
- при наличии точных сигналов сервис добавляет 1-2 кратких уточнения из `keywords` / `entities` / `date_mentions`.

После retrieval сервис:
- берёт top кандидатов;
- отправляет только ограниченный top-N во внешний reranker;
- режет слишком длинный `page_content` до компактной версии;
- переставляет секции кандидата в порядке `MESSAGES -> CONTEXT`, чтобы reranker сначала видел сам ответ, а потом overlap-контекст;
- после ответа reranker использует local boost как мягкий stabilizer для exact/entity-heavy кейсов;
- пересортировывает кандидаты по score reranker.

Это повышает качество первых позиций, а значит и `nDCG@50`.

### Защита от rate limit

Внешний reranker может отвечать `429 Too Many Requests`.

Чтобы из-за этого не падать целиком:
- rerank делается только для небольшого top-N;
- в reranker отправляется укороченный текст кандидата;
- если внешний `/score` вернул `429`, сервис не падает `500`, а возвращает retrieval order fallback без rerank.

Аналогичная защита есть и у dense embeddings:
- если внешний `/embeddings` вернул `429` или другой upstream error, сервис не падает `500`;
- dense retrieval для этого запроса отключается, и поиск продолжается по sparse-ветке.

## 5. Final assembly

После rerank сервис:
- собирает `message_ids` из payload;
- пытается мягко переупорядочить их внутри chunk по message-level exact signals;
- для вопросов вида `первое`, `исходный вопрос`, `начало` умеет слегка поднимать ранние сообщения внутри chunk;
- после этого ещё ранжирует кандидаты уже на уровне отдельных `message_id`, чтобы правильный ответ из второго chunk мог обогнать нерелевантный первый `message_id` из chunk выше;
- дедуплицирует их с сохранением порядка;
- ограничивает итоговый список до `50`.

Это важно, потому что:
- метрика считается по `message_ids`;
- всё после первых 50 всё равно отбрасывается.

## 6. Что search пока не делает

Сервис пока не:
- применяет жёсткие date filters в Qdrant;
- использует жёсткие metadata filters по `participants` / `mentions`;
- использует отдельную сложную query strategy по `date_range`.

То есть текущая реализация уже сильнее baseline, но ещё оставляет пространство для тюнинга.
