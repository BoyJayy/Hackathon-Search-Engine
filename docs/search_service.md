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

Для plain-вопросов без `keywords` / `hyde` сервис добавляет небольшие domain expansions по известным техническим темам: Go 1.18, SIGABRT/macOS, CGO, PDF/OCR, Qdrant, oncall, release smoke-check, migrations, Terraform provider, demo и technology cards.
Это recall-oriented слой: он помогает найти ответ, когда вопрос содержит тему, а сам ответ содержит только факты.

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

На время жизни контейнера сервис кэширует dense embeddings по паре `model + text`.
Это особенно полезно на eval, где встречаются повторяющиеся или почти повторяющиеся enriched-запросы:
- меньше внешних вызовов;
- ниже шанс получить `429`;
- повторные проверки идут быстрее.

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

Ключевые retrieval-параметры читаются из env:
- `MAX_DENSE_QUERIES`
- `MAX_SPARSE_QUERIES`
- `DENSE_PREFETCH_K`
- `SPARSE_PREFETCH_K`
- `RETRIEVE_K`

Это позволяет гонять search-only sweep без пересборки образа и без re-ingest.

## 4. Rerank

Перед внешним rerank сервис делает мягкий local rescoring кандидатов.

Он использует:
- точные phrase hits из `keywords`, `entities`, `date_mentions`, `asker`;
- token hits из `search_text` / `text` / части `variants` / `hyde`;
- optional intent-aware сигнал через `INTENT_ALIGNMENT_WEIGHT`: summary-вопросы поднимают ответы с документом/ссылкой, detail-вопросы поднимают короткие содержательные ответы;
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
- смешивает score reranker с исходным retrieval-порядком через `RERANK_ALPHA`;
- пересортировывает кандидаты по blended score.

Это повышает качество первых позиций, а значит и `nDCG@50`.

Reranker scores тоже кэшируются по `model + query + candidate text`.
Это снижает число повторных `/score` вызовов и помогает не упираться в rate limit.

Ключевые rerank-параметры читаются из env:
- `RERANK_ALPHA`
- `RERANK_LIMIT`
- `RERANK_MAX_TEXT_CHARS`
- `UPSTREAM_CACHE_MAX_ITEMS`
- `UPSTREAM_MAX_RETRIES`
- `UPSTREAM_RETRY_DELAY_SECONDS`

Default `RERANK_ALPHA = 0.3`, то есть боевой режим делает blended rerank, но оставляет retrieval order основным якорем. Это recall-safe и всё ещё поднимает первые позиции.
Default retrieval depth тоже broadened: `DENSE_PREFETCH_K = 70`, `RETRIEVE_K = 150`, `RERANK_LIMIT = 20`, `MAX_SPARSE_QUERIES = 8`.
Default `INTENT_ALIGNMENT_WEIGHT = 0.0`, потому что на сервере важнее не просадить recall; intent-layer оставлен как ручной knob для sweep.

### Защита от rate limit

Внешний reranker может отвечать `429 Too Many Requests`.

Чтобы из-за этого не падать целиком:
- rerank делается только для небольшого top-N;
- в reranker отправляется укороченный текст кандидата;
- при `429` сервис делает короткий retry;
- если внешний `/score` вернул `429`, сервис не падает `500`, а возвращает retrieval order fallback без rerank.

Аналогичная защита есть и у dense embeddings:
- при `429` сервис делает короткий retry;
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
