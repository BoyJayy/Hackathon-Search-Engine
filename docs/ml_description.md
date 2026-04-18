# ML Description

Техническое описание реализованных ML-компонентов. Живой документ — дописывается по мере прохождения [роадмапа](ml-roadmap.md).

---

## Полный pipeline

Две фазы: **индексация** (один раз при загрузке данных) и **поиск** (на каждый вопрос).

### Фаза 1 — индексация

```
data/Go Nova.json  (messages[])
       │
       ▼
┌─────────────────────────────────────────────┐
│ [A] index-service  POST /index              │
│     build_chunks(overlap_msgs, new_msgs)    │
│     → filter + sort + boundary-cut + overlap│
│     → IndexAPIItem[]                        │
└─────────────────────────────────────────────┘
       │
       │  IndexAPIItem {
       │    page_content,    ← в payload, вход реранкера
       │    dense_content,   ← в dense-модель
       │    sparse_content,  ← в sparse-модель
       │    message_ids      ← в payload, финальный вывод
       │  }
       │
       ├──► dense_content  ───► POST к внешнему dense API (Qwen3-Embedding-0.6B, Basic Auth)
       │                         возвращает vector[1024] float32
       │
       ├──► sparse_content ───► POST /sparse_embedding (локальный index-service)
       │                         fastembed + Qdrant/bm25
       │                         возвращает {indices, values}
       │
       └──► page_content   ───► НЕ эмбеддится, лежит в payload
       │
       ▼
┌─────────────────────────────────────────────┐
│ [B] ingestion orchestrator  (eval/ingest.py)│
│     stable_chunk_id(chat_id, message_ids)   │
│     upsert(qdrant_point)                    │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ [C] Qdrant collection                       │
│     named vectors: "dense" (cosine, 1024)   │
│                    "sparse" (bm25 + IDF)    │
│     payload: page_content + metadata        │
└─────────────────────────────────────────────┘
```

Один чанк → **одна точка Qdrant с двумя векторами** (`dense` + `sparse`) + payload. `chunk_id` стабильный (hash от `chat_id + sorted(message_ids)`) → повторный ingest идемпотентен.

### Фаза 2 — поиск

```
question = {text, variants, hyde, keywords, entities, date_range, asker, ...}
       │
       ├──► text ───► dense API      ───► query_dense[1024]
       │
       ├──► text ───► fastembed bm25 ───► query_sparse{indices, values}
       │
       ▼
┌─────────────────────────────────────────────┐
│ [D] Qdrant hybrid query                     │
│     prefetch_1 = top DENSE_PREFETCH_K       │
│                   по dense cosine           │
│     prefetch_2 = top SPARSE_PREFETCH_K      │
│                   по sparse BM25+IDF        │
│     fusion = RRF(prefetch_1, prefetch_2)    │
│     return top RETRIEVE_K                   │
└─────────────────────────────────────────────┘
       │
       │  points[] с payload.page_content + metadata
       │
       ▼
┌─────────────────────────────────────────────┐
│ [E] Rerank                                  │
│     POST к внешнему rerank API              │
│     nvidia/llama-nemotron-rerank-1b-v2      │
│     input: (query, [page_content1, ...])    │
│     output: [score1, ...]                   │
│     → points пересортированы по score       │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ [F] Final assembly  (TODO — Этап 10)        │
│     flatten message_ids по всем чанкам      │
│     dedup + diversity control               │
│     ограничить 50 unique msg_ids            │
└─────────────────────────────────────────────┘
       │
       ▼
{"results": [{"message_ids": ["...", "...", ...]}]}
```

**RRF (Reciprocal Rank Fusion):** сумма `1/(60 + rank_i)` по каждому списку. Rank-based, не чувствителен к абсолютным значениям — поэтому работает для несопоставимых dense cosine vs BM25.

### Роли трёх полей контента

Чанк хранит **три версии текста** — каждая под свою модель. Сейчас все три идентичны (`"\n".join(msg_texts)`), разделение — Этап 2 роадмапа.

| Поле | Куда идёт | Идеальное содержимое |
|---|---|---|
| `dense_content` | Вход dense-модели | Чистая семантика: тексты сообщений + извлечённые сущности. Без `CHAT_ID:`-заголовков, они засоряют эмбеддинг. |
| `sparse_content` | Вход BM25 | Keyword dump: текст + имена + emails + даты + названия продуктов. Нормализованная морфология русского. |
| `page_content` | В payload → вход реранкера | Структурированный диалог с timestamps и авторами. Помогает реранкеру понять «кто кого спрашивает» → растит nDCG. |

### Что уже реализовано vs план

| Компонент | Статус | Ссылка |
|---|---|---|
| [A] chunking (message-boundary + overlap) | ✅ реализовано | [`index/main.py`](../index/main.py), Этап 1 |
| [A] `page`/`dense`/`sparse` разделение | ⏳ дубли, Этап 2 | [roadmap](ml-roadmap.md#этап-2-content-separation-page--dense--sparse) |
| [B] stable chunk_id + ensure_collection | ✅ реализовано | [`eval/ingest.py`](../eval/ingest.py), Этап 3 |
| [B] ingestion payload metadata | ✅ базовый набор | `eval/ingest.py:build_metadata` |
| [D] hybrid retrieval (RRF) | ✅ baseline | [`search/main.py`](../search/main.py), Этап 6 |
| [D] query expansion (variants/HyDE/keywords) | ⏳ не реализовано | Этап 4 |
| [D] soft-filters (entities/date_range) | ⏳ не реализовано | Этап 3 |
| [E] rerank top 40-60 (а не top 10) | ✅ фикс применён | Этап 7 |
| [E] score mixing `α·rerank + (1-α)·retrieval` | ⏳ не реализовано | Этап 7 |
| [F] final assembly (dedup + diversity) | ⏳ не реализовано | Этап 10 |
| Eval harness (Recall@50, nDCG@50, score) | ✅ реализовано | [`eval/`](../eval/), [eval-harness.md](eval-harness.md) |

---

## Пример на пальцах

Прогоним весь pipeline на тривиальном чате. Чтобы влезло в пример, представим `UPPER_CHARS=500` (в реальности 1600).

### Вход — 6 сообщений в чате «Релиз v2.1»

```
m01  [10:00]  alice@corp    thread=None   "Ребят, когда деплой v2.1?"
m02  [10:02]  bob@corp      thread=None   "В пятницу в 18:00 планируем"
m03  [10:05]  alice@corp    thread=None   "Нужен доступ к проду?"
m04  [10:07]  bob@corp      thread=None   "Да, попроси Carol оформить"
m05  [10:10]  alice@corp    thread=None   "Carol, нужен доступ для релиза v2.1"
m06  [10:12]  carol@corp    thread=None   "Оформлю до четверга"
```

### Фаза 1 — индексация

**[A] `/index` build_chunks:**

Все в одном треде, малые gap-ы — hard boundary не срабатывает. Сработает только size boundary. Пусть каждое сообщение ~180 chars. Накапливаем:

- m01 (180) → current_len=180
- + m02 (180) → 360 < 500, добавляем
- + m03 (180) → 540 > 500 (UPPER), 360 ≥ 125 (LOWER) → **эмитим chunk_A**

`chunk_A = [m01, m02]`, переносим overlap (last 2) → `current = [m01, m02]`.

Хм, overlap=2 а в current 2 → overlap = весь чанк → он повторится. Для примера возьмём overlap=1:

- `chunk_A = [m01, m02]`, overlap carry `[m02]`
- current = [m02] → + m03 → 360 → + m04 → 540 > 500 → **эмитим chunk_B = [m02, m03, m04]**, carry `[m04]`
- current = [m04] → + m05 → 360 → + m06 → 540 > 500 → **эмитим chunk_C = [m04, m05, m06]**

Итог: **3 чанка**, m02 и m04 повторяются в соседних (это overlap).

**IndexAPIItem[0]** (chunk_A):

```python
{
  "page_content":   "Ребят, когда деплой v2.1?\nВ пятницу в 18:00 планируем",
  "dense_content":  "Ребят, когда деплой v2.1?\nВ пятницу в 18:00 планируем",
  "sparse_content": "Ребят, когда деплой v2.1?\nВ пятницу в 18:00 планируем",
  "message_ids":    ["m01", "m02"]
}
```

(на Этапе 1 все три поля идентичны)

**[A+B] Эмбеддинги + ingest для chunk_A:**

```
dense_content  ──► POST dense API ──► vector[1024] = [0.12, -0.04, 0.31, ..., 0.07]
sparse_content ──► fastembed bm25 ──► indices=[42, 117, 9001, 2048, 777]
                                      values =[2.1, 1.8, 3.4, 1.2, 0.9]
```

**Один Qdrant point:**

```python
PointStruct(
    id = uuid5(NAMESPACE, "chat-relz:m01,m02"),   # стабильный, детерминированный
    vector = {
        "dense":  [0.12, -0.04, ..., 0.07],
        "sparse": SparseVector(indices=[42,117,9001,2048,777],
                               values =[2.1,1.8,3.4,1.2,0.9]),
    },
    payload = {
        "page_content": "Ребят, когда деплой v2.1?\nВ пятницу в 18:00 планируем",
        "metadata": {
            "chat_name":  "Релиз v2.1",
            "chat_type":  "group",
            "message_ids": ["m01", "m02"],
            "start": "1700000000",
            "end":   "1700000120",
            "participants": ["alice@corp", "bob@corp"],
            "mentions": [],
            "contains_forward": False,
            "contains_quote":   False
        }
    }
)
```

То же самое для chunk_B и chunk_C — всего **3 точки в Qdrant**.

### Фаза 2 — поиск

**Вопрос:** «Когда релиз v2.1 и кто даёт доступ к проду?»

**[D] Эмбеддинг запроса + hybrid retrieval:**

```
query                             │
  │                               │
  ├► dense API ► query_dense[1024]│ Qdrant query_points(
  │                               │   prefetch=[
  ├► fastembed ► query_sparse     │     Prefetch(query=query_dense,  using=dense,  limit=100),
  │                               │     Prefetch(query=query_sparse, using=sparse, limit=100)
  │                               │   ],
  │                               │   query=FusionQuery(RRF),
  │                               │   limit=100
  │                               │ )
```

Dense семантически близок к **chunk_A** (слова «деплой», «v2.1», «пятница») и **chunk_B** («доступ к проду»).
Sparse BM25 ловит точное совпадение `"v2.1"` → тоже chunk_A и chunk_C (там есть `"релиза v2.1"`).

**RRF объединяет ранги:**

| chunk | dense rank | sparse rank | RRF score |
|---|---|---|---|
| chunk_A | 1 | 1 | 1/(60+1) + 1/(60+1) ≈ 0.033 |
| chunk_C | 3 | 2 | 1/(60+3) + 1/(60+2) ≈ 0.032 |
| chunk_B | 2 | 4 | 1/(60+2) + 1/(60+4) ≈ 0.032 |

Qdrant вернёт все 3 в таком порядке: A, C, B.

**[E] Rerank:**

```
POST rerank API
  text_1 = "Когда релиз v2.1 и кто даёт доступ к проду?"
  text_2 = [
    chunk_A.page_content,   # про "когда деплой, в пятницу"
    chunk_C.page_content,   # про "доступ для релиза v2.1, оформлю"
    chunk_B.page_content    # про "нужен доступ к проду, попроси Carol"
  ]
```

Реранкер вернёт scores:

```
chunk_B: 0.94   ← отвечает на обе части вопроса (доступ + Carol)
chunk_A: 0.81   ← отвечает на первую (когда)
chunk_C: 0.76   ← частично (релиз v2.1, но про оформление доступа)
```

Пересортировка: **B → A → C**.

**[F] Final assembly** (пока не реализовано, но концепт):

```python
flat_ids = []
for chunk in [B, A, C]:
    flat_ids.extend(chunk.message_ids)
# = [m02, m03, m04, m01, m02, m04, m05, m06]

dedup_preserve_order = [m02, m03, m04, m01, m05, m06]   # 6 уникальных
# m02 и m04 появлялись дважды (overlap) — оставили первое появление

# ограничить 50 (у нас всего 6)
return {"results": [{"message_ids": [m02, m03, m04, m01, m05, m06]}]}
```

**Gold answer** (например): `{m02, m04, m06}` — «в пятницу», «попроси Carol», «оформлю до четверга».

**Метрики для этого вопроса:**

```
predicted = [m02, m03, m04, m01, m05, m06]
relevant  = {m02, m04, m06}

Recall@50 = |{m02,m04,m06} ∩ predicted[:50]| / 3 = 3/3 = 1.0    ← все пойманы
nDCG@50   = (1/log2(2) + 1/log2(4) + 1/log2(7)) / IDCG ≈ 0.85
score     = 0.8·1.0 + 0.2·0.85 = 0.97
```

Если бы мы не сделали overlap — m02 и m04 были бы только в одном чанке. Если бы retrieval нашёл только chunk_C — потеряли бы m02 и m04 из выдачи → Recall=1/3. Вот **зачем нужен overlap**.

Если бы не было реранка — порядок A, C, B. m04 на позиции 6 вместо 3 → nDCG ниже. Вот **зачем нужен rerank**.

Если бы не было dedup — `[m02, m03, m04, m01, m02, m04, m05, m06]`, и в топ-50 первые 4 слота жёстко «прибиты», из 6 unique msg_ids мы бы получили те же 6, но при более строгих K это бы сожрало слоты. Вот **зачем нужна final assembly**.

---

## Chunking — [`index/main.py`](../index/main.py)

### Что делаем

Из списка сообщений чата строим **чанки** — единицы, которые дальше эмбеддятся (dense + sparse) и грузятся в Qdrant. Поиск работает на уровне чанков, не отдельных сообщений.

Задача chunking: нарезать так, чтобы:
1. **Контекст сохранялся** — сообщение-ответ не оторвано от сообщения-вопроса.
2. **Размер был подходящим** — не слишком большой (размазывает семантику), не слишком мелкий (нет контекста для rerank).
3. **Важные сообщения на границах** не терялись — overlap их подхватывает.
4. **Индекс не раздувался** — дубликаты жрут место и слоты в топе.

### Pipeline

```
overlap_messages + new_messages
          ↓
  [1] render + filter        → выбросить системщину и hidden
          ↓
  [2] sort by (time, id)     → воспроизводимость
          ↓
  [3] walk messages          → накапливать в current[]
          ↓
  [4] boundary check         → emit current как chunk
          ↓
  [5] keep only chunks       → с хотя бы одним new_message
      with ≥1 new msg
```

### [1] Render

Из одного `Message` собираем строку:

```
message.text
message.parts[*].text          # ссылки, вложения с описанием
message.file_snippets          # извлечённый текст из файлов
```

Всё склеивается через `\n`, `.strip()`. Empty render → сообщение исключается из chunking.

### [2] Filter

**Выкидываем (жёстко):**
- `is_hidden == true` — удалённые/скрытые
- `is_system == true` — служебные события (вход/выход участников)
- Пустой render (нет ни `text`, ни `parts[*].text`, ни `file_snippets`)

**Что НЕ делаем:**
- **Не фильтруем по длине** `len(text) < N`. Короткое сообщение может быть ценным — версия (`v1.18`), ссылка, email, имя. Content-aware фильтрация — отложена до Этапа 9 (нормализация).

### [3] Sort

Стабильная сортировка `(time, id)`. Даёт воспроизводимость между запусками: одни и те же сообщения → один и тот же порядок → одинаковые чанки → одинаковые `chunk_id` (hash от message_ids в [ingestion](../eval/ingest.py)).

### [4] Boundaries

Проход по сообщениям, накопление в `current[]`. Решение «резать или нет» на каждом шаге — по трём осям.

#### Hard boundaries (режем всегда, overlap не переносится)

```python
thread_change = msg.thread_sn != prev.thread_sn
time_gap      = msg.time - prev.time > 3600   # > 1 час
```

Смысл: если поменялся тред или прошёл большой промежуток — это разные разговоры, склеивать их нет смысла.

#### Size boundary (режем, если накопили достаточно, overlap переносится)

```python
current_len + len(msg) > UPPER_CHARS and current_len >= LOWER_CHARS
```

Смысл: держим размер в рабочей полосе. `LOWER_CHARS` не даёт эмитить крохотные чанки (меньше 400 chars = меньше ~100 токенов = нет контекста). `UPPER_CHARS` не даёт разрастаться сверх меры.

#### Overlap — последние N сообщений переносятся

Если сработал **size** boundary — последние `OVERLAP_MESSAGES=2` сообщения копируются в начало следующего чанка. Идея: сообщение на стыке («Вопрос:...» в конце одного чанка, «Ответ:...» в начале следующего) не теряет контекст ни с одной стороны.

Если сработал **hard** boundary (thread/gap) — overlap **не** переносится. Контекст другого треда/давно прошедшего разговора не помогает, только засоряет.

### [5] Output filter

Эмитим только чанки, содержащие ≥1 сообщение из `new_messages`. Чанки, состоящие только из `overlap_messages` (прошлой инкрементальной порции) — пропускаем, они уже были проиндексированы раньше.

### Параметры

| Константа | Значение | Почему |
|---|---|---|
| `UPPER_CHARS` | 1600 | ~400 токенов. Верх «рабочей полосы» по [roadmap Этап 1](ml-roadmap.md). Выше — теряется семантическая фокусировка. |
| `LOWER_CHARS` | 400 | ~100 токенов. Ниже — чанк слишком мал для rerank judgment. |
| `TIME_GAP_SECONDS` | 3600 | Час — универсальный индикатор «разговор закончился». Больше — склеиваем разные разговоры. Меньше — режем естественные паузы. |
| `OVERLAP_MESSAGES` | 2 | +36% индекса на плотных тредах, 36% сообщений покрыты двумя чанками. См. тюнинг ниже. |

### Как подбирали параметры

Эмпирическая проверка — sweep на реальных (`data/Go Nova.json`, 25 msgs) и синтетических данных (50 msgs × 250 chars).

**Overlap:** тестировали 0..6.
- 0 — дубликатов нет, но boundary-сообщения страдают.
- 1 — консервативно: 16% boundary coverage.
- **2 — выбранный компромисс:** +36% индекса, 36% boundary coverage.
- 3 — +66% индекса, 66% coverage. Diminishing returns.
- 4+ — взрыв индекса, 5+ начинает ломать (overlap ≥ chunk_size_в_сообщениях).

**UPPER_CHARS:** тестировали 800..2800.
- 800 + overlap=2 — dup 2.88x (катастрофа на плотных данных — overlap близок к chunk-size-in-msgs).
- 1200 — +92% coverage, но dup 1.92x.
- **1600 — сладкая зона:** dup 1.44 синтетика, 1.08 real.
- 2000-2800 — меньше гранулярности, больше контекста.

**Почему именно 1600, не 2000:** Recall весит 0.8 в итоговом score, nDCG 0.2. → retrieval важнее rerank quality → меньше чанков-кандидатов = хуже. Мелкие чанки дают больше distinct retrieval-слотов. Граница где rerank ещё умеет судить релевантность — ~150-400 токенов. 1600 chars = ~400 токенов = минимум для rerank, максимум для retrieval гранулярности.

### Правило масштабирования

`OVERLAP_MESSAGES` не должен превышать `chunk_size_в_сообщениях / 3`. Иначе каждый чанк — почти-дубль предыдущего, индекс раздувается нелинейно.

На текущих параметрах (UPPER=1600, msg ≈ 250-500 chars → chunk_size ≈ 4-6 msgs) overlap=2 = 33-50% от chunk — верхняя граница допустимого. Если данные окажутся с более короткими сообщениями (chunk_size → 8-10 msgs), можно безопасно поднять до 3.

### Output schema

Каждый чанк — `IndexAPIItem`:

```python
{
    "page_content":   str,       # текст чанка (пока = dense = sparse, см. Этап 2)
    "dense_content":  str,
    "sparse_content": str,
    "message_ids":    list[str], # id всех сообщений внутри чанка (включая overlap)
}
```

**Важно про `message_ids`:**
- Включает **все** сообщения чанка, в том числе overlap-хвост из предыдущего.
- Это означает: одно сообщение может присутствовать в `message_ids` двух соседних чанков (если оно попало в overlap) — это ожидаемо.
- Final assembly в search-service (Этап 10) дедуплицирует сообщения в итоговом выводе.

### Известные ограничения

- **Одиночные длинные сообщения не режутся.** Если одно сообщение в исходнике > `UPPER_CHARS`, чанк получается больше лимита (на Nova — max 4527 chars). Message-boundary chunking сознательно не режет внутри сообщения. Для dense embedding (Qwen3, 8192 токенов) это не проблема; для rerank — на границе ок.
- **LOWER_CHARS не гарантирован.** Финальный чанк (конец данных) может быть любого размера, даже меньше `LOWER_CHARS`. Аналогично — если hard boundary срабатывает рано.
- **`page_content` = `dense_content` = `sparse_content`.** Это исправляется в Этапе 2 (content separation).
- **Нет шаблона диалога** (`ЧАТ / ТИП / THREAD / timestamps`). Тоже Этап 2.

### Связанные этапы роадмапа

- [Этап 1](ml-roadmap.md#этап-1-chunking-высокий-impact) — текущая реализация.
- [Этап 2](ml-roadmap.md#этап-2-content-separation-page--dense--sparse) — разделить page/dense/sparse, шаблон `page_content` с разметкой диалога.
- [Этап 3](ml-roadmap.md#этап-3-ingestion-orchestrator--qdrant-payload) — стабильный `chunk_id` из `message_ids` (уже реализовано в [`eval/ingest.py`](../eval/ingest.py)).
- [Этап 9](ml-roadmap.md#этап-9-нормализация-текста) — content-aware фильтрация коротких сообщений, стоп-слова для sparse.
- [Этап 10](ml-roadmap.md#этап-10-final-assembly-критично-для-recall50) — dedup сообщений, появляющихся в нескольких чанках через overlap.

---

## Остальные этапы pipeline

Каждый этап — что меняется, зачем, и как это выглядит на том же примере «Релиз v2.1».

---

### Этап 2. Content separation — три версии одного чанка

**Что меняется.** Сейчас `page_content = dense_content = sparse_content`. Разделяем по ролям:

- `dense_content` — чистая семантика, без служебных заголовков
- `sparse_content` — keyword dump: текст + сущности + имена + даты, лексемы нормализованы
- `page_content` — структурированный диалог с timestamps, уходит в rerank

**Зачем.** Dense-модель шумит на заголовках `CHAT_ID:...`. Sparse выигрывает от дополнительных ключевиков (имён, emails). Rerank лучше понимает структуру диалога с timestamp-ами.

**На примере chunk_A** (m01, m02):

```text
dense_content:
Ребят, когда деплой v2.1? В пятницу в 18:00 планируем

sparse_content:
ребят когда деплой v2.1 пятница 18:00 планируем alice@corp bob@corp
релиз деплоить деплой v2 v2.1 18

page_content:
ЧАТ: Релиз v2.1
ТИП: group
THREAD: no_thread

[2023-11-14 10:00:00 | alice@corp]
Ребят, когда деплой v2.1?

[2023-11-14 10:02:00 | bob@corp]
В пятницу в 18:00 планируем
```

Итог: dense-вектор такой же, но **sparse теперь ловит** запрос «кто bob@corp?» даже если в оригинале текст был просто «Bob:». Rerank точнее отвечает на «когда» — видит timestamp.

---

### Этап 3. Ingestion + soft-filters по metadata

**Что меняется.** В payload уже есть `participants`, `mentions`, `time_start/end`, `contains_forward/quote`. Этап 3 — **использовать их в search** как soft-filters.

**Зачем.** `question.entities.people`, `question.date_range` — богатые сигналы от проверяющей системы. Отсечь 90% нерелевантных чанков → освободить слоты в топ-50.

**На примере.** Вопрос:

```json
{
  "text": "Когда релиз v2.1 и кто даёт доступ к проду?",
  "entities": {"people": ["carol"], "emails": []},
  "date_range": {"from": "2023-11-13", "to": "2023-11-17"}
}
```

Qdrant query получает дополнительные `should` filters:

```python
filter=Filter(should=[
    FieldCondition(key="metadata.participants", match=MatchAny(any=["carol"])),
    FieldCondition(key="metadata.mentions",     match=MatchAny(any=["carol"])),
    FieldCondition(key="metadata.time_start",
                   range=Range(gte=1699833600, lte=1700179200)),
])
```

`should`, не `must` — это **boost, не фильтр**. chunk_C (с carol) поднимется в ранжировании; chunk_A/B где carol нет — остаются, но с меньшим весом.

**Результат.** В нашем примере chunk_C с `participants=["alice@corp","carol@corp"]` получит +15-20% к RRF-score → всплывёт выше в prefetch → уверенней войдёт в топ после rerank.

---

### Этап 4. Query expansion — самый жирный win

**Что меняется.** Сейчас search использует только `question.text`. Игнорируем богатые поля: `variants`, `hyde`, `keywords`, `entities`, `search_text`.

**Зачем.** Dense-модель видит один вариант формулировки — промахивается по синонимам. Sparse не знает ключевые слова которые не в основном тексте.

**Dense — multi-query retrieval (НЕ склеивать в один вектор):**

```
primary   = search_text or text
variants  = [text_paraphrase_1, text_paraphrase_2]
hyde      = [hypothetical_answer_1, hypothetical_answer_2]

→ embed каждый отдельно → 3-5 dense векторов
→ Qdrant: prefetch для каждого → RRF объединяет
```

**Sparse — один запрос, keyword dump:**

```python
sparse_query_text = " ".join([
    search_text,
    *keywords,
    *entities.people, *entities.emails, *entities.documents,
    *date_mentions
])
```

**На примере.** Вопрос в расширенной форме:

```json
{
  "text": "Когда релиз v2.1 и кто даёт доступ к проду?",
  "search_text": "релиз v2.1 доступ прод",
  "variants": [
    "Какого числа деплой v2.1?",
    "Кто отвечает за production access к релизу"
  ],
  "hyde": [
    "Релиз v2.1 запланирован на пятницу в 18:00",
    "Доступ к production оформляет Carol, заявка подаётся за день"
  ],
  "keywords": ["релиз", "v2.1", "деплой", "доступ", "прод", "production"],
  "entities": {"people": ["carol"]}
}
```

Теперь:
- dense prefetch_1 ловит chunk_A по `search_text`
- dense prefetch_2 — chunk_A, B по variant «деплой»
- dense prefetch_3 — chunk_B, C по hyde про оформление доступа
- sparse — все 3 по keyword dump

RRF объединяет 4 списка → все релевантные чанки с высокой уверенностью.

**Impact.** Роадмап: +5-10 pp Recall@50. Самое выгодное изменение.

---

### Этап 5. Sparse модель — выбор

**Что меняется.** Сейчас `Qdrant/bm25` через fastembed. Альтернативы:

| Модель | Плюсы | Минусы |
|---|---|---|
| BM25 + pymorphy3 стемминг | Дёшево, ru-friendly | Кастомная токенизация |
| SPLADE++ multilingual | Семантический sparse | Тяжёлый, 15-мин SLA под угрозой |
| BM42 (Qdrant) | Attention-weighted | В основном английский |

**Стратегия.** Начать с BM25 + ru-стемминг → замер → SPLADE только если запас времени.

**На примере.** С BM25 запрос «доступа» не находит `chunk_B` где написано «доступ» (разные токены без стемминга). С pymorphy3 — оба токена нормализуются в `доступ` → находит.

---

### Этап 6. Retrieval & Fusion — поднять K

**Что меняется.** Текущие константы в [`search/main.py`](../search/main.py):

```python
DENSE_PREFETCH_K  = 10   →  80-100
SPARSE_PREFETCH_K = 30   →  80-100
RETRIEVE_K        = 20   →  80-120
```

**Зачем.** При K=50 ответе, ретривить 10-20 кандидатов — это потолок Recall=20/50=40%. Нужен широкий prefetch, узкое финальное сжатие реранком.

**Fusion — остаёмся на RRF, не лезем в weighted.** Rank-based → безопасно по несопоставимым скорам. Weighted `α·dense + (1-α)·sparse` требует нормализации, легко ломает ранжирование.

**На примере.** Сейчас с `DENSE_PREFETCH_K=10` мы бы потеряли chunk_B если оно на 11-м месте в dense ранге. С `K=80` — спокойно ловим.

---

### Этап 7. Rerank — top 40-60, score mixing

**Что меняется.**

1. Был баг `rerank_candidates = points[:10]` — рерank делался только по первым 10, хвост (11-20) возвращался в retrieval-порядке. **Уже пофикшено** — rerank теперь обрабатывает все переданные points.
2. Поднять вход реранкера до top 40-60 кандидатов (не больше — API лимит).
3. **Score mixing** опционально:

```python
final_score = α * rerank_score + (1 - α) * retrieval_rrf_score   # α=0.7
```

**Зачем.** Чистый rerank опасен — единичный faulty score роняет правильный ответ. Mixing страхует.

**На примере.** Допустим rerank ошибся и дал chunk_B score=0.3 (вместо 0.94). Только по rerank — chunk_B улетел в конец. С mixing: `0.7·0.3 + 0.3·0.033 ≈ 0.22` — всё ещё выше chunk_A (`0.7·0.81 + 0.3·0.033 ≈ 0.58`), но risk смягчён для edge-cases.

---

### Этап 8. Asker-aware boost

**Что меняется.** `question.asker` — автор вопроса. Сейчас не используется.

**Зачем.** Вопросы часто про свой контекст: переписку с собой, свои дела. Не фильтр — слабый boost.

**На примере.** Вопрос задан `alice@corp`. Все 3 чанка содержат её — boost мал (все равны). Но если бы был chunk_D из чужого треда без alice — этот boost его бы придавил.

**Правило:** `+5%` к RRF если `asker ∈ participants`, `+2%` если `asker` упомянут в тексте/mentions. **Не** `must`-фильтр — релевантный ответ может быть от коллеги.

---

### Этап 9. Нормализация текста — мелочи на 1-2 pp

Суммарно небольшой impact, но бесплатно:

- **Unicode NFC** — одинаковые буквы с разным представлением `é` vs `é` становятся равными
- **Lowercase для sparse** (не для dense — мешает)
- **Фильтр `is_system=true`** — шум
- **Стоп-слова для sparse** — осторожно, «кто/где/когда» в вопросах важны, в индексе чаще нет
- **Форварды/цитаты** — A/B: inline с префиксом `[fwd]` vs отдельный чанк
- **Content-aware фильтр** коротких: `v1.18`, email, ссылка — keep; «ок», «спасибо» — drop

**На примере.** Если в исходнике сообщение m07 = «+1», без content-aware фильтра оно попадёт в чанк и добавит шум. После Этапа 9 — дропается до chunking.

---

### Этап 10. Final assembly — критично для Recall@50

**Что меняется.** После rerank — **post-processing**.

```python
# 1. Flatten
flat_ids = []
for chunk in reranked_chunks:
    flat_ids.extend(chunk.message_ids)

# 2. Dedup по пересечению message_ids
#    если новый chunk покрывает msg_ids, которые уже есть >80% — выкинуть
#    оставляем лучший из near-duplicate группы

# 3. Diversity control
#    max 5-10 chunks из одного thread в top-50
#    max 5-10 chunks из одного temporal window (1ч)

# 4. Финальный список msg_ids — уникальные, 50 штук
```

**Зачем.** Без dedup overlap-чанки забивают топ дублями. Без diversity — один длинный тред захватывает все 50 слотов, игнорируя другие релевантные темы.

**На примере.** В «Пример на пальцах» мы это уже делали вручную. Формализуем:

```
chunks_reranked = [chunk_B, chunk_A, chunk_C]
chunk_B.msg_ids = [m02, m03, m04]
chunk_A.msg_ids = [m01, m02]                    # m02 overlap с B
chunk_C.msg_ids = [m04, m05, m06]               # m04 overlap с B

overlap(A, B) = |{m02}| / |{m01,m02}| = 50%     < 80%, не dedup
overlap(C, B) = |{m04}| / |{m04,m05,m06}| = 33% < 80%, не dedup

flatten + unique_preserve_order = [m02, m03, m04, m01, m05, m06]
```

Если бы overlap был 2 сообщения и chunk_A = [m01, m02, m03], chunk_B = [m02, m03, m04]:
`overlap(A, B) = 2/3 = 66%` — тоже не dedup по 80%-threshold. Порог нужно тюнить на eval.

**Impact.** Роадмап: +3-8 pp Recall@50 **без изменения retrieval/rerank**.

---

### Этап 11. Перф-оптимизации — для SLA

Если ML-улучшения вылезают за 15-мин индексации / 60-сек поиск:

1. **Async gather** dense+sparse в search (`asyncio.gather`) — −30% latency
2. **Batching dense API** 32-64 чанка в запрос (не по одному)
3. **fastembed `parallel=0, batch_size=128`** — полное использование CPU
4. **Lifespan prewarm** моделей — убрать холодный старт
5. **uvloop + httptools**
6. **`query_batch_points`** для multi-query expansion — 1 RTT вместо N
7. **1 uvicorn worker + threads** vs 8 workers — для тяжёлых моделей 1 лучше (модель одна в RAM)

**На примере.** Если вопрос с 5 dense-запросами (primary + 2 variants + 2 hyde) — наивно 5 × 200ms = 1с. С `asyncio.gather` — все параллельно, ~250ms. С `query_batch_points` в Qdrant — один RTT.

---

### Этап 12. Ablation — финальная сборка

Перед каждым сабмитом — таблица:

| Change | Recall@50 | nDCG@50 | Score | Index t | Search P95 |
|---|---|---|---|---|---|
| Baseline | 0.42 | 0.31 | 0.40 | 10m | 2s |
| + message-boundary chunking | 0.48 | 0.33 | 0.45 | 11m | 2s |
| + query expansion | 0.58 | 0.41 | 0.55 | 11m | 3s |
| + final assembly dedup | 0.64 | 0.44 | 0.60 | 11m | 3s |

**Финальное решение — только изменения с positive delta.** Не тащить всё подряд.

---

### Итоговая выдача на примере после всех этапов

Тот же вопрос «Когда релиз v2.1 и кто даёт доступ к проду?», полный pipeline:

1. **Chunking (Этап 1)** — 3 чанка, overlap=2
2. **Content separation (Этап 2)** — sparse ловит `carol@corp`, page_content структурирован
3. **Query expansion (Этап 4)** — 3-5 dense запросов, sparse keyword dump
4. **Soft-filter (Этап 3)** — boost чанкам с `participants=[carol]`
5. **Retrieval K=100 (Этап 6)** — все 3 чанка легко проходят, плюс запасной шум
6. **Rerank top 40-60 (Этап 7)** — пересортировка, chunk_B топ
7. **Final assembly (Этап 10)** — dedup + diversity → `[m02, m03, m04, m01, m05, m06]`
8. **Asker boost (Этап 8)** — alice всюду, тут без эффекта

Метрики: `Recall=1.0, nDCG≈0.85, Score=0.97` — тот же результат что в изначальном примере, **но теперь это устойчиво**: если реранкер сошёл с ума, query expansion компенсирует; если один dense-запрос промахнулся, другие вытянут; если overlap дал дубли, dedup их уберёт.

Вот это и есть pipeline целиком.
