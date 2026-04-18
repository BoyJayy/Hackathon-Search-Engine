# ML Roadmap — Hackathon Search Engine

Полный план работ по ML-составляющей проекта. Бэкенд, Docker, деплой — **не трогаем**. Только качество поиска: chunking, embeddings, retrieval, rerank, Qdrant.

---

## Метрика и ориентиры

Итоговая метрика оценки:

```text
score = recall_avg * 0.8 + ndcg_avg * 0.2
```

Где `K = 50` (если вернём >50 `message_id`, хвост отбрасывается).

**Вывод:** Recall доминирует. Стратегия — максимизировать покрытие правильных сообщений в топ-50, nDCG улучшать рерankом и diversity control.

**Ограничения:**
- Индексация всего датасета: ~15 мин (SLA 20 мин)
- Ответ на 1 вопрос: ≤60 сек
- Ресурсы: 4 CPU / 7 GB RAM на контейнер
- Нет интернета в index/search — всё локально или через проксирующие dense/rerank API

---

## Архитектура pipeline

Полная индексация — это не только `/index`. Нужно явно разделить ответственности:

```
[1] index-service           -> chunk semantics only (text → chunks)
                               → отдаёт page_content / dense_content / sparse_content / message_ids

[2] ingestion orchestrator  -> точки для Qdrant
                               → dense embed (внешний API, батчи)
                               → sparse embed (/sparse_embedding)
                               → stable chunk_id = hash(chat_id, thread_sn, first_msg_id, last_msg_id, content)
                               → payload с metadata
                               → upsert в Qdrant

[3] search-service          -> query builder + hybrid retrieval + rerank + final assembly
```

**Почему это важно.** Частый скрытый провал: сильный `/index`, но корявый ingestion → upsert с нестабильными id → дубли → search кажется слабым, но проблема раньше. Зафиксировать схему и id — на старте.

**Схема point в Qdrant:**

```json
{
  "id": "<stable_hash>",
  "vector": {"dense": [...], "sparse": {"indices": [...], "values": [...]}},
  "payload": {
    "page_content": "...",
    "metadata": {
      "chat_name": "...",
      "chat_type": "...",
      "chat_id": "...",
      "chat_sn": "...",
      "thread_sn": "...",
      "message_ids": ["..."],
      "time_start": "...",
      "time_end": "...",
      "participants": ["..."],
      "mentions": ["..."],
      "contains_forward": false,
      "contains_quote": false
    }
  }
}
```

---

## Этап 0. Eval harness (блокер всего)

**Зачем.** Без локальной метрики любые изменения — вслепую. Нельзя понять, помогло или ухудшило.

**Что делать:**
- Скрипт, прогоняющий данные + вопросы через `index` + `search`
- Recall@50, nDCG@50, итоговый score
- Per-question логирование: какие msg_id потеряны и на каком этапе (retrieval / rerank / dedup)
- Фиксировать baseline текущего шаблона

**Реализация.** См. [`eval/`](../eval/) + [`docs/eval-harness.md`](eval-harness.md).

**Результат.** Есть цифра, с которой сравнивать каждое следующее изменение. Плюс — видны категории ошибок.

---

## Этап 1. Chunking (высокий impact)

**Проблема сейчас.** Текущий `build_chunks` режет текст по 512 символов безотносительно границ сообщений. В итоге:
- Одно сообщение может разорваться пополам
- Чанк может начинаться с середины фразы → dense embedding шумный
- `message_ids` у чанка включает сообщения, от которых в тексте остался огрызок

**Что делать.**

1. **Нормализация messages** перед chunking: извлечь `text + parts[*].text + file_snippets`, помечать `is_forward/is_quote`, выделить mentions.
2. **Фильтрация мусора** (осторожная — см. ниже).
3. **Стабильная сортировка:** `(time, id)` — воспроизводимость.
4. **Чанкинг по message-boundaries.** Группировать целые сообщения, не символы. Три оси:
   - **thread-aware:** если `thread_sn` — в одном чанке должны быть сообщения одного треда
   - **time-aware:** большой gap по времени → разрыв чанка
   - **size-aware:** контроль верхней/нижней границы
5. **Overlap:** 1–2 сообщения или 15–20% размера (больше = раздувает индекс + дубли в топе).
6. **Целевые размеры** (ориентир):
   - нижняя: 80–120 токенов
   - рабочая: 150–350
   - верхняя: 450–550
   - исключения: короткое, но ценное сообщение (версия, ссылка, имя) — ок само по себе

### Фильтрация мусора — осторожно

**Жёстко убирать:**
- `is_hidden == true`
- Полностью пустой текст и пустой file_snippet
- Чисто системные сообщения (`member_event` без полезного контента)

**Осторожно убирать** (не тупо по длине):
- Короткие "ок", "да", "спасибо" — шум
- **НО** короткое сообщение с конкретикой (ссылка, email, версия `v1.18`, дата, номер релиза) — **ценно, оставить**
- forward/quote без нового текста — можно дроп, но чаще внутри есть цитируемый текст, который нужен

Эвристика: не по `len(text) < N`, а по содержимому — есть ли URL, email, цифра, upper-case имя.

---

## Этап 2. Content separation (page / dense / sparse)

**Проблема.** Сейчас все три поля равны. У каждого — своя роль.

| Поле | Роль | Что кладём |
|---|---|---|
| `page_content` | Вход реранкера + возвращается в payload | Структурированный текст с контекстом диалога |
| `dense_content` | Вход dense модели | Чистый семантический текст |
| `sparse_content` | Вход sparse модели | Keyword-heavy dump |

### `page_content` — шаблон

Реранкеру полезно видеть структуру диалога, не голый текст:

```text
ЧАТ: {chat.name}
ТИП: {chat.type}
CHAT_ID: {chat.id}
THREAD: {thread_sn or "no_thread"}

[2025-10-20 14:31:00 | alice@corp]
Первое сообщение чанка.

[2025-10-20 14:32:10 | bob@corp]
Ответ на первое.

Вложения:
...
```

Это **напрямую растит nDCG** — реранкер лучше понимает, где вопрос, где ответ, кто с кем говорит.

### `dense_content`

Оптимизирован под семантику:
- Основной текст сообщений
- `file_snippets`
- Упоминания имён / сущностей
- **Без** служебных заголовков (`CHAT_ID`), которые засоряют эмбеддинг

### `sparse_content`

Keyword dump:
- Исходный текст
- `file_snippets`
- Имена, emails, названия документов, продуктов, систем
- Ссылки, даты
- (опц.) нормализованные формы слов

**Гипотеза.** Sparse будет ловить *"кто руководитель команды X"* (редкие названия), dense — *"что обсуждали про релиз"* (перефразы).

---

## Этап 3. Ingestion orchestrator + Qdrant payload

**Stable chunk ID.** Вместо `uuid.uuid4()` — детерминированный хэш:

```python
chunk_id = sha256(
    f"{chat_id}|{thread_sn or ''}|{message_ids[0]}|{message_ids[-1]}|{content_hash}"
).hexdigest()[:32]
```

Upsert станет идемпотентным, повторная индексация не плодит дубли.

**Payload metadata** (каждого чанка):
- `participants` — уникальные `sender_id`
- `mentions` — упоминания
- `chat_name`, `chat_type`, `chat_sn`, `chat_id`
- `time_start`, `time_end` (unix)
- `contains_forward`, `contains_quote`
- `thread_sn`

**Фильтры в `/search`:**
- `question.entities.people` / `emails` → `should match` по participants/mentions
- `question.date_range.from/to` → range по `time_start`/`time_end`
- `question.entities.names` → дополнительный `should match` по chat_name

**Режим `should`, не `must`** — soft-filters. Жёсткий `must` только если сигнал заведомо точный (email с `@`, ISO-дата в `date_range`). Иначе ошибки extraction выкинут правильные чанки.

**Impact.** На вопросах про конкретного человека/дату — отсекает 90% нерелевантных чанков, освобождая топ-50.

---

## Этап 4. Query expansion — самый жирный win

**Проблема.** `question` приходит с богатыми полями, но текущий search использует только `question.text`:

```python
variants    # переформулировки вопроса
hyde        # гипотетические ответы (HyDE)
keywords    # ключевые слова
entities    # people, emails, documents, names, links
date_mentions, date_range
search_text # оптимизированная под полнотекст версия
```

### Dense query

**Primary:** `search_text` если не пустой, иначе `text`.

**HyDE — отдельными запросами, не в один вектор.**
Наивно: `embed(text + hyde[0] + hyde[1] + ...)` — получаем размытый вектор, теряем семантику.
Правильно: **multi-query retrieval**:
```
prefetch_1 = retrieve(embed(primary))
prefetch_2 = retrieve(embed(hyde[0]))
prefetch_3 = retrieve(embed(variants[0]))
...
fusion через RRF
```

Это дороже по rate limit dense API, но качество сильно выше.

**Если rate limit жмёт:** оставить primary + 1–2 лучших variant, HyDE опционально.

### Sparse query

Sparse любит keyword dumping — тут можно всё в один:

```python
sparse_query = sparse(
    search_text + " " +
    " ".join(keywords) + " " +
    " ".join(entities.names) + " " +
    " ".join(entities.people) + " " +
    " ".join(entities.emails) + " " +
    " ".join(entities.documents) + " " +
    " ".join(date_mentions)
)
```

### Метаданные запроса

- `date_range` → фильтр (soft) на retrieval этапе
- `asker` → слабый boost для чанков где `sender_id == asker` или asker упоминается в тексте/mentions (см. Этап 8)
- `asked_on` — полезен только если `date_range` пустой (нормализует "вчера", "на прошлой неделе")

**Impact.** Обычно +5–10 pp Recall@50. Самое выгодное изменение.

---

## Этап 5. Sparse модель

**Сейчас.** `Qdrant/bm25` через fastembed. Простой, быстрый, но:
- Не учитывает морфологию русского (релиз / релиза / релизу — разные токены)
- Не семантический — синонимы не ловятся

**Альтернативы.**

| Модель | Плюсы | Минусы |
|---|---|---|
| **BM25 + pymorphy3 стемминг** | Дёшево, ru-friendly | Ручная токенизация, кастом |
| **BM42 (Qdrant)** | Attention-weighted | Только английский в основном |
| **SPLADE++ multilingual** | Семантический sparse | Тяжёлый, 15-мин лимит под угрозой |
| **naver/splade-v3-distilbert** | Компромисс по весу | Английский |

**Стратегия.** Начать с BM25 + ru-стемминг. Если запас по времени индексации — попробовать SPLADE на части датасета, сравнить качество vs время.

**Важно.** Sparse-модель упаковывается внутрь Docker-образа (нет интернета). Проверить размер.

---

## Этап 6. Retrieval & Fusion

**Текущие K.** `DENSE_PREFETCH_K=10`, `SPARSE_PREFETCH_K=30`, `RETRIEVE_K=20` — **всё мало**.

**Что поменять (стартовая точка):**
- `DENSE_PREFETCH_K` 10 → **80–100**
- `SPARSE_PREFETCH_K` 30 → **80–100**
- `RETRIEVE_K` 20 → **80–120** (на вход rerank)
- После rerank + final assembly возвращать **50**

### Fusion — RRF first, не спеши на weighted

**Стартовый вариант:** RRF (уже есть в baseline). Rank-based — score-scale безразличен.

**НЕ делать сразу:** `α·dense + (1-α)·sparse` weighted fusion. Score разных источников (cosine vs BM25) несопоставимы. Нужна нормализация — легко сломать ранжирование.

**План:**
1. RRF как baseline → замер
2. Только если eval покажет улучшение — пробовать DBSF / нормализованный weighted
3. Иначе — остаёмся на RRF

### Multi-vector late interaction (опционально)

ColBERT как третий prefetch — даёт прирост на редких запросах. Только если запас времени индексации и места в памяти.

---

## Этап 7. Rerank

**Проблема (баг сейчас).** Код:
```python
rerank_candidates = points[:10]       # рерankит только 10
# возвращает 20, но позиции 11–20 — в retrieval-порядке, не rerank
```

Топ-10 переранжируется, хвост — нет. **Прямой минус nDCG.**

**Что делать.**

1. Рерankать **топ 40–60 кандидатов** (из prefetch 80–120). Не 100+ — rerank API лимиты по `text_2`, плюс латенси.
2. Возвращать все 50 отсортированными по rerank score.
3. После rerank → final assembly (Этап 10).

**Вход реранкера:**
- Query: `search_text` если есть, иначе `text`. Плюс 1–2 важных keyword — аккуратно, чтобы не сбить.
- Document: `page_content` (полный контекст с шаблоном).

**Score mixing (опция).** Полная замена порядка реранком опасна — единичная ошибка реранкера роняет результат.

```text
final_score = α * rerank_score + (1 - α) * retrieval_score
```
α=0.7 обычно ок.

**Про rate limit.** Один rerank на вопрос = ок. Multi-query rerank — считать.

---

## Этап 8. Asker-aware boost (слабый)

Поле `question.asker` — автор вопроса. Бесплатный сигнал.

**Когда полезен:**
- Вопрос про самого автора
- Переписка с его участием
- Документы/темы, связанные с ним

**Как применять:**
- **Слабый boost** (не фильтр!) для чанков где `sender_id == asker`
- Slabsh boost для чанков где email/имя asker встречается в тексте или mentions

**Чего не делать.** Не превращать в `must` — релевантный результат может быть написан другим человеком.

---

## Этап 9. Нормализация текста

Мелкие вещи, суммарно 1–2 pp:

- Unicode NFC
- Lowercasing для sparse (не для dense — мешает)
- Фильтр системных сообщений (`is_system=true`) — шум
- Стоп-слова ru для sparse (осторожно — "кто", "где", "когда" в вопросах важны, в индексе чаще нет)
- Форварды/цитаты: inline в чанк с префиксом `[fwd]` — A/B vs отдельный чанк
- Очистка `file_snippets` от служебных токенов

---

## Этап 10. Final assembly (критично для Recall@50)

После rerank работа не закончена. Нужен отдельный post-processing.

### Dedup по message_ids

Retrieval + rerank часто возвращают **пересекающиеся чанки** (overlap делает своё). Это убивает Recall — топ забит дубликатами вместо разнообразных правильных сообщений.

**Что делать:**
- Dedup по пересечению `message_ids`: если новый chunk покрывает msg_ids, которые уже есть в выдаче >80% — выкинуть
- Оставить лучший из набора near-duplicate overlap-чанков

### Diversity control

Не давать одному thread / временному окну занять весь топ:
- Ограничить: max N=5–10 чанков из одного thread в top-50
- Ограничить: max N=5–10 чанков из одного temporal window (например, 1 час)

### Финальный список message_ids

После dedup — собрать message_ids из чанков в порядке ранжирования, сохраняя **уникальность по msg_id** (а не по чанку).

По ТЗ формат ответа — один элемент `results` с `message_ids: [...]`. Ограничить 50 уникальными msg_ids.

**Impact.** Может добавить 3–8 pp Recall@50 **без изменения retrieval/rerank** — чисто за счёт того, что в слоты топ-50 больше не впихиваются дубли.

---

## Этап 11. Перф-оптимизации (для SLA)

Если улучшения качества вылезают за 15-минутный лимит индексации:

1. **Async gather dense+sparse** в search (`asyncio.gather`) — −30% latency per query
2. **Batching dense API** при индексации — один запрос на 32–64 чанка, не по одному (max_model_len=8192, влезет)
3. **Fastembed `parallel=0`, `batch_size=128`** — полное использование CPU
4. **Lifespan prewarm** моделей — убрать холодный старт
5. **uvloop + httptools** для uvicorn (+10–20%)
6. **Qdrant `query_batch_points`** для multi-query expansion — один RTT вместо N
7. **1 uvicorn worker + threads=4** vs **8 workers** — замерить (для тяжёлых моделей 1 лучше, модель одна в RAM)
8. **INT8-квантизация** sparse-модели, если доступно в fastembed

---

## Этап 12. Ablation & финальная сборка

Для каждого изменения фиксировать:

| Change | Recall@50 | nDCG@50 | Score | Index time | Search P95 |
|---|---|---|---|---|---|
| Baseline | ... | ... | ... | ... | ... |
| + message-boundary chunking | ... | ... | ... | ... | ... |
| + query expansion (variants) | ... | ... | ... | ... | ... |
| + HyDE multi-query | ... | ... | ... | ... | ... |
| + final assembly dedup | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

**Финальное решение — только изменения с positive delta.** Не тащить всё подряд.

---

## Safe baseline path — путь к первому сильному сабмиту

Если времени мало, а нужен сразу рабочий результат (не игрушка), делать в **таком порядке**:

### Шаг 1. Переписать `/index`
- Нормализация сообщений (text + parts + file_snippets)
- Осторожная фильтрация мусора
- Thread/time-aware chunking
- Разные `page_content / dense_content / sparse_content`
- Корректные `message_ids`

### Шаг 2. Зафиксировать Qdrant schema
- Stable chunk_id (hash, не uuid)
- Payload metadata (participants, mentions, time, thread_sn, contains_forward/quote)
- Upsert идемпотентный

### Шаг 3. Hybrid search без переусложнения
- Primary dense query = `search_text` (иначе `text`)
- Sparse query = `search_text + keywords + entities.* + date_mentions`
- Fusion = **RRF** (не weighted — пока)
- Prefetch 80–100 / 80–100, retrieve 80–120
- Rerank top 40–60
- Final assembly: dedup по message_ids, diversity по thread

### Шаг 4. Первый сабмит. Замер.

### Шаг 5. Только потом — бусты
- `variants` expansion
- HyDE multi-query
- `date_range` soft filter
- Entity boosts
- Asker boost
- Weighted fusion / DBSF — если RRF как baseline измерен
- SPLADE вместо BM25 — если есть запас по времени

Каждое изменение — один замер, одна строка в ablation.

---

## Приоритет (ROI)

Ранжировано по «impact / риск / время внедрения»:

| # | Изменение | Impact | Риск | Время |
|---|---|---|---|---|
| 1 | Eval harness | Blocker | — | 2–4 ч |
| 2 | Stable chunk_id + upsert schema | High | Low | 1 ч |
| 3 | Rerank топ-50, вернуть 50 | High | Low | 10 мин |
| 4 | Prefetch K = 80–100 | High | Low | 5 мин |
| 5 | Final assembly (dedup + diversity) | **Highest** | Low | 2 ч |
| 6 | Query expansion (variants + keywords) | **Highest** | Medium | 2–3 ч |
| 7 | Chunking по message-boundaries + page_content шаблон | High | Medium | 4–6 ч |
| 8 | Dense/sparse separation + entities в sparse | Medium-High | Low | 1–2 ч |
| 9 | Payload metadata + soft filters (people / date_range) | Medium-High | Medium | 2–4 ч |
| 10 | HyDE multi-query retrieval | Medium-High | Medium | 2 ч |
| 11 | Async gather + batching dense | Medium | Low | 1 ч |
| 12 | Asker-aware boost | Low-Medium | Low | 1 ч |
| 13 | RRF → tune fusion (DBSF / weighted) | Medium | Medium | 2 ч |
| 14 | Sparse модель (BM25 → BM42 / SPLADE) | Medium | **High** (время индексации) | 4–8 ч |
| 15 | Нормализация текста | Low-Medium | Low | 2 ч |

**Первые 2 дня.** 1 → 2 → 3 → 4 → 5 → 6 → 7. ~70% выигрыша.

**Следующие дни.** 8 → 9 → 10 → 11.

**Если осталось время.** 12 → 13 → 14 → 15.

---

## Скрытые риски

Типовые грабли, на которых сыпется решение даже при сильном ML:

### Сильный `/index`, но слабый ingestion
- Не делается upsert / делается с uuid-id (дубли)
- Плохой payload — search не может фильтровать
- Результат: search кажется слабым, но проблема раньше

### Сильный retrieval, но плохой final assembly
- Топ-50 забит overlap-дублями
- Повторяющиеся `message_ids`
- Один thread забирает 40/50 слотов
- Результат: recall падает несмотря на сильный retrieval

### Переусложнение слишком рано
- Сложные boosts до замера baseline
- Weighted fusion до RRF baseline
- Смешивание всех полей `question` без контроля
- Результат: хаос в метриках, не понять что помогло

### Недооценка enriched question
- Игнор `search_text`, `keywords`, `entities`, `date_range`
- Это **бесплатный сигнал** от проверяющей системы — единственное в пайплайне, что "уже сделано за нас". Не использовать — прямая потеря скора.

### HyDE в один вектор
- `embed(text + " " + " ".join(hyde))` размывает семантику
- Только multi-query retrieval с fusion

### Фильтр по длине сообщений
- Просто `len(text) < 10 → drop` — потеряешь ценные короткие: `v1.18`, email, ссылку, имя. Нужна content-aware эвристика.

---

## Что не делаем

- Не меняем контракты `/index`, `/sparse_embedding`, `/search` (запрещено ТЗ)
- Не используем сторонние LLM/embedding API (запрещено ТЗ)
- Не тащим интернет в контейнеры (запрещено ТЗ)
- Не усложняем архитектуру ради изящества — только то, что двигает метрику
- Не тюним 10 вещей за один прогон — одно изменение → замер → решение
