# ML Roadmap — Hackathon Search Engine

Полный план работ по ML-составляющей проекта. Бэкенд, Docker, деплой — **не трогаем**. Только качество поиска: chunking, embeddings, retrieval, rerank, Qdrant.

---

## Метрика и ориентиры

Итоговая метрика оценки:

```
score = recall_avg * 0.8 + ndcg_avg * 0.2
```

Где `K = 50` (если вернём >50 `message_id`, хвост отбрасывается).

**Вывод:** Recall доминирует. Стратегия — максимизировать покрытие правильных сообщений в топ-50, nDCG улучшать рерankом.

**Ограничения:**
- Индексация всего датасета: ~15 мин (SLA 20 мин)
- Ответ на 1 вопрос: ≤60 сек
- Ресурсы: 4 CPU / 7 GB RAM на контейнер
- Нет интернета в index/search — всё локально или через проксирующие dense/rerank API

---

## Этап 0. Eval harness (блокер всего)

**Зачем.** Без локальной метрики любые изменения — вслепую. Нельзя понять, помогло или ухудшило.

**Что делать:**
- Написать скрипт, прогоняющий `data/Go Nova.json` + test-вопросы через локально запущенные `index` + `search`
- Считать Recall@50 и nDCG@50, усреднять по вопросам
- Логировать per-question промахи: какой `message_id` правильный, какие вернулись, на каком этапе он потерян (retrieval / rerank / chunking)
- Фиксировать baseline текущего шаблона как reference score

**Результат.** Есть цифра, с которой сравнивать каждое следующее изменение. Плюс — видны категории ошибок (по типу вопроса, по чату, по длине).

---

## Этап 1. Chunking (высокий impact)

**Проблема сейчас.** Текущий `build_chunks` режет текст по 512 символов безотносительно границ сообщений. В итоге:
- Одно сообщение может разорваться пополам
- Чанк может начинаться с середины фразы → dense embedding шумный
- `message_ids` у чанка включает сообщения, от которых в тексте остался огрызок → retrieval даст чанк, но правильный msg_id не попадёт

**Что делать.**

1. **Чанкинг по message-boundaries.** Группировать целые сообщения в чанк (N сообщений, например N=5–10). Overlap 1–2 сообщения.
2. **Учёт `parts[*].text`.** Во многих сообщениях значимая часть в `parts` (цитаты, форварды), а не в `text`. Сейчас это частично есть, но игнорятся типы частей. Сохранять целиком с префиксами типа `[цитата]`, `[форвард]`.
3. **Thread-aware chunking.** Если есть `thread_sn` — чанки формировать внутри треда, не мешая с другими сообщениями того же чата.
4. **Time-window alternative.** Вместо фиксированного N — окно T минут (реплики быстрого обмена — в один чанк, паузы — разделитель).
5. **Размер чанка.** A/B: 3 / 5 / 10 сообщений, overlap 1 / 2.

**Метрика.** Recall@50 на hold-out вопросах.

---

## Этап 2. Separation of page_content / dense_content / sparse_content

**Проблема.** Сейчас все три поля равны. Это упущение — у dense и sparse разная природа.

- **Dense** ловит семантику, перефразы, синонимы. Ему полезны нормализованный текст + контекст (имя чата, участники).
- **Sparse** (BM25/SPLADE) ловит точные совпадения ключевых терминов. Ему полезны редкие имена, emails, названия продуктов.
- **page_content** — это то, что видит реранкер и то, что возвращается как payload. Пусть будет оригиналом.

**Что делать.**

```
page_content    = оригинал чанка (для rerank input)
dense_content   = нормализованный текст + chat_name + participants (семантический контекст)
sparse_content  = текст + mentions + entities.names + emails + file_snippets
```

**Гипотеза.** Sparse будет ловить вопросы вида *"кто руководитель команды X"* (редкие названия), dense — *"что обсуждали про релиз"* (перефразы). Разделение усилит оба.

---

## Этап 3. Payload metadata + фильтры в Qdrant

**Проблема.** Сейчас в payload чанка сохраняются базовые поля, но они не используются для фильтрации в `/search`.

**Что сохранять в payload чанка:**
- `participants` — уникальные `sender_id` в чанке
- `mentions` — упоминания
- `chat_name`, `chat_type`, `chat_sn`, `chat_id`
- `time_start`, `time_end` (unix timestamps) — min/max time сообщений
- `contains_forward`, `contains_quote`
- `thread_sn` (если есть)

**Что использовать в search как фильтр:**
- `question.entities.people` / `question.entities.emails` → фильтр `should match` по participants/mentions
- `question.date_range.from` / `to` → фильтр по `time_start`/`time_end`
- `question.entities.names` → может быть дополнительной фильтрацией по chat_name

**Фильтры применять как `should` (soft), не `must`,** чтобы не выкинуть правильные чанки при ошибках в entity extraction. Либо `must` только если есть сильный сигнал (точный email).

**Impact.** На вопросах про конкретного человека/дату — отсекает 90% нерелевантных чанков, освобождая место в топ-50 для правильных.

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

Всё это **бесплатный сигнал от проверяющей системы** — надо использовать.

**Что делать.**

1. **Dense query** (вариант A — одиночный запрос):
   ```
   dense_query = embed(text + " " + search_text + " ".join(variants[:2]))
   ```

2. **Dense query** (вариант B — multi-query + fusion):
   - embed каждый из `[text, *variants, *hyde]`
   - N отдельных prefetch в Qdrant
   - RRF/DBSF fusion результатов
   - **Дороже по rate limit dense API, но лучше по качеству**

3. **Sparse query:**
   ```
   sparse_query = sparse(text + " " + " ".join(keywords) + " ".join(entities.names) + " ".join(entities.people))
   ```
   Sparse очень любит keyword dumping.

4. **HyDE полезен** когда вопрос абстрактный, а ответ конкретный. Dense embed гипотетического ответа ближе к реальному сообщению, чем embed вопроса.

**Impact.** На моих оценках подобных pipeline — обычно +5–10 pp Recall@50. Самое выгодное изменение.

---

## Этап 5. Sparse модель

**Сейчас.** `Qdrant/bm25` через fastembed. Простой, быстрый, но:
- Не учитывает морфологию русского (релиз / релиза / релизу — разные токены)
- Не семантический — синонимы не ловятся

**Альтернативы.**

| Модель | Плюсы | Минусы |
|---|---|---|
| **BM25 + pymorphy3 стемминг** | Дёшево, ru-friendly | Ручная токенизация, нужен кастом |
| **BM42 (Qdrant)** | Attention-weighted, дешевле SPLADE | Только английский в основном |
| **SPLADE++ multilingual** | Семантический sparse, лучший retrieval | Тяжёлый, медленный, 15-мин лимит под угрозой |
| **Naver/splade-v3-distilbert** | Компромисс по весу | Английский |

**Стратегия.** Начать с BM25 + ru-стемминг. Если есть запас по времени индексации — попробовать SPLADE на части датасета, сравнить качество и время.

**Важно.** Sparse-модель упаковывается внутрь Docker-образа (нет интернета). Проверить размер модели.

---

## Этап 6. Retrieval & Fusion

**Текущие K.** `DENSE_PREFETCH_K=10`, `SPARSE_PREFETCH_K=30`, `RETRIEVE_K=20` — **всё мало**.

**Что поменять.**

- `DENSE_PREFETCH_K`  10 → **100**
- `SPARSE_PREFETCH_K` 30 → **100**
- `RETRIEVE_K` 20 → **100** (на вход rerank)
- Возвращать из search **ровно 50** (максимум по метрике)

**Fusion.**

Сейчас — RRF (Reciprocal Rank Fusion). Альтернативы:
- **DBSF** (Distribution-Based Score Fusion) — нормализует score по распределению, часто лучше RRF на гетерогенных источниках
- **Weighted score fusion** — `α·dense_score + (1-α)·sparse_score` с нормализацией

A/B на eval harness. RRF — хороший дефолт, DBSF может дать +1–2 pp.

**Multi-vector late interaction (опционально).** Добавить ColBERT как третий prefetch — если позволяет время/ресурсы. Даёт прирост на редких запросах.

---

## Этап 7. Rerank

**Проблема (баг).** Сейчас код:
```python
rerank_candidates = points[:10]       # рерankит только 10
# возвращает points (где 20), с перемешанным топом и неотсортированным хвостом
```

То есть топ-10 реранкается, а позиции 11–20 — как retrieval их выдал. Это **минус nDCG напрямую**.

**Что делать.**

1. Рерankать **топ-50** (совпадает с K метрики)
2. Возвращать все 50 отсортированными по rerank score
3. Не брать 100+ (rerank API имеет лимиты по размеру text_2, плюс время)

**Вход реранкера.**
- Текущее: `rerank_input = question.text`
- Попробовать: `question.text + " " + " ".join(keywords)` — усиливает key terms

**Score mixing.** Вместо полного переопределения порядка реранком:
```
final_score = α * rerank_score + (1 - α) * retrieval_score
```
α=0.7 обычно хорошо. Защищает от единичных ошибок реранкера.

**Важно про rate limit.** Reranker API тоже лимитирован. Один rerank на вопрос — ок, но если захотим multi-query rerank — считать.

---

## Этап 8. Нормализация текста

Мелкие вещи, которые в сумме дают 1–2 pp:

- Unicode NFC нормализация
- Lowercasing (для sparse, не для dense)
- Фильтрация системных сообщений (`is_system=true`) из индекса — шум
- Стоп-слова ru для sparse
- Форварды/цитаты: inline в чанк с префиксом `[fwd] ...` — A/B vs отдельный чанк
- Очистка `file_snippets` от служебных токенов

---

## Этап 9. Перф-оптимизации (для укладки в SLA)

Если улучшения качества вылезают за 15-минутный лимит индексации:

1. **Async gather dense+sparse** в search (`asyncio.gather`) — −30% latency per query
2. **Batching dense API** при индексации — один запрос на 32–64 чанка, не по одному
3. **Fastembed `parallel=0`, `batch_size=128`** — полное использование CPU
4. **Lifespan prewarm** моделей — убрать холодный старт первого запроса
5. **uvloop + httptools** для uvicorn
6. **Qdrant `query_batch_points`** если делаем multi-query expansion
7. **1 uvicorn worker + threads=4** vs **8 workers** — замерить (для тяжёлых моделей — 1 worker лучше, модель одна в RAM)
8. **INT8-квантизация** sparse-модели, если доступно в fastembed

---

## Этап 10. Ablation & финальная сборка

Для каждого изменения фиксировать:

| Change | Recall@50 | nDCG@50 | Score | Index time | Search P95 |
|---|---|---|---|---|---|
| Baseline | ... | ... | ... | ... | ... |
| + message-boundary chunking | ... | ... | ... | ... | ... |
| + query expansion | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

**Финальное решение — только изменения с positive delta.** Не тащить всё подряд.

---

## Приоритет (ROI)

Ранжировано по «impact / риск / время внедрения»:

| # | Изменение | Impact | Риск | Время |
|---|---|---|---|---|
| 1 | Eval harness | Blocker | — | 2–4 ч |
| 2 | Rerank топ-50, вернуть 50 | High | Low | 10 мин |
| 3 | Prefetch K = 100 | High | Low | 5 мин |
| 4 | Query expansion (variants + hyde + keywords) | **Highest** | Medium | 2–3 ч |
| 5 | Chunking по сообщениям | High | Medium | 3–5 ч |
| 6 | Dense/sparse separation + entities в sparse | Medium-High | Low | 1–2 ч |
| 7 | Payload metadata + фильтры (people/date_range) | Medium-High | Medium | 2–4 ч |
| 8 | Async gather + batching dense | Medium | Low | 1 ч |
| 9 | Fusion tuning (RRF vs DBSF) | Medium | Low | 1 ч |
| 10 | Sparse модель (BM25→BM42/SPLADE) | Medium | **High** (время индексации) | 4–8 ч |
| 11 | Нормализация текста | Low-Medium | Low | 2 ч |

**План первых 2 дней.** 1 → 2 → 3 → 4 → 5. Этого достаточно чтобы выжать ~70% возможного выигрыша.

**План следующих дней.** 6 → 7 → 8 → 9. Замерять после каждого.

**Если осталось время.** 10, 11.

---

## Что не делаем

- Не меняем контракты `/index`, `/sparse_embedding`, `/search` (запрещено ТЗ)
- Не используем сторонние LLM/embedding API (запрещено ТЗ)
- Не тащим интернет в контейнеры (запрещено ТЗ)
- Не усложняем архитектуру ради изящества — только то, что двигает метрику
