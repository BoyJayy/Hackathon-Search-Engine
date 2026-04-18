# Eval Harness — Этап 0

Локальный замер `Recall@50`, `nDCG@50` и итогового `score`. Нужен, чтобы любое изменение (chunking, prefetch, query expansion) проверялось числом, а не на глаз.

## Важное ограничение

Ground-truth Q&A датасет локально **не предоставлен** организаторами — только `data/Go Nova.json` (чат без вопросов/ответов). Абсолютная метрика считается только в проверяющей системе при сдаче.

Локально `eval/run.py` работает, если передать свой JSONL-датасет формата `{id, question, answer.message_ids}` через `--dataset path.jsonl`. Полезно для smoke-теста pipeline и замера relative-delta между изменениями.

## Файлы

- [`eval/metrics.py`](../eval/metrics.py) — Recall@K, nDCG@K, score
- [`eval/ingest.py`](../eval/ingest.py) — разово заполняет Qdrant через index-сервис + dense API
- [`eval/run.py`](../eval/run.py) — прогоняет датасет через search, печатает метрики

## Требования

- Docker compose поднят (`qdrant`, `index`, `search`)
- Переменные: `OPEN_API_LOGIN`, `OPEN_API_PASSWORD` (Basic Auth к внешнему dense/rerank API)
- Python 3.14 + `pip install -r eval/requirements.txt`

## Запуск

```bash
# 1. Поднять инфру
export OPEN_API_LOGIN=...
export OPEN_API_PASSWORD=...
docker compose up --build -d

# 2. Заполнить Qdrant (один раз после изменений в index/)
python eval/ingest.py

# 3. Замер (датасет — обязателен)
python eval/run.py --dataset path/to/dataset.jsonl --verbose
```

## Workflow при работе над качеством

```bash
# baseline
python eval/run.py --dataset eval/dataset.jsonl > results/baseline.txt

# изменил chunking → переиндексировал → замерил
docker compose restart index
python eval/ingest.py     # перегрузить чанки в Qdrant (id стабилен по hash — upsert перезаписывает)
python eval/run.py --dataset eval/dataset.jsonl > results/after_chunking.txt

# сравнить
diff results/baseline.txt results/after_chunking.txt
```

**После каждого изменения — записывай `score` в таблицу ablation** (см. `docs/ml-roadmap.md` Этап 10).

## Что замеряется

```text
score = 0.8 * recall_avg + 0.2 * ndcg_avg
```

Где `recall_avg` и `ndcg_avg` усреднены по всем вопросам датасета. `K=50` по умолчанию (можно `--k 10` для диагностики rerank).

## Расширение датасета

Посмотри `data/Go Nova.json`, найди тематический кусок переписки, сформулируй вопрос и выпиши `id` сообщений, которые его подтверждают:

```jsonl
{"id": "qN", "question": {"text": "..."}, "answer": {"message_ids": ["3888...", "3999..."]}}
```

Можно заполнять другие поля `question` (`variants`, `hyde`, `keywords`, `entities`, `search_text`) — они прокинутся в search и сэмулируют реальный вход системы оценки.

## Ограничения текущей реализации

- Пересчитываем dense на каждом `ingest.py` → тратит rate limit внешнего API. Можно кешировать `dense_content → vector` локально, если станет проблемой.
- Rerank API зовётся из search, его тоже расходуем на каждый прогон eval. При активной итерации — делать `--verbose` и бить небольшие куски датасета.
- Текущий baseline index-сервис **не пушит в Qdrant**, поэтому без `eval/ingest.py` поиск вернёт пусто. Это нормально — так в шаблоне.
