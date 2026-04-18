# Пример решения

Этот репозиторий содержит минимальный шаблон решения, совместимый с нашей проверяющей системой.

В шаблоне есть два сервиса:
- `index` — получает сообщения и строит чанки для индексации;
- `search` — получает вопрос и возвращает `message_ids` найденных сообщений.

Это не эталонное решение по качеству поиска. Его задача — показать корректный контракт и базовый pipeline.

## Состав репозитория

- `index/main.py` — сервис индексации;
- `search/main.py` — сервис поиска;
- `data/Go Nova.json` — анонимизированный пример реального чата;
- `index/Makefile`, `search/Makefile` — локальная сборка и публикация образов.

## Что можно менять

Вы можете менять:
- внутреннюю логику chunking;
- формирование текста для dense;
- формирование sparse векторов;
- retrieval и rerank pipeline;
- любые эвристики и фильтры.

Вы не должны менять:
- контракт `POST /index`;
- контракт `POST /sparse_embedding`;
- контракт `POST /search`.

Если вы меняете request/response этих endpoint'ов, решение перестанет проходить проверку.

## Формат данных

Для разбора входных данных используйте `data/Go Nova.json`.

Особенно важно посмотреть:
- `messages`;
- поля `text`, `parts`, `mentions`;
- metadata чанков в payload Qdrant (можете увидеть её формат в коде сервиса `search`).

Во многих сообщениях значимая часть текста находится не в корневом `text`, а в `parts[*].text`.

Поле `parts` может содержать:
- обычный текст;
- цитату;
- пересланное сообщение.

Если вы только начинаете работу с шаблоном, сначала откройте `data/Go Nova.json`, затем сравните его со схемами в `index/main.py` и `search/main.py`.

При индексации в metadata чанка мы также сохраняем полезные поля:
- `participants` — уникальные `sender_id` сообщений, попавших в чанк;
- `mentions` — уникальные упоминания из сообщений чанка;
- `contains_forward` — есть ли в чанке пересланные сообщения;
- `contains_quote` — есть ли в чанке цитаты других сообщений.

Этими полями можно пользоваться в запросе к `Qdrant`.

## Сервис `index`

`index` должен реализовать:
- `GET /health`
- `POST /index`
- `POST /sparse_embedding`

>ВАЖНО! У `index`-контейнера нет доступа в интернет. Поэтому sparse-модель должна быть доступна локально внутри образа.

### `GET /health`

Endpoint для health check контейнера.

Мы используем его, чтобы проверить, что контейнер успешно запустился и сервис готов принимать запросы.

На выходе endpoint должен вернуть `200 OK`.

### `POST /index`

На вход приходят:
- `chat`;
- `overlap_messages`;
- `new_messages`.

Endpoint должен:
- принять новую порцию сообщений для индексации;
- построить чанки;
- вернуть текст чанков и связанные `message_ids`.

На выходе сервис должен вернуть `results[]`, где каждый элемент содержит:
- `page_content` — текст чанка, который сохраняется в payload;
- `dense_content` — текст для dense embedding;
- `sparse_content` — текст для sparse embedding;
- `message_ids` — id сообщений, связанных с чанком.

Пример запроса:

```bash
curl -X POST "http://localhost:8001/index" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "chat": {
        "id": "chat-1",
        "name": "Go Nova",
        "sn": "chat-1@chat.agent",
        "type": "channel",
        "is_public": true
      },
      "overlap_messages": [
        {
          "id": "1",
          "time": 1710000000,
          "text": "Обсуждаем релиз Go",
          "sender_id": "u1",
          "file_snippets": "",
          "parts": [],
          "mentions": [],
          "is_system": false,
          "is_hidden": false,
          "is_forward": false,
          "is_quote": false
        }
      ],
      "new_messages": [
        {
          "id": "2",
          "time": 1710000060,
          "text": "Релиз Go перенесли на следующую неделю",
          "sender_id": "u2",
          "file_snippets": "",
          "parts": [],
          "mentions": [],
          "is_system": false,
          "is_hidden": false,
          "is_forward": false,
          "is_quote": false
        }
      ]
    }
  }'
```

Пример ответа:

```json
{
  "results": [
    {
      "page_content": "Обсуждаем релиз Go\nРелиз Go перенесли на следующую неделю",
      "dense_content": "Обсуждаем релиз Go\nРелиз Go перенесли на следующую неделю",
      "sparse_content": "Обсуждаем релиз Go\nРелиз Go перенесли на следующую неделю",
      "message_ids": ["1", "2"]
    }
  ]
}
```

### `POST /sparse_embedding`

На вход endpoint получает:
- `texts: string[]`

Endpoint должен:
- принять batch текстов;
- посчитать sparse-вектора;
- вернуть их в формате, совместимом с Qdrant.

На выходе должен вернуть:
- `vectors[].indices`
- `vectors[].values`

Этот endpoint мы используем для вычисления sparse векторов.

Пример запроса:

```bash
curl -X POST "http://localhost:8001/sparse_embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Релиз Go перенесли на следующую неделю",
      "VK GPT обсуждали в отдельном чате"
    ]
  }'
```

Пример ответа:

```json
{
  "vectors": [
    {
      "indices": [1452, 9081, 12044],
      "values": [0.6931, 1.0986, 0.6931]
    },
    {
      "indices": [317, 2801, 9102],
      "values": [0.6931, 0.6931, 1.3863]
    }
  ]
}
```


## Сервис `search`

`search` должен реализовать:
- `GET /health`
- `POST /search`

>ВАЖНО! У `search`-контейнера тоже нет доступа в интернет. Поэтому sparse-модель должна быть доступна локально внутри образа.


### `GET /health`

Endpoint для health check контейнера.

Мы используем его, чтобы проверить, что контейнер успешно запустился и сервис готов принимать запросы.

На выходе endpoint должен вернуть `200 OK`.

### `POST /search`

На вход `POST /search` приходит вопрос пользователя с некоторой метадатой (можете изучить её в коде сервиса `search`).

Endpoint должен:
1. получить вектора для поискового запроса;
2. выполнить retrieval по коллекции в Qdrant;
3. при необходимости выполнить rerank;
4. вернуть `message_ids` найденных сообщений.

На выходе `POST /search` должен вернуть `results[]`, где каждый элемент содержит:
- `message_ids`

Пример запроса:

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": {
      "text": "Что писали про релиз Go?"
    }
  }'
```

Пример ответа:

```json
{
  "results": [
    {
      "message_ids": ["2", "1"]
    }
  ]
}
```

## Доступ к нашим сервисам из ваших контейнеров

Во время проверки мы передаем адреса наших сервисов через env. Внутри ваших контейнеров нужно использовать именно эти переменные, а не хардкодить адреса.

### `index`

`index` должен использовать:
- `HOST`
- `PORT`

### `search`

`search` должен использовать:
- `HOST`
- `PORT`
- `API_KEY`
- `EMBEDDINGS_DENSE_URL`
- `RERANKER_URL`
- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`
- `QDRANT_DENSE_VECTOR_NAME`
- `QDRANT_SPARSE_VECTOR_NAME`

## Открытый API dense и rerank

Для dense embeddings и rerank мы предоставляем внешний HTTP API с Basic Auth. 
Будьте осторожны при использовании этих URL'ов, для каждой команды есть ограничение для использования (rate limit) в символах в минуту.

Базовый URL:
- `http://83.166.249.64:18001`

Доступные endpoint'ы:
- `POST /embeddings` — dense embeddings;
- `GET /embeddings/models` — список dense embedding моделей;
- `POST /score` — rerank;
- `GET /score/models` — список rerank моделей.

Примеры запросов:

Dense embeddings:

```bash
curl -u "$LOGIN:$PASSWORD" \
  -X POST "http://83.166.249.64:18001/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["Пример поискового запроса"],
    "encoding_format": "base64"
  }'
```

Список dense embedding моделей:

```bash
curl -u "$LOGIN:$PASSWORD" \
  "http://83.166.249.64:18001/embeddings/models"
```

Rerank:

```bash
curl -u "$LOGIN:$PASSWORD" \
  -X POST "http://83.166.249.64:18001/score" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-nemotron-rerank-1b-v2",
    "text_1": "Что обсуждали про релиз Go?",
    "text_2": [
      "Первый кандидат для реранка",
      "Второй кандидат для реранка"
    ]
  }'
```

Список rerank моделей:

```bash
curl -u "$LOGIN:$PASSWORD" \
  "http://83.166.249.64:18001/score/models"
```

## Примеры вопросов

Примеры вопросов из test датасета, на котором оцениваются ваши решения.

Примеры:
- `В какой чат можно написать вопрос про VK GPT?`
- `Кто сейчас является руководителем команды коробочных продуктов Tarantool?`
- `Какие проблемы были с сервисом oauth-antibot в августе 2020?`

## Оценивание

При оценивании мы считаем две метрики:
- `Recall@K`
- `nDCG@K`

Где `K` = 50. Если ваше решение возвращает больше 50-и `message_id`, то все после первых 50-и отбрасываются и не учитываются.

Для каждой посылки мы усредняем значения этих метрик по всем вопросам датасета и получаем:
- `recall_avg`
- `ndcg_avg`

Итоговый `score` рассчитывается по формуле:

```text
score = recall_avg * 0.8 + ndcg_avg * 0.2
```

## Время выполнения

Время выполнения почти полностью состоит из вычисления векторов и индексации и занимает приблизительно `15 минут`
Вы должны уложиться в `20 минут`

## Локальный запуск

Для локального запуска используйте `docker compose`.

Перед запуском укажите учетные данные для внешнего dense/rerank API:

```bash
export OPEN_API_LOGIN=...
export OPEN_API_PASSWORD=...
```

Если эти переменные не заданы, `docker compose up` завершится с ошибкой.

Запуск:

```bash
docker compose up --build
```

Будут подняты:
- `qdrant` на `localhost:6333`
- `index` на `localhost:8001`
- `search` на `localhost:8002`

`docker compose` также создаст локальную коллекцию `evaluation` в Qdrant.

>ВАЖНО! В текущем примере INDEX не вставляет чанки в Qdrant, поэтому `search` не вернет чанков

Сборка и запуск `index`:

```bash
cd index
make build
make run
```

Сборка и запуск `search`:

```bash
cd search
make build
make run
```

## Сдача решения

### Шаг 1. Получение логина и пароля

В личном кабинете получите логин и пароль для входа в Docker registry

### Шаг 2. Настройка Docker

Registry для хранения образов будет доступен по адресу `83.166.249.64:5000`.
Поскольку он работает без TLS, необходимо добавить его в список insecure registries в настройках Docker.

#### Docker Desktop (macOS / Windows)

1. Откройте Docker Desktop -> **Settings** -> **Docker Engine**.
2. Добавьте в JSON-конфиг поле `insecure-registries`:

```json
{
  "insecure-registries": ["83.166.249.64:5000"]
}
```

3. Нажмите **Apply & Restart**.

#### CLI — Linux

1. Откройте файл с конфигурацией докер демона в режиме редактирования

```bash
sudo nano /etc/docker/daemon.json
```

2. Добавьте в JSON-конфиг поле `insecure-registries`:

```json
{
  "insecure-registries": ["83.166.249.64:5000"]
}
```

3. Перезапустите docker

```bash
sudo systemctl restart docker
```

### Шаг 3. Логин в registry

Необходимо пройти аутентификацию в docker registry, используя логин и пароль, полученные на шаге 1.

```bash
docker login 83.166.249.64:5000 -u <login> -p <password>
```

Ожидаемый вывод: `Login Succeeded`.

### Шаг 4. Сборка образов

Соберите образы своих Index Service и Search Service.
Образы должны иметь тег, который имеет строгий формат:

- Для Index Service - 83.166.249.64:5000/{team_id}/index-service:latest
- Для Search Service - 83.166.249.64:5000/{team_id}/search-service:latest

Образы должны быть собраны под платформу linux/amd64

```bash
docker build --platform linux/amd64 -t 83.166.249.64:5000/{team_id}/index-service:latest {path_to_index_service_dir}
docker build --platform linux/amd64 -t 83.166.249.64:5000/{team_id}/search-service:latest {path_to_search_service_dir}
```

### Шаг 5. Публикация образов в registry

```bash
docker push 83.166.249.64:5000/{team_id}/index-service:latest
docker push 83.166.249.64:5000/{team_id}/search-service:latest
```

### Шаг 6. Запуск процесса оценивания

На страничке оценивания нажать кнопку "Запустить/Оценить"

### Примечание

Для удобства сборки и публикации образов в шаблоне решения уже есть `Makefile`'ы с нужными командами

Примеры:

```bash
cd index
export LOGIN=... PASSWORD=... TEAM_ID=...
make push
```

```bash
cd search
export LOGIN=... PASSWORD=... TEAM_ID=...
make push
```
