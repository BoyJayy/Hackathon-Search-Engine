FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV CHUNK_SIZE=10
ENV FASTEMBED_CACHE_PATH=/models/fastembed
ENV HF_HOME=/models/huggingface

RUN mkdir -p /models/fastembed /models/huggingface
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"

EXPOSE 8000

CMD ["python", "main.py"]
