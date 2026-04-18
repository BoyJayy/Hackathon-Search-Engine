FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV QDRANT_COLLECTION_NAME=evaluation
ENV QDRANT_DENSE_VECTOR_NAME=dense
ENV QDRANT_SPARSE_VECTOR_NAME=sparse
ENV EMBEDDINGS_DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B
ENV FASTEMBED_CACHE_PATH=/models/fastembed
ENV HF_HOME=/models/huggingface

RUN mkdir -p /models/fastembed /models/huggingface
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"

EXPOSE 8000

CMD ["python", "main.py"]
