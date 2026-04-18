LOGIN ?= 
PASSWORD ?= 
TEAM_ID ?= 
DOCKER_REGISTRY_URL ?= 83.166.249.64:5000
PORT ?= 8000
QDRANT_URL ?= 
QDRANT_COLLECTION_NAME ?= evaluation
EMBEDDINGS_DENSE_URL ?= 
EMBEDDINGS_DENSE_MODEL ?= Qwen/Qwen3-Embedding-0.6B
API_KEY ?= dev-api-key
RERANKER_URL ?= 
QDRANT_DENSE_VECTOR_NAME ?= dense
QDRANT_SPARSE_VECTOR_NAME ?= sparse

REQUIRED_RUN_VARS := QDRANT_URL EMBEDDINGS_DENSE_URL API_KEY RERANKER_URL

IMAGE = $(DOCKER_REGISTRY_URL)/$(TEAM_ID)/search-service:latest

.PHONY: login build run push check-run-env

login:
	@: $(if $(LOGIN),,$(error LOGIN is required for make login))
	@: $(if $(PASSWORD),,$(error PASSWORD is required for make login))
	docker login $(DOCKER_REGISTRY_URL) -u $(LOGIN) -p $(PASSWORD)

build:
	@: $(if $(TEAM_ID),,$(error TEAM_ID is required for make build))
	docker build -t $(IMAGE) ./

run: build
	@: $(foreach var,$(REQUIRED_RUN_VARS),$(if $($(var)),,$(error $(var) is required for make run)))
	docker run --rm -p $(PORT):8000 \
		--add-host=host.docker.internal:host-gateway \
		-e QDRANT_URL=$(QDRANT_URL) \
		-e QDRANT_COLLECTION_NAME=$(QDRANT_COLLECTION_NAME) \
		-e EMBEDDINGS_DENSE_URL=$(EMBEDDINGS_DENSE_URL) \
		-e EMBEDDINGS_DENSE_MODEL=$(EMBEDDINGS_DENSE_MODEL) \
		-e API_KEY=$(API_KEY) \
		-e RERANKER_URL=$(RERANKER_URL) \
		-e QDRANT_DENSE_VECTOR_NAME=$(QDRANT_DENSE_VECTOR_NAME) \
		-e QDRANT_SPARSE_VECTOR_NAME=$(QDRANT_SPARSE_VECTOR_NAME) \
		$(IMAGE)

push: login build
	docker push $(IMAGE)
