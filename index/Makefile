LOGIN ?= 
PASSWORD ?= 
TEAM_ID ?= 
DOCKER_REGISTRY_URL ?= 83.166.249.64:5000
PORT ?= 8000

IMAGE = $(DOCKER_REGISTRY_URL)/$(TEAM_ID)/index-service:latest

.PHONY: login build run push

login:
	@: $(if $(LOGIN),,$(error LOGIN is required for make login))
	@: $(if $(PASSWORD),,$(error PASSWORD is required for make login))
	docker login $(DOCKER_REGISTRY_URL) -u $(LOGIN)  -p $(PASSWORD)

build:
	@: $(if $(TEAM_ID),,$(error TEAM_ID is required for make build))
	docker build -t $(IMAGE) ./

run: build
	docker run --rm -p $(PORT):8000 $(IMAGE)

push: login build
	docker push $(IMAGE)
