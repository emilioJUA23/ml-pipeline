IMAGE_NAME    := ml-pipeline
DEV_CONTAINER := ml-pipeline-dev
MLFLOW_PORT   := 5001

# Absolute path to the project root (so volume mounts work from any CWD)
PROJECT_DIR := $(shell pwd)

.PHONY: help build dev stop logs test train shell mlflow-ui clean fclean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Image ──────────────────────────────────────────────────────────────────────

build: ## Build the Docker image (deps only — code comes via volume)
	docker build -t $(IMAGE_NAME) .

# ── Dev container ──────────────────────────────────────────────────────────────

dev: build ## Start persistent dev container with full project dir mounted
	@if docker ps -q -f name=$(DEV_CONTAINER) | grep -q .; then \
		echo "Dev container '$(DEV_CONTAINER)' is already running."; \
	else \
		docker run -d \
			--name $(DEV_CONTAINER) \
			-e MLFLOW_TRACKING_URI=file:///app/mlruns \
			-v $(PROJECT_DIR):/app \
			-p $(MLFLOW_PORT):5000 \
			$(IMAGE_NAME) \
			sleep infinity && \
		echo "Dev container started. Code at $(PROJECT_DIR) is live-mounted at /app."; \
	fi

stop: ## Stop and remove the dev container
	docker rm -f $(DEV_CONTAINER) 2>/dev/null && echo "Stopped $(DEV_CONTAINER)." || true

logs: ## Follow dev container logs
	docker logs -f $(DEV_CONTAINER)

# ── Run inside dev container ────────────────────────────────────────────────────

test: ## Run full test suite inside the dev container
	docker exec -i $(DEV_CONTAINER) pytest tests/ -v --cov=steps --cov-report=term-missing

train: ## Run the pipeline inside the dev container
	docker exec -i $(DEV_CONTAINER) python main.py

shell: ## Open an interactive bash shell inside the dev container
	docker exec -it $(DEV_CONTAINER) bash

mlflow-ui: ## Start MLflow UI inside the dev container (http://localhost:5001)
	docker exec -d $(DEV_CONTAINER) mlflow ui --host 0.0.0.0 --port 5000
	@echo "MLflow UI → http://localhost:$(MLFLOW_PORT)"

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean: stop ## Stop the dev container

fclean: clean ## Stop container AND remove the Docker image
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
