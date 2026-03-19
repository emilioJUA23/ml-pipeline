IMAGE_NAME  := ml-pipeline
CONTAINER   := ml-pipeline-run
MLFLOW_PORT := 5001

# Absolute path to the project root (so volume mounts work from any CWD)
PROJECT_DIR := $(shell pwd)

.PHONY: help build train train-custom test mlflow-ui shell clean fclean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Image ──────────────────────────────────────────────────────────────────────

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

# ── Training ───────────────────────────────────────────────────────────────────

train: build ## Run training with default hyperparams (alpha=0.7, l1_ratio=0.7)
	docker run --rm \
		-e MLFLOW_TRACKING_URI=file:///app/mlruns \
		-v $(PROJECT_DIR)/mlruns:/app/mlruns \
		-v $(PROJECT_DIR)/mlflow-artifacts:/app/mlflow-artifacts \
		-v $(PROJECT_DIR)/data:/app/data \
		--name $(CONTAINER) \
		$(IMAGE_NAME)

train-custom: build ## Run training with custom params — usage: make train-custom ALPHA=0.3 L1=0.5
	docker run --rm \
		-e MLFLOW_TRACKING_URI=file:///app/mlruns \
		-v $(PROJECT_DIR)/mlruns:/app/mlruns \
		-v $(PROJECT_DIR)/mlflow-artifacts:/app/mlflow-artifacts \
		-v $(PROJECT_DIR)/data:/app/data \
		--name $(CONTAINER) \
		$(IMAGE_NAME) python main.py --alpha $(ALPHA) --l1_ratio $(L1)

# ── Tests ─────────────────────────────────────────────────────────────────────

test: build ## Run full test suite with coverage
	docker run --rm \
		-e MLFLOW_TRACKING_URI=file:///app/mlruns \
		-v $(PROJECT_DIR)/mlruns:/app/mlruns \
		-v $(PROJECT_DIR)/data:/app/data \
		-v $(PROJECT_DIR)/red-wine-quality.csv:/app/red-wine-quality.csv \
		--name $(CONTAINER)-test \
		$(IMAGE_NAME) \
		pytest tests/ -v --cov=steps --cov-report=term-missing

# ── MLflow UI ──────────────────────────────────────────────────────────────────

mlflow-ui: ## Start the MLflow tracking UI at http://localhost:5001
	docker run --rm \
		-v $(PROJECT_DIR)/mlruns:/app/mlruns \
		-v $(PROJECT_DIR)/mlflow-artifacts:/app/mlflow-artifacts \
		-p $(MLFLOW_PORT):5000 \
		--name mlflow-ui \
		$(IMAGE_NAME) \
		mlflow ui --host 0.0.0.0 --port 5000

# ── Dev shell ──────────────────────────────────────────────────────────────────

shell: build ## Open an interactive bash shell inside the container
	docker run --rm -it \
		-v $(PROJECT_DIR)/mlruns:/app/mlruns \
		-v $(PROJECT_DIR)/mlflow-artifacts:/app/mlflow-artifacts \
		-v $(PROJECT_DIR)/data:/app/data \
		-v $(PROJECT_DIR)/main.py:/app/main.py \
		--name $(CONTAINER)-shell \
		$(IMAGE_NAME) bash

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean: ## Remove stopped containers
	docker rm -f $(CONTAINER) mlflow-ui 2>/dev/null || true

fclean: clean ## Remove containers AND the Docker image
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
