# ml-pipeline

End-to-end ML pipeline demonstration. The model (ElasticNet on red wine quality) is just an example — the focus is the pipeline infrastructure.

## Goals

| # | Feature | Status |
|---|---|---|
| 1 | Data ingestion & cleaning | done |
| 2 | Feature engineering | pending |
| 3 | Model training / retraining | done |
| 4 | Prediction API | pending |
| 5 | Testing & code coverage | done |
| 6 | Accuracy / model evaluation | pending |
| 7 | Artifact versioning (MLflow) | done |
| 8 | Local deployment via Docker | done |
| 9 | Fully automated pipeline steps (Makefile) | done |
| 10 | CI/CD (local checks & tests, no cloud deploy) | pending |

## Requirements

- Docker
- `make`

No local Python setup needed — everything runs inside the container.

## Quick Start

```bash
# Build the image
make build

# Train with default hyperparams (alpha=0.7, l1_ratio=0.7)
make train

# Train with custom hyperparams
make train-custom ALPHA=0.3 L1=0.5

# Run test suite
make test

# Open MLflow UI at http://localhost:5001
make mlflow-ui

# Drop into a shell inside the container
make shell
```

## Makefile Targets

| Target | Description |
|---|---|
| `make build` | Build the Docker image |
| `make train` | Run training with default params |
| `make train-custom ALPHA=X L1=Y` | Run training with custom hyperparams |
| `make test` | Run full test suite with coverage |
| `make mlflow-ui` | Start MLflow UI at `localhost:5001` |
| `make shell` | Interactive bash shell in container |
| `make clean` | Remove stopped containers |
| `make fclean` | Remove containers and image |
| `make help` | List all targets |

## Viewing Artifacts

After `make train`, run `make mlflow-ui` and open **http://localhost:5001**.

Navigate to `experiment_1` → select a run → **Artifacts** tab. You'll find:
- `raw_data/red-wine-quality.csv` — original dataset as ingested
- `cleaned_data/cleaned_wine.csv` — typed, deduplicated, clipped dataset
- `linear_model/` — serialized ElasticNet model

## Project Structure

```
ml-pipeline/
├── main.py                  # Pipeline entry point (Hydra-configured)
├── steps/
│   ├── ingest.py            # Step 1: load CSV, log raw artifact
│   └── clean.py             # Step 2: type enforcement, dedup, clip, log cleaned artifact
├── conf/
│   └── pipeline.yaml        # Hydra config (paths, hyperparams, experiment name)
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── unit/                # Unit tests for each step
│   ├── data/                # Schema and range validation tests
│   ├── mlflow_artifacts/    # MLflow metric and artifact logging tests
│   └── integration/         # End-to-end pipeline tests
├── red-wine-quality.csv     # Raw dataset
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
└── Makefile                 # Automation targets
```

## Notes

- MLflow experiment: `experiment_1`
- Train/test split: 75/25, random seed: 40
- All Docker run targets set `MLFLOW_TRACKING_URI=file:///app/mlruns` to ensure artifacts persist correctly via volume mount
- `mlruns/` is git-ignored — artifact history lives on your local machine only
