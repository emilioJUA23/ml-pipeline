# ml-pipeline

End-to-end ML pipeline demonstration predicting hotel booking cancellations. The model is just an example — the focus is the pipeline infrastructure.

## Goals

| # | Feature | Status |
|---|---|---|
| 1 | Data ingestion & cleaning | done |
| 2 | Feature engineering | done |
| 3 | Model training / retraining | pending |
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
# Build the image and start the persistent dev container
make dev

# Run the pipeline (ingest → clean)
make train

# Run test suite with coverage
make test

# Open MLflow UI at http://localhost:5001
make mlflow-ui

# Drop into a bash shell inside the container
make shell

# Stop the dev container
make stop
```

> Code changes on your host are live inside the container immediately — no rebuild needed.
> Only run `make build` again when `requirements.txt` changes.

## Makefile Targets

| Target | Description |
|---|---|
| `make build` | Build the Docker image (deps only) |
| `make dev` | Start persistent dev container with project dir mounted |
| `make stop` | Stop and remove the dev container |
| `make train` | Run the pipeline inside the dev container |
| `make test` | Run full test suite with coverage |
| `make mlflow-ui` | Start MLflow UI at `localhost:5001` |
| `make shell` | Interactive bash shell inside the dev container |
| `make logs` | Follow dev container logs |
| `make clean` | Stop the dev container |
| `make fclean` | Stop container and remove the Docker image |
| `make help` | List all targets |

## Viewing Artifacts

After `make train`, run `make mlflow-ui` and open **http://localhost:5001**.

Navigate to `hotel_booking_pipeline` → select a run → **Artifacts** tab:
- `raw_data/hotel_bookings.csv` — original dataset as ingested
- `cleaned_data/cleaned_hotel_bookings.csv` — nulls filled, outliers removed, typed dataset
- `engineered_data/engineered_hotel_bookings.csv` — 62 model-ready features

## Dataset

**Hotel Booking Demand** — 119,390 bookings from two hotels (2015–2017).

| Property | Value |
|---|---|
| Raw rows | 119,390 |
| Rows after cleaning | 86,727 |
| Columns | 30 (32 raw minus 2 leaky) |
| Target | `is_canceled` (binary: 27.3% positive) |
| Leaky columns dropped | `reservation_status`, `reservation_status_date` |

## Project Structure

```
ml-pipeline/
├── main.py                      # Pipeline entry point (Hydra-configured)
├── steps/
│   ├── ingest.py                # Step 1: load CSV, validate schema, drop leaky cols, log artifact
│   ├── clean.py                 # Step 2: fill nulls, remove outliers, dedup, log artifact
│   └── feature_engineer.py     # Step 3: derive 32 features, encode categoricals, log artifact
├── conf/
│   └── pipeline.yaml            # Hydra config (paths, model params, experiment name)
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests per step (ingest, clean, feature_engineer)
│   ├── data/                    # Schema and value validation tests
│   ├── mlflow_artifacts/        # MLflow metric and artifact logging tests
│   └── integration/             # End-to-end pipeline tests against real data
├── data/
│   └── raw/
│       └── hotel_bookings.csv   # Raw dataset
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Installs deps only — code comes via volume mount
└── Makefile                     # Automation targets
```

## Notes

- MLflow experiment: `hotel_booking_pipeline`
- Train/test split: 75/25, random seed: 40 / 42
- `mlruns/` and `mlflow-artifacts/` are git-ignored — run history lives on your local machine only
- All `make` targets that run code use `docker exec` against the persistent dev container
