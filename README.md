# ml-pipeline

End-to-end ML pipeline demonstration predicting hotel booking cancellations. The model is just an example — the focus is the pipeline infrastructure.

<!-- updated by emilioJUA23 -->

## Goals

| # | Feature | Status |
|---|---|---|
| 1 | Data ingestion & cleaning | done |
| 2 | Feature engineering | done |
| 3 | Model training / retraining | done |
| 4 | Prediction API | done |
| 5 | Testing & code coverage | done |
| 6 | Accuracy / model evaluation | done |
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

# Run the full pipeline (ingest → clean → feature engineering → train)
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
- `models/adr_predictor/` — GradientBoostingRegressor for price prediction
- `models/cancel_predictor/` — RandomForestClassifier for cancellation prediction
- `artifacts/feature_columns_adr.json` — ordered feature list for ADR model inference
- `artifacts/feature_defaults.json` — training-set medians/modes used as inference defaults
- `artifacts/country_encoding_map.json` — country code → encoded value lookup
- `artifacts/feature_importances_adr.csv` — ranked ADR model feature importances
- `artifacts/feature_importances_cancel.csv` — ranked cancellation model feature importances

**Metrics logged per run:**

| Metric | Model | Description |
|---|---|---|
| `adr_rmse` | ADR Regressor | Root mean squared error |
| `adr_mae` | ADR Regressor | Mean absolute error |
| `adr_r2` | ADR Regressor | R² score |
| `cancel_auc` | Cancel Classifier | AUC-ROC |
| `cancel_f1` | Cancel Classifier | F1 score |
| `cancel_precision` | Cancel Classifier | Precision |
| `cancel_recall` | Cancel Classifier | Recall |

## Recommendation API

The pipeline produces a FastAPI app (`api/app.py`) that answers:
> *"Given my hotel type, country, party size, and number of nights — what month and day of the week should I book to get the lowest price?"*

```bash
# Start the API (after make train)
docker exec -i ml-pipeline-dev uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload

# Example request
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"hotel_type": "city", "guest_country": "GBR", "adults": 2, "children": 0, "num_days": 5}'
```

**Response:**
```json
{
  "hotel_type": "city",
  "guest_country": "GBR",
  "adults": 2,
  "children": 0,
  "num_days": 5,
  "best": {
    "month": "January",
    "day_of_week": "Monday",
    "estimated_adr": 72.50,
    "estimated_total": 362.50
  },
  "recommendations": [ ... ]
}
```

The API sweeps all 84 arrival-date combinations (12 months × 7 days) through the trained ADR model and returns the top-N cheapest windows.

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
│   ├── feature_engineer.py     # Step 3: derive 32 features, encode categoricals, log artifact
│   ├── train.py                 # Step 4: train ADR regressor + cancel classifier, log models
│   └── predict.py               # Recommendation logic: sweep 84 combos, rank by predicted ADR
├── api/
│   └── app.py                   # FastAPI recommendation API (POST /recommend)
├── conf/
│   └── pipeline.yaml            # Hydra config (paths, model params, experiment name)
├── tests/
│   ├── conftest.py              # Shared fixtures (including synthetic engineered DataFrames)
│   ├── unit/                    # Unit tests per step (126 tests, 100% coverage)
│   ├── data/                    # Schema and value validation tests
│   ├── mlflow_artifacts/        # MLflow metric and artifact logging tests
│   └── integration/             # End-to-end pipeline + API tests
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
