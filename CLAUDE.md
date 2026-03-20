# CLAUDE.md — ML Pipeline Project

## Project Overview

End-to-end ML pipeline demonstration using hotel booking cancellation prediction as the example model. The model itself is not the focus — the **pipeline infrastructure** is.

## Goals

1. Data cleaning process
2. Feature engineering
3. Model training / retraining process
4. Prediction API
5. Testing and code coverage
6. Accuracy / model evaluation
7. Artifact versioning (MLflow)
8. Local deployment via Docker
9. Fully automated pipeline steps (Makefile)
10. CI/CD (local checks and tests; no cloud deploy due to budget)

## Current State

### Done
- `main.py` — Hydra-configured entry point wiring ingest → clean. Uses `conf/pipeline.yaml` for all params.
- `steps/ingest.py` — loads raw hotel bookings CSV, validates 32-column schema, drops leaky columns (`reservation_status`, `reservation_status_date`), logs raw file as first MLflow artifact under `raw_data/`
- `steps/clean.py` — fills nulls (children/agent/company→0, country→"Unknown"), removes outliers (adults>10, adr<0 or >2000, extreme stay lengths), drops duplicates, enforces dtypes (is_canceled→int8), logs cleaned CSV under `cleaned_data/` + 4 metrics
- `conf/pipeline.yaml` — Hydra config for data paths, model params, MLflow experiment name (`hotel_booking_pipeline`)
- `steps/feature_engineer.py` — builds 32 new features across 7 groups (temporal, booking behavior, stay composition, guest history, room & service, booking source, encoding); outputs 62-column all-numeric DataFrame; logs engineered CSV + 2 metrics to MLflow
- `tests/` — 90 tests, all passing, 100% coverage on `steps/`; unit, data, mlflow artifact, and integration suites
- `Dockerfile` — `python:3.12.4-slim`, installs deps only (code comes via volume mount), exposes port 5000
- `Makefile` — `dev` (persistent container), `test`, `train`, `shell`, `mlflow-ui`, `stop`, `logs`, `build`, `clean`, `fclean`
- `requirements.txt` — all deps including `pytest==8.3.4` and `pytest-cov==6.0.0`
- MLflow tracking: experiment `hotel_booking_pipeline`, file-based backend, artifacts in `mlflow-artifacts/`
- Docker dev workflow: full project dir mounted at `/app`; all make targets use `docker exec` — no rebuild on code changes

### Still To Build
- Model training (`steps/train.py`) — binary classifier on `is_canceled`, AUC-ROC + F1 metrics, log model artifact
- Prediction / inference API (FastAPI or Flask)
- Model accuracy threshold evaluation
- CI/CD workflow (GitHub Actions or similar)

## Tech Stack

| Tool | Role |
|---|---|
| scikit-learn | Model training (binary classifier, TBD) |
| MLflow | Experiment tracking, artifact versioning |
| Hydra | Config management |
| Docker | Local containerized deployment (persistent dev container) |
| Makefile | Pipeline automation |
| pandas / numpy | Data handling |

## Key Files

| File | Purpose |
|---|---|
| `main.py` | Training script |
| `data/raw/hotel_bookings.csv` | Raw dataset (target: `is_canceled`, 119,390 rows × 32 cols) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition |
| `Makefile` | Automation targets |
| `mlruns/` | MLflow run metadata |
| `mlflow.db` | MLflow SQLite tracking store |
| `mlflow-artifacts/` | MLflow artifact store |

## Running the Pipeline

All dev and testing happens inside Docker. Never use the local venv — it is not the source of truth.

```bash
# 1. Start the persistent dev container (do this once per session)
make dev

# 2. Run tests inside the container (code changes on host are live)
make test

# 3. Run the pipeline
make train

# 4. Open a shell inside the container
make shell

# 5. MLflow UI → http://localhost:5001
make mlflow-ui

# 6. Stop the dev container when done
make stop
```

The entire project directory is mounted at `/app` inside the container.
Any code change on the host is immediately visible — no rebuild needed.

## Dev / QA Workflow

Every pipeline step is built using two agents working in tandem. This is the standard process for all future work.

### Agents

| Agent | File | Role |
|---|---|---|
| `mlops-dev` | `.claude/agents/mlops-dev.md` | Implements the pipeline step |
| `mlops-qa` | `.claude/agents/mlops-qa.md` | Writes and runs the test suite |

### Process per step

```
1. Spawn mlops-dev and mlops-qa IN PARALLEL
   - Dev  → implements the step (steps/<name>.py, conf/, main.py wiring)
   - QA   → writes the test suite (tests/unit, tests/data, tests/mlflow, tests/integration)

2. Once Dev finishes → QA runs tests against the implementation

3. If FAIL:
   - QA returns structured failure report to Dev
   - Dev fixes and resubmits
   - Repeat up to 3 times total

4. After 3 failures OR PASS → surface to user:
   - PASS: show summary, ask user to approve
   - FAIL (3x): show all failure reports, ask user how to proceed

5. User approves → commit changes + update README.md / CLAUDE.md as appropriate
```

### Rules

- QA never modifies implementation files; Dev never modifies test files.
- Every step must reach PASS before moving to the next one.
- Commits only happen after explicit user approval.
- On each new step, update this file's **Current State** section accordingly.

### Known limitation

Subagents (mlops-dev, mlops-qa) do not inherit write/bash permissions from the parent session. When they hit the sandbox wall, the main Claude instance takes over and executes both roles directly, following the same conventions and quality bar.

## Artifact Strategy

Every pipeline step must produce and store a versioned artifact via MLflow. This is a core design principle, not optional.

Expected artifacts per step:
- **Data cleaning** — cleaned dataset
- **Feature engineering** — transformed feature set
- **Model training** — serialized model + hyperparams + metrics
- **Evaluation** — evaluation report / score summary
- **Prediction API** — (logged on deploy, e.g. model version tag)

All artifacts go through `mlflow.log_artifact()` or `mlflow.sklearn.log_model()` so every run is fully reproducible and auditable from the MLflow UI.

## Feature Engineering Plan

Designed from EDA on the cleaned 86,727-row dataset. To be implemented in `steps/feature_engineer.py`.

### Group 1 — Temporal
| Feature | Source | Signal |
|---|---|---|
| `arrival_month_num` | `arrival_date_month` ordinal (Jan=1…Dec=12) | Seasonal trend |
| `arrival_season` | month bins → Winter/Spring/Summer/Fall | Summer peaks at 32% cancel |
| `is_high_season` | Jul/Aug = 1 | Sharpest cancel spike |
| `arrival_day_of_week` | Reconstruct date from year+month+day | Weekend vs weekday arrivals |

### Group 2 — Booking Behavior
| Feature | Source | Signal |
|---|---|---|
| `lead_time_bucket` | Bins: 0-7, 8-30, 31-90, 91-180, 181-365, 365+ | Cancel rate: 9.6% → 40% |
| `is_long_lead` | lead_time > 90 | Binary cut at inflection point |

### Group 3 — Stay Composition
| Feature | Source | Signal |
|---|---|---|
| `total_nights` | weekend + week nights | Length of stay |
| `total_guests` | adults + children + babies | Group size |
| `is_family` | children > 0 or babies > 0 | Family booking behavior |
| `revenue_estimate` | adr × total_nights | Booking value proxy |
| `is_zero_night` | total_nights == 0 | Anomalous — 3.9% cancel |

### Group 4 — Guest History
| Feature | Source | Signal |
|---|---|---|
| `has_prev_cancel` | previous_cancellations > 0 | 67.5% vs 26.5% cancel rate |
| `prev_cancel_rate` | prev_cancels / (prev_cancels + prev_bookings) | Ratio; 0 if no history |

### Group 5 — Room & Service
| Feature | Source | Signal |
|---|---|---|
| `room_type_match` | reserved_room_type == assigned_room_type | Mismatch → only 4.7% cancel |
| `has_special_requests` | total_of_special_requests > 0 | Strong commitment signal |
| `has_parking` | required_car_parking_spaces > 0 | Commitment signal |
| `has_booking_changes` | booking_changes > 0 | Re-engagement signal |
| `has_waiting_list` | days_in_waiting_list > 0 | Waited = more committed |

### Group 6 — Booking Source
| Feature | Source | Signal |
|---|---|---|
| `is_direct_booking` | agent == 0 and company == 0 | Direct = lower cancel |
| `is_company_booking` | company > 0 | Corporate booking flag |

### Group 7 — Categorical Encoding
| Column | Strategy | Reason |
|---|---|---|
| `hotel` | Binary (City=1, Resort=0) | 2 values |
| `deposit_type` | One-hot | Non Refund = 94.6% cancel — its own world |
| `market_segment` | One-hot | 8 values, strong signal (Online TA 35%) |
| `customer_type` | One-hot | 4 values (Transient 30% vs Group 7.8%) |
| `meal` | One-hot | 5 values |
| `distribution_channel` | One-hot | 5 values |
| `arrival_date_month` | Drop (replaced by month_num + season) | Redundant after encoding |
| `country` | Target encoding | 178 unique values |
| `reserved_room_type` / `assigned_room_type` | Drop after `room_type_match` | Raw values add noise |

### Top predictors (from EDA)
1. `deposit_type` (Non Refund = 94.6% cancel)
2. `lead_time` / `lead_time_bucket`
3. `has_prev_cancel` (67.5% vs 26.5%)
4. `room_type_match` (4.7% vs 31.2%)
5. `total_of_special_requests` (33% with 0 → 5.6% with 5)
6. `market_segment`
7. `is_repeated_guest` (7.7% vs 28.1%)
8. `customer_type`
9. `adr` / `revenue_estimate`
10. `arrival_season`

## Notes

- Random seed: `np.random.seed(40)` / `random_state=42`
- Train/test split: 75/25
- MLflow experiment name: `hotel_booking_pipeline`
- Target: `is_canceled` (binary: 0=not canceled, 1=canceled); 27.3% positive rate after cleaning
- Leaky columns dropped at ingest: `reservation_status`, `reservation_status_date`
- Model metrics: AUC-ROC, F1, precision, recall (not accuracy — imbalanced target)
- Use `class_weight='balanced'` on the classifier
- Project is local-only; no cloud deployment planned
- Docker volume mounts ensure MLflow data persists on the host between runs
- All dev/testing uses Docker (`make dev` → `make test` / `make train`); never use local venv
