# CLAUDE.md — ML Pipeline Project

## Project Overview

End-to-end ML pipeline demonstration using red wine quality prediction as the example model. The model itself is not the focus — the **pipeline infrastructure** is.

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
- `main.py` — Hydra-configured entry point wiring ingest → clean → train. Uses `conf/pipeline.yaml` for all params.
- `steps/ingest.py` — loads raw CSV, validates schema, logs raw file as first MLflow artifact under `raw_data/`
- `steps/clean.py` — enforces dtypes (features→float64, quality→int8), drops duplicates/nulls/out-of-range rows, clips to plausible ranges, logs cleaned CSV under `cleaned_data/` + 4 metrics
- `conf/pipeline.yaml` — Hydra config for data paths, model hyperparams, MLflow experiment name
- `tests/` — 46 tests, 100% coverage on `steps/`; unit, data, mlflow artifact, and integration suites
- `Dockerfile` — `python:3.12.4-slim`, copies steps/conf/tests, sets `GIT_PYTHON_REFRESH=quiet`, exposes port 5000
- `Makefile` — build, train, train-custom, test, mlflow-ui, shell, clean, fclean
- `requirements.txt` — all deps including `pytest==8.3.4` and `pytest-cov==6.0.0`
- MLflow tracking: experiment `experiment_1`, SQLite backend (`mlflow.db`), artifacts in `mlflow-artifacts/`

### Still To Build
- Feature engineering pipeline
- Prediction / inference API (FastAPI or Flask)
- Model accuracy threshold evaluation
- CI/CD workflow (GitHub Actions or similar)

## Tech Stack

| Tool | Role |
|---|---|
| scikit-learn | Model training (ElasticNet) |
| MLflow | Experiment tracking, artifact versioning |
| Hydra | Config management (installed, not yet wired in) |
| Docker | Local containerized deployment |
| Makefile | Pipeline automation |
| pandas / numpy | Data handling |

## Key Files

| File | Purpose |
|---|---|
| `main.py` | Training script |
| `red-wine-quality.csv` | Raw dataset (target: `quality`, range 3–9) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition |
| `Makefile` | Automation targets |
| `mlruns/` | MLflow run metadata |
| `mlflow.db` | MLflow SQLite tracking store |
| `mlflow-artifacts/` | MLflow artifact store |

## Running the Pipeline

```bash
# Docker (preferred)
make build
make train
make train-custom ALPHA=0.3 L1=0.5
make mlflow-ui        # http://localhost:5000

# Local Python
source venv/bin/activate
python main.py --alpha 0.7 --l1_ratio 0.7
mlflow ui
```

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

## Notes

- Random seed: `np.random.seed(40)` / `random_state=42`
- Train/test split: 75/25
- MLflow experiment name: `experiment_1`
- Project is local-only; no cloud deployment planned
- Docker volume mounts ensure MLflow data persists on the host between runs
