# ml-pipeline

End-to-end ML pipeline demonstration. The model (ElasticNet on red wine quality) is just an example — the focus is the pipeline infrastructure.

## Goals

| # | Feature | Status |
|---|---|---|
| 1 | Data cleaning process | pending |
| 2 | Feature engineering | pending |
| 3 | Model training / retraining | done |
| 4 | Prediction API | pending |
| 5 | Testing & code coverage | pending |
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

# Open MLflow UI at http://localhost:5000
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
| `make mlflow-ui` | Start MLflow UI at `localhost:5000` |
| `make shell` | Interactive bash shell in container |
| `make clean` | Remove stopped containers |
| `make fclean` | Remove containers and image |
| `make help` | List all targets |

## Local Python (alternative)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py --alpha 0.7 --l1_ratio 0.7
mlflow ui
```

## Project Structure

```
ml-pipeline/
├── main.py                  # Training script
├── red-wine-quality.csv     # Raw dataset
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
├── Makefile                 # Automation targets
├── data/                    # Runtime data copies
├── mlruns/                  # MLflow run metadata (persisted via volume)
└── mlflow-artifacts/        # MLflow artifact store (persisted via volume)
```

## Notes

- MLflow experiment: `experiment_1`
- Train/test split: 75/25, random seed: 40
- Model artifacts and metrics persist on the host via Docker volume mounts
