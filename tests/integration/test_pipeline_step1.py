import pytest
import mlflow
from steps.ingest import ingest
from steps.clean import clean, FEATURE_COLUMNS


REAL_CSV = "red-wine-quality.csv"


def test_ingest_real_csv(mlflow_tracking_uri):
    """Ingest loads the real dataset without error."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run():
        df = ingest(REAL_CSV)
    assert len(df) > 100  # dataset has 1599 rows


def test_clean_real_csv(mlflow_tracking_uri):
    """Ingest + clean on real CSV produces a valid typed DataFrame."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run():
        raw = ingest(REAL_CSV)
        df = clean(raw)

    assert len(df) > 0
    for col in FEATURE_COLUMNS:
        assert str(df[col].dtype) == "float64", f"{col} should be float64"
    assert str(df["quality"].dtype) == "int8"
    assert df.isnull().sum().sum() == 0


def test_clean_artifacts_exist_on_disk(mlflow_tracking_uri, tmp_path):
    """MLflow artifacts are written to disk after the run."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run() as run:
        raw = ingest(REAL_CSV)
        clean(raw)

    client = mlflow.tracking.MlflowClient()
    raw_artifacts = client.list_artifacts(run.info.run_id, "raw_data")
    cleaned_artifacts = client.list_artifacts(run.info.run_id, "cleaned_data")

    assert len(raw_artifacts) > 0, "No raw_data artifact found"
    assert len(cleaned_artifacts) > 0, "No cleaned_data artifact found"
