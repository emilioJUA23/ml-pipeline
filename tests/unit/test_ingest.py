import pytest
import pandas as pd
import mlflow
from steps.ingest import ingest, EXPECTED_COLUMNS


def test_ingest_happy_path(raw_csv, mlflow_tracking_uri, tmp_path):
    """Valid CSV returns a DataFrame with all expected columns."""
    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        df = ingest(raw_csv)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 0


def test_ingest_missing_file(mlflow_tracking_uri):
    """Missing file raises FileNotFoundError."""
    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        with pytest.raises(FileNotFoundError, match="not found"):
            ingest("nonexistent_file.csv")


def test_ingest_missing_columns(tmp_path, mlflow_tracking_uri):
    """CSV missing required columns raises ValueError."""
    bad_csv = str(tmp_path / "bad.csv")
    pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(bad_csv, index=False)

    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        with pytest.raises(ValueError, match="missing expected columns"):
            ingest(bad_csv)


def test_ingest_logs_artifact(raw_csv, mlflow_tracking_uri, tmp_path):
    """Raw CSV is logged as an MLflow artifact under raw_data/."""
    mlflow.set_experiment("test_ingest")
    with mlflow.start_run() as run:
        ingest(raw_csv)

    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "raw_data")]
    assert any("test_wine.csv" in a for a in artifacts)


def test_ingest_strips_column_whitespace(tmp_path, mlflow_tracking_uri, valid_wine_df):
    """Column names with surrounding whitespace are stripped after ingest."""
    df_spaces = valid_wine_df.copy()
    df_spaces.columns = [f" {c} " for c in df_spaces.columns]
    csv_path = str(tmp_path / "spaced.csv")
    df_spaces.to_csv(csv_path, index=False)

    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        result = ingest(csv_path)

    for col in result.columns:
        assert col == col.strip(), f"Column '{col}' has surrounding whitespace"
