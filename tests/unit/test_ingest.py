import pytest
import pandas as pd
import mlflow
from steps.ingest import ingest, EXPECTED_COLUMNS, LEAKY_COLUMNS


def test_ingest_happy_path(raw_csv_with_leaky, mlflow_tracking_uri):
    """Valid CSV returns a DataFrame without leaky columns."""
    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        df = ingest(raw_csv_with_leaky)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in LEAKY_COLUMNS:
        assert col not in df.columns, f"Leaky column '{col}' should be dropped"


def test_ingest_drops_leaky_columns(raw_csv_with_leaky, mlflow_tracking_uri):
    """reservation_status and reservation_status_date are dropped."""
    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        df = ingest(raw_csv_with_leaky)

    for col in LEAKY_COLUMNS:
        assert col not in df.columns


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


def test_ingest_logs_raw_artifact(raw_csv_with_leaky, mlflow_tracking_uri):
    """Raw CSV is logged as an MLflow artifact under raw_data/."""
    mlflow.set_experiment("test_ingest")
    with mlflow.start_run() as run:
        ingest(raw_csv_with_leaky)

    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "raw_data")]
    assert len(artifacts) > 0


def test_ingest_strips_column_whitespace(tmp_path, mlflow_tracking_uri, valid_hotel_df):
    """Column names with surrounding whitespace are stripped after ingest."""
    df = valid_hotel_df.copy()
    # Add leaky columns so schema validation passes
    df["reservation_status"] = "Check-Out"
    df["reservation_status_date"] = "2016-07-03"
    df.columns = [f" {c} " for c in df.columns]
    csv_path = str(tmp_path / "spaced.csv")
    df.to_csv(csv_path, index=False)

    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        result = ingest(csv_path)

    for col in result.columns:
        assert col == col.strip(), f"Column '{col}' has surrounding whitespace"


def test_ingest_returns_correct_column_count(raw_csv_with_leaky, mlflow_tracking_uri):
    """Returned DataFrame has 30 columns (32 raw minus 2 leaky)."""
    mlflow.set_experiment("test_ingest")
    with mlflow.start_run():
        df = ingest(raw_csv_with_leaky)

    expected_cols = len(EXPECTED_COLUMNS) - len(LEAKY_COLUMNS)
    assert len(df.columns) == expected_cols
