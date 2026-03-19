import mlflow
from steps.ingest import ingest
from steps.clean import clean


def test_cleaning_metrics_logged(valid_wine_df, mlflow_tracking_uri):
    """rows_before, rows_after, rows_dropped, columns are logged as MLflow metrics."""
    mlflow.set_experiment("test_artifacts")
    with mlflow.start_run() as run:
        clean(valid_wine_df)

    data = mlflow.get_run(run.info.run_id).data
    assert "rows_before" in data.metrics
    assert "rows_after" in data.metrics
    assert "rows_dropped" in data.metrics
    assert "columns" in data.metrics


def test_rows_before_gte_rows_after(dirty_wine_df, mlflow_tracking_uri):
    """Logged rows_before >= rows_after."""
    mlflow.set_experiment("test_artifacts")
    with mlflow.start_run() as run:
        clean(dirty_wine_df)

    metrics = mlflow.get_run(run.info.run_id).data.metrics
    assert metrics["rows_before"] >= metrics["rows_after"]


def test_rows_dropped_equals_difference(dirty_wine_df, mlflow_tracking_uri):
    """rows_dropped == rows_before - rows_after."""
    mlflow.set_experiment("test_artifacts")
    with mlflow.start_run() as run:
        clean(dirty_wine_df)

    m = mlflow.get_run(run.info.run_id).data.metrics
    assert m["rows_dropped"] == m["rows_before"] - m["rows_after"]


def test_cleaned_csv_artifact_logged(valid_wine_df, mlflow_tracking_uri):
    """Cleaned CSV artifact is logged under cleaned_data/."""
    mlflow.set_experiment("test_artifacts")
    with mlflow.start_run() as run:
        clean(valid_wine_df, cleaned_filename="cleaned_wine.csv")

    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "cleaned_data")]
    assert any("cleaned_wine.csv" in a for a in artifacts)


def test_raw_csv_artifact_logged(raw_csv, mlflow_tracking_uri):
    """Raw CSV artifact is logged under raw_data/ by ingest step."""
    mlflow.set_experiment("test_artifacts")
    with mlflow.start_run() as run:
        ingest(raw_csv)

    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "raw_data")]
    assert len(artifacts) > 0
