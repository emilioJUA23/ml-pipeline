import mlflow
from steps.ingest import ingest, LEAKY_COLUMNS
from steps.clean import clean, NUMERIC_COLUMNS, TARGET_COLUMN

REAL_CSV = "data/raw/hotel_bookings.csv"


def test_ingest_real_csv(mlflow_tracking_uri):
    """Ingest loads the real dataset without error and drops leaky columns."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run():
        df = ingest(REAL_CSV)
    assert len(df) > 10000
    for col in LEAKY_COLUMNS:
        assert col not in df.columns


def test_clean_real_csv(mlflow_tracking_uri):
    """Ingest + clean on real CSV produces a valid typed DataFrame."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run():
        raw = ingest(REAL_CSV)
        df = clean(raw)

    assert len(df) > 0
    assert df[TARGET_COLUMN].dtype.name == "int8"
    assert df.isnull().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_clean_real_csv_reduces_rows(mlflow_tracking_uri):
    """Cleaning removes at least some rows from the real dataset (outliers/dupes)."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run():
        raw = ingest(REAL_CSV)
        df = clean(raw)

    assert len(df) < len(raw)


def test_clean_artifacts_exist_on_disk(mlflow_tracking_uri):
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
