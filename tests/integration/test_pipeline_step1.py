import mlflow
from steps.ingest import ingest, LEAKY_COLUMNS
from steps.clean import clean, TARGET_COLUMN
from steps.feature_engineer import feature_engineer

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


def test_feature_engineer_real_csv(mlflow_tracking_uri):
    """Full pipeline ingest → clean → feature_engineer runs without error on real data."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run():
        raw = ingest(REAL_CSV)
        cleaned = clean(raw)
        df = feature_engineer(cleaned)

    assert len(df) == len(cleaned)
    assert df.isnull().sum().sum() == 0
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns after feature engineering: {non_numeric}"


def test_feature_count_real_csv(mlflow_tracking_uri):
    """Feature engineering produces more columns than cleaning."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run():
        raw = ingest(REAL_CSV)
        cleaned = clean(raw)
        engineered = feature_engineer(cleaned)

    assert len(engineered.columns) > len(cleaned.columns)


def test_all_artifacts_exist_on_disk(mlflow_tracking_uri):
    """raw_data, cleaned_data, and engineered_data artifacts are all logged."""
    mlflow.set_experiment("test_integration")
    with mlflow.start_run() as run:
        raw = ingest(REAL_CSV)
        cleaned = clean(raw)
        feature_engineer(cleaned)

    client = mlflow.tracking.MlflowClient()
    assert len(client.list_artifacts(run.info.run_id, "raw_data")) > 0
    assert len(client.list_artifacts(run.info.run_id, "cleaned_data")) > 0
    assert len(client.list_artifacts(run.info.run_id, "engineered_data")) > 0
