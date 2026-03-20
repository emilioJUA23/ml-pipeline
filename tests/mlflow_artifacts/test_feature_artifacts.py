import pytest
import mlflow
from steps.feature_engineer import feature_engineer


@pytest.fixture
def varied_hotel_df(valid_hotel_df):
    df = valid_hotel_df.copy()
    df["deposit_type"] = ["No Deposit", "Non Refund", "Refundable",
                          "No Deposit", "Non Refund", "Refundable",
                          "No Deposit", "Non Refund"]
    df["market_segment"] = ["Online TA", "Corporate", "Direct", "Groups",
                            "Offline TA/TO", "Aviation", "Online TA", "Corporate"]
    df["customer_type"] = ["Transient", "Contract", "Group", "Transient-Party",
                           "Transient", "Contract", "Group", "Transient-Party"]
    df["meal"] = ["BB", "HB", "SC", "FB", "BB", "HB", "SC", "Undefined"]
    df["distribution_channel"] = ["TA/TO", "Direct", "Corporate", "GDS",
                                   "TA/TO", "Direct", "Corporate", "GDS"]
    df["country"] = ["PRT", "GBR", "FRA", "ESP", "PRT", "GBR", "FRA", "ESP"]
    return df


def test_feature_engineering_metrics_logged(varied_hotel_df, mlflow_tracking_uri):
    """feature_count and engineered_rows are logged as MLflow metrics."""
    mlflow.set_experiment("test_fe_artifacts")
    with mlflow.start_run() as run:
        feature_engineer(varied_hotel_df)

    data = mlflow.get_run(run.info.run_id).data
    assert "feature_count" in data.metrics
    assert "engineered_rows" in data.metrics


def test_feature_count_metric_is_positive(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe_artifacts")
    with mlflow.start_run() as run:
        feature_engineer(varied_hotel_df)

    metrics = mlflow.get_run(run.info.run_id).data.metrics
    assert metrics["feature_count"] > 0


def test_engineered_rows_matches_input(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe_artifacts")
    with mlflow.start_run() as run:
        feature_engineer(varied_hotel_df)

    metrics = mlflow.get_run(run.info.run_id).data.metrics
    assert metrics["engineered_rows"] == len(varied_hotel_df)


def test_engineered_csv_artifact_logged(varied_hotel_df, mlflow_tracking_uri):
    """Engineered CSV is logged under engineered_data/ in MLflow."""
    mlflow.set_experiment("test_fe_artifacts")
    with mlflow.start_run() as run:
        feature_engineer(varied_hotel_df, engineered_filename="engineered_hotel_bookings.csv")

    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "engineered_data")]
    assert any("engineered_hotel_bookings.csv" in a for a in artifacts)
