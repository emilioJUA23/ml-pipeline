import pytest
import mlflow
import pandas as pd
from steps.feature_engineer import feature_engineer, TARGET_COLUMN


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


def test_no_nulls_after_feature_engineering(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    null_counts = result.isnull().sum()
    assert null_counts.sum() == 0, f"Nulls found:\n{null_counts[null_counts > 0]}"


def test_target_column_unchanged(varied_hotel_df, mlflow_tracking_uri):
    """is_canceled values must not change during feature engineering."""
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert (result[TARGET_COLUMN].values == varied_hotel_df[TARGET_COLUMN].values).all()


def test_row_count_unchanged(varied_hotel_df, mlflow_tracking_uri):
    """Feature engineering must not add or remove rows."""
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert len(result) == len(varied_hotel_df)


def test_all_columns_are_numeric(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    non_numeric = result.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns remain: {non_numeric}"


def test_column_names_are_snake_case(varied_hotel_df, mlflow_tracking_uri):
    """All column names should be lowercase with underscores only."""
    import re
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    for col in result.columns:
        assert re.match(r"^[a-z0-9_]+$", col), f"Column '{col}' is not snake_case"


def test_arrival_season_values(varied_hotel_df, mlflow_tracking_uri):
    """arrival_season must be in {0, 1, 2, 3}."""
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert set(result["arrival_season"].unique()).issubset({0, 1, 2, 3})


def test_arrival_day_of_week_range(varied_hotel_df, mlflow_tracking_uri):
    """arrival_day_of_week must be 0–6."""
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert result["arrival_day_of_week"].between(0, 6).all()


def test_prev_cancel_rate_between_0_and_1(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert result["prev_cancel_rate"].between(0.0, 1.0).all()


def test_revenue_estimate_non_negative(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe_data")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert (result["revenue_estimate"] >= 0).all()
