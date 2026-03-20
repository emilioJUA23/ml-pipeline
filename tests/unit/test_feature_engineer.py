import pytest
import numpy as np
import pandas as pd
import mlflow
from steps.feature_engineer import (
    feature_engineer,
    TARGET_COLUMN,
    MONTH_MAP,
    SEASON_MAP,
    COLUMNS_TO_DROP,
    ONE_HOT_COLUMNS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def varied_hotel_df(valid_hotel_df):
    """Extend the base fixture with varied categoricals so one-hot encoding fires."""
    df = valid_hotel_df.copy()
    # Give each row a different deposit_type / market_segment / customer_type / meal
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
    df["reserved_room_type"] = ["A", "A", "B", "A", "A", "B", "A", "A"]
    df["assigned_room_type"] = ["A", "B", "B", "A", "A", "A", "B", "A"]
    return df


# ── Happy path ────────────────────────────────────────────────────────────────

def test_feature_engineer_returns_dataframe(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(varied_hotel_df)


def test_feature_engineer_does_not_modify_input(varied_hotel_df, mlflow_tracking_uri):
    original_cols = list(varied_hotel_df.columns)
    original_len = len(varied_hotel_df)
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        feature_engineer(varied_hotel_df)
    assert list(varied_hotel_df.columns) == original_cols
    assert len(varied_hotel_df) == original_len


def test_output_has_more_columns_than_input(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert len(result.columns) > len(varied_hotel_df.columns)


# ── No string columns ─────────────────────────────────────────────────────────

def test_no_object_columns_in_output(varied_hotel_df, mlflow_tracking_uri):
    """All columns must be numeric after feature engineering."""
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    object_cols = result.select_dtypes(include="object").columns.tolist()
    assert object_cols == [], f"Found string columns: {object_cols}"


# ── Dropped columns ───────────────────────────────────────────────────────────

def test_original_categorical_columns_are_dropped(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    for col in COLUMNS_TO_DROP + ONE_HOT_COLUMNS:
        assert col not in result.columns, f"Column '{col}' should have been dropped"


def test_target_column_preserved(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert TARGET_COLUMN in result.columns


# ── Derived features present ──────────────────────────────────────────────────

@pytest.mark.parametrize("col", [
    "arrival_month_num", "arrival_season", "is_high_season", "arrival_day_of_week",
    "lead_time_bucket", "is_long_lead",
    "total_nights", "total_guests", "is_family", "revenue_estimate", "is_zero_night",
    "has_prev_cancel", "prev_cancel_rate",
    "room_type_match", "has_special_requests", "has_parking",
    "has_booking_changes", "has_waiting_list",
    "is_direct_booking", "is_company_booking",
    "hotel_city", "country_encoded",
])
def test_derived_feature_present(varied_hotel_df, mlflow_tracking_uri, col):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert col in result.columns, f"Expected feature '{col}' not found"


# ── Feature value correctness ─────────────────────────────────────────────────

def test_total_nights_correct(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    expected = varied_hotel_df["stays_in_weekend_nights"] + varied_hotel_df["stays_in_week_nights"]
    assert (result["total_nights"].values == expected.values).all()


def test_total_guests_correct(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    expected = varied_hotel_df["adults"] + varied_hotel_df["children"] + varied_hotel_df["babies"]
    assert (result["total_guests"].values == expected.values).all()


def test_room_type_match_correct(varied_hotel_df, mlflow_tracking_uri):
    """room_type_match=1 when reserved==assigned, 0 otherwise."""
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    expected = (varied_hotel_df["reserved_room_type"] == varied_hotel_df["assigned_room_type"]).astype("int8")
    assert (result["room_type_match"].values == expected.values).all()


def test_has_prev_cancel_correct(mlflow_tracking_uri, valid_hotel_df):
    df = valid_hotel_df.copy()
    df["previous_cancellations"] = [0, 1, 0, 2, 0, 0, 3, 0]
    df["deposit_type"] = "No Deposit"
    df["market_segment"] = "Online TA"
    df["customer_type"] = "Transient"
    df["meal"] = "BB"
    df["distribution_channel"] = "TA/TO"
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(df)
    expected = (df["previous_cancellations"] > 0).astype("int8")
    assert (result["has_prev_cancel"].values == expected.values).all()


def test_prev_cancel_rate_zero_when_no_history(varied_hotel_df, mlflow_tracking_uri):
    df = varied_hotel_df.copy()
    df["previous_cancellations"] = 0
    df["previous_bookings_not_canceled"] = 0
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(df)
    assert (result["prev_cancel_rate"] == 0.0).all()


def test_revenue_estimate_correct(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    total_nights = varied_hotel_df["stays_in_weekend_nights"] + varied_hotel_df["stays_in_week_nights"]
    expected = varied_hotel_df["adr"] * total_nights
    np.testing.assert_array_almost_equal(result["revenue_estimate"].values, expected.values)


def test_hotel_city_flag(varied_hotel_df, mlflow_tracking_uri):
    df = varied_hotel_df.copy()
    df["hotel"] = ["City Hotel", "Resort Hotel", "City Hotel", "Resort Hotel",
                   "City Hotel", "City Hotel", "Resort Hotel", "City Hotel"]
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(df)
    expected = (df["hotel"] == "City Hotel").astype("int8")
    assert (result["hotel_city"].values == expected.values).all()


def test_arrival_month_num_correct(varied_hotel_df, mlflow_tracking_uri):
    """arrival_month_num maps month names to 1–12."""
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    expected_month = MONTH_MAP[varied_hotel_df["arrival_date_month"].iloc[0]]
    assert result["arrival_month_num"].iloc[0] == expected_month


def test_lead_time_bucket_range(varied_hotel_df, mlflow_tracking_uri):
    """lead_time_bucket values are in [0, 5]."""
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert result["lead_time_bucket"].between(0, 5).all()


def test_binary_features_are_0_or_1(varied_hotel_df, mlflow_tracking_uri):
    binary_cols = [
        "is_high_season", "is_long_lead", "is_family", "is_zero_night",
        "has_prev_cancel", "room_type_match", "has_special_requests",
        "has_parking", "has_booking_changes", "has_waiting_list",
        "is_direct_booking", "is_company_booking", "hotel_city",
    ]
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    for col in binary_cols:
        assert set(result[col].unique()).issubset({0, 1}), f"{col} has values outside {{0,1}}"


def test_country_encoded_is_numeric(varied_hotel_df, mlflow_tracking_uri):
    mlflow.set_experiment("test_fe")
    with mlflow.start_run():
        result = feature_engineer(varied_hotel_df)
    assert pd.api.types.is_numeric_dtype(result["country_encoded"])
    assert result["country_encoded"].between(0.0, 1.0).all()
