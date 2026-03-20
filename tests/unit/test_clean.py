import pytest
import numpy as np
import pandas as pd
import mlflow
from steps.clean import (
    clean, TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    NULL_FILL_ZERO, NULL_FILL_UNKNOWN, OUTLIER_BOUNDS,
)


def test_clean_happy_path(valid_hotel_df, mlflow_tracking_uri):
    """Clean DataFrame returns without error and preserves columns."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(valid_hotel_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_clean_target_dtype(valid_hotel_df, mlflow_tracking_uri):
    """is_canceled is int8 after cleaning."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(valid_hotel_df)
    assert result[TARGET_COLUMN].dtype == np.int8


def test_clean_fills_null_children(dirty_hotel_df, mlflow_tracking_uri):
    """Null children values are filled with 0."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_hotel_df)
    assert result["children"].isnull().sum() == 0


def test_clean_fills_null_country(dirty_hotel_df, mlflow_tracking_uri):
    """Null country values are filled with 'Unknown'."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_hotel_df)
    assert result["country"].isnull().sum() == 0
    assert "Unknown" in result["country"].values or result["country"].isnull().sum() == 0


def test_clean_drops_outlier_adults(dirty_hotel_df, mlflow_tracking_uri):
    """Rows with adults > 10 are removed."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_hotel_df)
    low, high = OUTLIER_BOUNDS["adults"]
    assert result["adults"].between(low, high).all()


def test_clean_drops_negative_adr(dirty_hotel_df, mlflow_tracking_uri):
    """Rows with adr < 0 are removed."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_hotel_df)
    assert (result["adr"] >= 0).all()


def test_clean_drops_duplicates(dirty_hotel_df, mlflow_tracking_uri):
    """Fully duplicate rows are removed."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_hotel_df)
    assert result.duplicated().sum() == 0


def test_clean_rows_never_increase(dirty_hotel_df, mlflow_tracking_uri):
    """Cleaning never adds rows."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_hotel_df)
    assert len(result) <= len(dirty_hotel_df)


def test_clean_does_not_modify_input(valid_hotel_df, mlflow_tracking_uri):
    """clean() does not mutate the input DataFrame."""
    original_len = len(valid_hotel_df)
    original_cols = list(valid_hotel_df.columns)
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        clean(valid_hotel_df)
    assert len(valid_hotel_df) == original_len
    assert list(valid_hotel_df.columns) == original_cols


def test_clean_column_names_no_whitespace(valid_hotel_df, mlflow_tracking_uri):
    """Column names have no leading/trailing whitespace after cleaning."""
    df_spaces = valid_hotel_df.copy()
    df_spaces.columns = [f" {c} " for c in df_spaces.columns]
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(df_spaces)
    for col in result.columns:
        assert col == col.strip()


@pytest.mark.parametrize("col,bounds", OUTLIER_BOUNDS.items())
def test_clean_outlier_bounds_enforced(valid_hotel_df, mlflow_tracking_uri, col, bounds):
    """All rows fall within defined outlier bounds after cleaning."""
    low, high = bounds
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(valid_hotel_df)
    assert result[col].between(low, high).all(), f"{col} has values outside [{low}, {high}]"
