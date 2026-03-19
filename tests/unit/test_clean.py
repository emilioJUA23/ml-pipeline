import pytest
import numpy as np
import pandas as pd
import mlflow
from steps.clean import clean, FEATURE_COLUMNS, TARGET_COLUMN, QUALITY_MIN, QUALITY_MAX


def test_clean_happy_path(valid_wine_df, mlflow_tracking_uri):
    """Clean DataFrame returns without error and has same columns."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(valid_wine_df)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(valid_wine_df.columns)


@pytest.mark.parametrize("col", FEATURE_COLUMNS)
def test_clean_feature_dtypes(valid_wine_df, mlflow_tracking_uri, col):
    """All feature columns are float64 after cleaning."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(valid_wine_df)
    assert result[col].dtype == np.float64, f"Column '{col}' should be float64"


def test_clean_quality_dtype(valid_wine_df, mlflow_tracking_uri):
    """quality column is int8 after cleaning."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(valid_wine_df)
    assert result[TARGET_COLUMN].dtype == np.int8


def test_clean_drops_duplicates(dirty_wine_df, mlflow_tracking_uri):
    """Fully duplicate rows are removed."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_wine_df)
    assert result.duplicated().sum() == 0


def test_clean_drops_out_of_range_quality(dirty_wine_df, mlflow_tracking_uri):
    """Rows with quality outside [3, 9] are dropped."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_wine_df)
    assert result[TARGET_COLUMN].between(QUALITY_MIN, QUALITY_MAX).all()


def test_clean_drops_null_quality(dirty_wine_df, mlflow_tracking_uri):
    """Rows with null quality are dropped."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_wine_df)
    assert result[TARGET_COLUMN].isnull().sum() == 0


def test_clean_rows_never_increase(dirty_wine_df, mlflow_tracking_uri):
    """Cleaning never adds rows — output length <= input length."""
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(dirty_wine_df)
    assert len(result) <= len(dirty_wine_df)


def test_clean_column_names_no_whitespace(valid_wine_df, mlflow_tracking_uri):
    """Column names have no leading/trailing whitespace after cleaning."""
    df_spaces = valid_wine_df.copy()
    df_spaces.columns = [f" {c} " for c in df_spaces.columns]
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        result = clean(df_spaces)
    for col in result.columns:
        assert col == col.strip()


def test_clean_does_not_modify_input(valid_wine_df, mlflow_tracking_uri):
    """clean() does not mutate the input DataFrame."""
    original_len = len(valid_wine_df)
    original_cols = list(valid_wine_df.columns)
    mlflow.set_experiment("test_clean")
    with mlflow.start_run():
        clean(valid_wine_df)
    assert len(valid_wine_df) == original_len
    assert list(valid_wine_df.columns) == original_cols
