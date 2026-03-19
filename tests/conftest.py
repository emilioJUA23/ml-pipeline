import os
import pytest
import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
ALL_COLUMNS = FEATURE_COLUMNS + ["quality"]


@pytest.fixture
def valid_wine_df():
    """Minimal valid wine DataFrame — 8 clean rows, correct dtypes."""
    data = {
        "fixed acidity":        [7.4, 7.8, 7.8, 11.2, 7.4, 7.4, 7.9, 8.5],
        "volatile acidity":     [0.70, 0.88, 0.76, 0.28, 0.70, 0.66, 0.60, 0.28],
        "citric acid":          [0.00, 0.00, 0.04, 0.56, 0.00, 0.00, 0.06, 0.56],
        "residual sugar":       [1.9, 2.6, 2.3, 1.9, 1.9, 1.8, 1.6, 1.8],
        "chlorides":            [0.076, 0.098, 0.092, 0.075, 0.076, 0.075, 0.069, 0.092],
        "free sulfur dioxide":  [11.0, 25.0, 15.0, 17.0, 11.0, 13.0, 15.0, 35.0],
        "total sulfur dioxide": [34.0, 67.0, 54.0, 60.0, 34.0, 40.0, 59.0, 103.0],
        "density":              [0.9978, 0.9968, 0.9970, 0.9980, 0.9978, 0.9978, 0.9964, 0.9969],
        "pH":                   [3.51, 3.20, 3.26, 3.16, 3.51, 3.51, 3.30, 3.30],
        "sulphates":            [0.56, 0.68, 0.65, 0.58, 0.56, 0.56, 0.46, 0.75],
        "alcohol":              [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 9.4, 10.5],
        "quality":              [5, 5, 5, 6, 5, 5, 5, 7],
    }
    df = pd.DataFrame(data)
    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype("float64")
    df["quality"] = df["quality"].astype("int8")
    return df


@pytest.fixture
def dirty_wine_df(valid_wine_df):
    """DataFrame with known dirty data: duplicates, out-of-range quality, nulls."""
    df = valid_wine_df.copy().astype({"quality": "float64"})

    # Duplicate first row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    # Row with quality out of range
    bad_quality = df.iloc[0].copy()
    bad_quality["quality"] = 11.0
    df = pd.concat([df, bad_quality.to_frame().T], ignore_index=True)

    # Row with null quality
    null_quality = df.iloc[0].copy()
    null_quality["quality"] = np.nan
    df = pd.concat([df, null_quality.to_frame().T], ignore_index=True)

    return df


@pytest.fixture
def raw_csv(tmp_path, valid_wine_df):
    """Write valid_wine_df to a temp CSV and return its path."""
    path = str(tmp_path / "test_wine.csv")
    valid_wine_df.to_csv(path, index=False)
    return path


@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Isolated MLflow tracking URI for test runs — never touches mlflow.db."""
    import mlflow
    uri = f"sqlite:///{tmp_path}/test_mlflow.db"
    mlflow.set_tracking_uri(uri)
    return uri
