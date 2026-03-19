import logging
import tempfile
import os
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns expected in the dataset
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET_COLUMN = "quality"

# Scientifically plausible clip ranges for red wine physicochemical properties.
# Sources: wine chemistry literature and UCI dataset observed ranges with margin.
FEATURE_CLIP_RANGES = {
    "fixed acidity":        (1.0,  20.0),
    "volatile acidity":     (0.05,  2.0),
    "citric acid":          (0.0,   1.5),
    "residual sugar":       (0.5,  30.0),
    "chlorides":            (0.01,  0.7),
    "free sulfur dioxide":  (0.0,  100.0),
    "total sulfur dioxide": (0.0,  350.0),
    "density":              (0.985, 1.005),
    "pH":                   (2.5,   4.5),
    "sulphates":            (0.2,   2.5),
    "alcohol":              (7.0,  15.0),
}

QUALITY_MIN = 3
QUALITY_MAX = 9


def clean(df: pd.DataFrame, cleaned_filename: str = "cleaned_wine.csv") -> pd.DataFrame:
    """Clean and strongly type the raw wine DataFrame.

    Steps:
        1. Strip whitespace from column names.
        2. Cast feature columns to float64.
        3. Drop rows with null or out-of-range quality.
        4. Cast quality to int8.
        5. Drop fully duplicate rows.
        6. Clip feature columns to plausible physical ranges.
        7. Log cleaning metrics and the cleaned CSV as MLflow artifacts.

    Args:
        df: Raw DataFrame from ingest step.
        cleaned_filename: Filename used when saving the cleaned artifact.

    Returns:
        Cleaned, strongly-typed DataFrame.
    """
    df = df.copy()
    rows_before = len(df)
    logger.info("Starting cleaning — %d rows", rows_before)

    # 1. Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Cast feature columns to float64
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # 3. Drop rows with null or out-of-range quality
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN].between(QUALITY_MIN, QUALITY_MAX)]

    # 4. Cast quality to int8
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")

    # 5. Drop fully duplicate rows
    df = df.drop_duplicates()

    # 6. Clip feature columns to plausible physical ranges
    for col, (low, high) in FEATURE_CLIP_RANGES.items():
        df[col] = df[col].clip(lower=low, upper=high)

    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    logger.info("Cleaning complete — %d rows remaining, %d dropped", rows_after, rows_dropped)

    # Log cleaning metrics to MLflow
    mlflow.log_metric("rows_before", rows_before)
    mlflow.log_metric("rows_after", rows_after)
    mlflow.log_metric("rows_dropped", rows_dropped)
    mlflow.log_metric("columns", len(df.columns))

    # Save cleaned CSV and log as artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        cleaned_path = os.path.join(tmpdir, cleaned_filename)
        df.to_csv(cleaned_path, index=False)
        mlflow.log_artifact(cleaned_path, artifact_path="cleaned_data")
        logger.info("Logged cleaned CSV artifact: %s", cleaned_filename)

    return df
