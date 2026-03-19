import logging
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
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
    "quality",
}


def ingest(raw_path: str) -> pd.DataFrame:
    """Load raw CSV and log it as the first pipeline artifact.

    Args:
        raw_path: Path to the raw CSV file.

    Returns:
        Raw DataFrame as loaded from disk.

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Ingesting raw data from: %s", raw_path)

    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    # Normalise column names before schema check
    df.columns = [c.strip() for c in df.columns]

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing expected columns: {missing}")

    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), raw_path)

    # Log raw file as the first pipeline artifact
    mlflow.log_artifact(raw_path, artifact_path="raw_data")
    logger.info("Logged raw CSV artifact: %s", raw_path)

    return df
