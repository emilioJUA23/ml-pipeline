import logging
import os
import tempfile
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "is_canceled"

# Numeric columns (excluding target)
NUMERIC_COLUMNS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "agent",
    "company",
]

# Categorical columns (string-typed)
CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]

# Columns where NaN means "none" (no agent, no company) — fill with 0
NULL_FILL_ZERO = ["children", "agent", "company"]

# Columns where NaN means unknown origin — fill with sentinel string
NULL_FILL_UNKNOWN = ["country"]

# Plausible bounds for outlier removal
OUTLIER_BOUNDS = {
    "adults": (1, 10),               # max 10 adults; 0-adult rows are data errors
    "adr": (0.0, 2000.0),            # negative rates are impossible; >2000 is extreme
    "stays_in_week_nights": (0, 30), # >30 week nights is implausible
    "stays_in_weekend_nights": (0, 10),
}


def clean(df: pd.DataFrame, cleaned_filename: str = "cleaned_hotel_bookings.csv") -> pd.DataFrame:
    """Clean and type-enforce the raw hotel bookings DataFrame.

    Steps:
        1. Strip whitespace from column names.
        2. Fill known NaN-means-zero columns with 0.
        3. Fill known NaN-means-unknown categoricals with "Unknown".
        4. Remove rows outside plausible bounds (outliers / data errors).
        5. Drop fully duplicate rows.
        6. Enforce dtypes: target → int8, numeric cols → correct types.
        7. Log cleaning metrics and the cleaned CSV as MLflow artifacts.

    Args:
        df: Raw DataFrame from ingest step (leaky columns already dropped).
        cleaned_filename: Filename used when saving the cleaned artifact.

    Returns:
        Cleaned, typed DataFrame ready for feature engineering.
    """
    df = df.copy()
    rows_before = len(df)
    logger.info("Starting cleaning — %d rows", rows_before)

    # 1. Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Fill NaN-means-zero columns
    for col in NULL_FILL_ZERO:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 3. Fill NaN-means-unknown categoricals
    for col in NULL_FILL_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 4. Remove outlier / data-error rows
    for col, (low, high) in OUTLIER_BOUNDS.items():
        if col in df.columns:
            before = len(df)
            df = df[df[col].between(low, high)]
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with %s outside [%s, %s]", dropped, col, low, high)

    # 5. Drop fully duplicate rows
    df = df.drop_duplicates()

    # 6. Enforce dtypes
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    logger.info("Cleaning complete — %d rows remaining, %d dropped", rows_after, rows_dropped)

    # 7. Log cleaning metrics to MLflow
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
