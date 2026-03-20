import logging
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

# All 32 columns expected in the raw hotel bookings CSV
EXPECTED_COLUMNS = {
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
}

# These columns directly encode the target — must be dropped to prevent data leakage
LEAKY_COLUMNS = ["reservation_status", "reservation_status_date"]


def ingest(raw_path: str) -> pd.DataFrame:
    """Load raw hotel bookings CSV, validate schema, drop leaky columns, and log artifact.

    Args:
        raw_path: Path to the raw CSV file.

    Returns:
        Raw DataFrame with leaky columns removed.

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If required columns are missing from the CSV.
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

    # Log raw file as the first pipeline artifact before dropping anything
    mlflow.log_artifact(raw_path, artifact_path="raw_data")
    logger.info("Logged raw CSV artifact: %s", raw_path)

    # Drop leaky columns — reservation_status encodes the target directly
    df = df.drop(columns=LEAKY_COLUMNS)
    logger.info("Dropped leaky columns: %s", LEAKY_COLUMNS)

    return df
