import logging
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "is_canceled"

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# 0=Winter, 1=Spring, 2=Summer, 3=Fall
SEASON_MAP = {
    12: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3,
}

LEAD_TIME_BINS = [0, 7, 30, 90, 180, 365, float("inf")]
LEAD_TIME_LABELS = [0, 1, 2, 3, 4, 5]  # ordinal: 0=same week … 5=over a year

# These are one-hot encoded; drop_first removes the first alphabetical level
ONE_HOT_COLUMNS = [
    "deposit_type",
    "market_segment",
    "customer_type",
    "meal",
    "distribution_channel",
]

# Original columns replaced by engineered features — dropped at the end
COLUMNS_TO_DROP = [
    "hotel",                  # → hotel_city
    "arrival_date_month",     # → arrival_month_num, arrival_season, is_high_season
    "reserved_room_type",     # → room_type_match
    "assigned_room_type",     # → room_type_match
    "country",                # → country_encoded (target encoding)
]


def feature_engineer(
    df: pd.DataFrame,
    engineered_filename: str = "engineered_hotel_bookings.csv",
) -> pd.DataFrame:
    """Build model-ready features from the cleaned hotel bookings DataFrame.

    Groups:
        1. Temporal        — month ordinal, season, high-season flag, day-of-week
        2. Booking behavior — lead-time bucket, long-lead flag
        3. Stay composition — total nights, guests, family flag, revenue estimate
        4. Guest history    — previous cancellation flag and rate
        5. Room & service   — room match, special requests, parking, changes, waitlist
        6. Booking source   — direct booking, company booking flags
        7. Encoding         — hotel binary, country target-encoded, categoricals one-hot

    Args:
        df: Cleaned DataFrame from the clean step.
        engineered_filename: Filename for the MLflow artifact.

    Returns:
        All-numeric DataFrame ready for model training.
    """
    df = df.copy()
    n_rows = len(df)
    n_input_cols = len(df.columns)
    logger.info("Starting feature engineering — %d rows, %d columns", n_rows, n_input_cols)

    # ── Group 1: Temporal ────────────────────────────────────────────────────
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_season"] = df["arrival_month_num"].map(SEASON_MAP)
    df["is_high_season"] = df["arrival_month_num"].isin([7, 8]).astype("int8")

    df["_arrival_date"] = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": df["arrival_month_num"],
            "day": df["arrival_date_day_of_month"],
        },
        errors="coerce",
    )
    df["arrival_day_of_week"] = df["_arrival_date"].dt.dayofweek
    df = df.drop(columns=["_arrival_date"])

    # ── Group 2: Booking Behavior ─────────────────────────────────────────────
    df["lead_time_bucket"] = (
        pd.cut(
            df["lead_time"],
            bins=LEAD_TIME_BINS,
            labels=LEAD_TIME_LABELS,
            right=True,
            include_lowest=True,
        )
        .astype(float)
        .astype("int8")
    )
    df["is_long_lead"] = (df["lead_time"] > 90).astype("int8")

    # ── Group 3: Stay Composition ─────────────────────────────────────────────
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype("int8")
    df["revenue_estimate"] = df["adr"] * df["total_nights"]
    df["is_zero_night"] = (df["total_nights"] == 0).astype("int8")

    # ── Group 4: Guest History ────────────────────────────────────────────────
    df["has_prev_cancel"] = (df["previous_cancellations"] > 0).astype("int8")
    total_prev = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["prev_cancel_rate"] = np.where(
        total_prev > 0,
        df["previous_cancellations"] / total_prev,
        0.0,
    )

    # ── Group 5: Room & Service ───────────────────────────────────────────────
    df["room_type_match"] = (
        df["reserved_room_type"] == df["assigned_room_type"]
    ).astype("int8")
    df["has_special_requests"] = (df["total_of_special_requests"] > 0).astype("int8")
    df["has_parking"] = (df["required_car_parking_spaces"] > 0).astype("int8")
    df["has_booking_changes"] = (df["booking_changes"] > 0).astype("int8")
    df["has_waiting_list"] = (df["days_in_waiting_list"] > 0).astype("int8")

    # ── Group 6: Booking Source ───────────────────────────────────────────────
    df["is_direct_booking"] = ((df["agent"] == 0) & (df["company"] == 0)).astype("int8")
    df["is_company_booking"] = (df["company"] > 0).astype("int8")

    # ── Group 7: Categorical Encoding ─────────────────────────────────────────
    # Hotel → binary flag
    df["hotel_city"] = (df["hotel"] == "City Hotel").astype("int8")

    # Country → target encoding (cancel rate per country; fill unknowns with global mean)
    global_cancel_rate = df[TARGET_COLUMN].mean()
    country_cancel_rate = df.groupby("country")[TARGET_COLUMN].mean()
    df["country_encoded"] = df["country"].map(country_cancel_rate).fillna(global_cancel_rate)

    # Remaining categoricals → one-hot (drop_first avoids perfect multicollinearity)
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True, dtype="int8")

    # Drop original columns now fully replaced by engineered features
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    # Normalise all column names to snake_case (handles spaces, slashes, dashes)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    n_output_cols = len(df.columns)
    logger.info(
        "Feature engineering complete — %d columns (was %d, +%d new features)",
        n_output_cols, n_input_cols, n_output_cols - n_input_cols,
    )

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.log_metric("feature_count", n_output_cols - 1)  # exclude target
    mlflow.log_metric("engineered_rows", n_rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, engineered_filename)
        df.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path="engineered_data")
        logger.info("Logged engineered CSV artifact: %s", engineered_filename)

    return df
