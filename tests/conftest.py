import pytest
import pandas as pd
import numpy as np


NUMERIC_COLUMNS = [
    "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests",
    "agent", "company",
]

CATEGORICAL_COLUMNS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "assigned_room_type",
    "deposit_type", "customer_type",
]

TARGET_COLUMN = "is_canceled"

# All columns after leaky ones are dropped (30 columns)
ALL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]


def _make_hotel_rows(n: int, canceled: int = 0) -> dict:
    """Return a dict of column → list for n hotel booking rows."""
    return {
        "hotel":                          ["City Hotel"] * n,
        "is_canceled":                    [canceled] * n,
        "lead_time":                      [20] * n,
        "arrival_date_year":              [2016] * n,
        "arrival_date_month":             ["July"] * n,
        "arrival_date_week_number":       [27] * n,
        "arrival_date_day_of_month":      [1] * n,
        "stays_in_weekend_nights":        [1] * n,
        "stays_in_week_nights":           [2] * n,
        "adults":                         [2] * n,
        "children":                       [0.0] * n,
        "babies":                         [0] * n,
        "meal":                           ["BB"] * n,
        "country":                        ["PRT"] * n,
        "market_segment":                 ["Online TA"] * n,
        "distribution_channel":           ["TA/TO"] * n,
        "is_repeated_guest":              [0] * n,
        "previous_cancellations":         [0] * n,
        "previous_bookings_not_canceled": [0] * n,
        "reserved_room_type":             ["A"] * n,
        "assigned_room_type":             ["A"] * n,
        "booking_changes":                [0] * n,
        "deposit_type":                   ["No Deposit"] * n,
        "agent":                          [9.0] * n,
        "company":                        [0.0] * n,
        "days_in_waiting_list":           [0] * n,
        "customer_type":                  ["Transient"] * n,
        "adr":                            [100.0] * n,
        "required_car_parking_spaces":    [0] * n,
        "total_of_special_requests":      [0] * n,
    }


@pytest.fixture
def valid_hotel_df():
    """Minimal valid hotel bookings DataFrame — 8 clean rows, no nulls."""
    rows = _make_hotel_rows(8)
    # Mix canceled / not-canceled
    rows["is_canceled"] = [0, 1, 0, 1, 0, 0, 1, 0]
    return pd.DataFrame(rows)


@pytest.fixture
def dirty_hotel_df(valid_hotel_df):
    """DataFrame with known dirty data: duplicates, nulls, and outlier rows."""
    df = valid_hotel_df.copy()

    # Duplicate first row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    # Row with null country
    null_country = df.iloc[0].copy()
    null_country["country"] = np.nan
    df = pd.concat([df, null_country.to_frame().T], ignore_index=True)

    # Row with null children
    null_children = df.iloc[0].copy()
    null_children["children"] = np.nan
    df = pd.concat([df, null_children.to_frame().T], ignore_index=True)

    # Row with outlier adults (55)
    outlier_adults = df.iloc[0].copy()
    outlier_adults["adults"] = 55
    df = pd.concat([df, outlier_adults.to_frame().T], ignore_index=True)

    # Row with negative adr
    neg_adr = df.iloc[0].copy()
    neg_adr["adr"] = -10.0
    df = pd.concat([df, neg_adr.to_frame().T], ignore_index=True)

    return df


@pytest.fixture
def raw_csv_with_leaky(tmp_path, valid_hotel_df):
    """Write a CSV that includes leaky columns (as the real dataset does)."""
    df = valid_hotel_df.copy()
    df["reservation_status"] = "Check-Out"
    df["reservation_status_date"] = "2016-07-03"
    path = str(tmp_path / "hotel_bookings.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def raw_csv(tmp_path, valid_hotel_df):
    """Write valid_hotel_df to a temp CSV (no leaky columns) and return its path."""
    path = str(tmp_path / "hotel_bookings.csv")
    valid_hotel_df.to_csv(path, index=False)
    return path


@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Isolated MLflow tracking URI for test runs — never touches mlflow.db."""
    import mlflow
    uri = f"sqlite:///{tmp_path}/test_mlflow.db"
    mlflow.set_tracking_uri(uri)
    return uri
