"""Deterministic shared calendar features; models may add their own features."""

import numpy as np
import pandas as pd

CALENDAR_FEATURES = [
    "hour", "month", "dayofyear", "hour_sin", "hour_cos", "year_sin", "year_cos"
]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the existing UTC calendar and leap-year cyclic encodings."""
    data = frame.copy()
    data["ds"] = pd.to_datetime(data["ds"], utc=True, errors="raise")
    if data["ds"].isna().any():
        raise ValueError("Calendar features require nonmissing timestamps.")
    data["hour"] = data["ds"].dt.hour
    data["month"] = data["ds"].dt.month
    data["dayofyear"] = data["ds"].dt.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    days_in_year = np.where(data["ds"].dt.is_leap_year, 366, 365)
    annual_position = (data["dayofyear"] - 1 + data["hour"] / 24) / days_in_year
    data["year_sin"] = np.sin(2 * np.pi * annual_position)
    data["year_cos"] = np.cos(2 * np.pi * annual_position)
    return data
