"""Training-only benchmarks shared by all model evaluations."""

import numpy as np
import pandas as pd

from .features import add_time_features

BASELINE_NAMES = ("month_hour", "repeat_last_day")


def predict_baseline(history: pd.DataFrame, future: pd.DataFrame, name: str) -> pd.DataFrame:
    """Forecast from pre-cutoff history; never consume future target values.

    Repeat-last-day uses exact timestamps. A missing previous-day temperature
    falls back to the training month/hour mean, then the training global mean.
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline: {name}")
    if "y" in future.columns:
        raise ValueError("Future inputs must not contain actual temperatures.")
    train = add_time_features(history).sort_values("ds")
    targets = add_time_features(future[["ds"]]).sort_values("ds")
    if train.empty or targets.empty or train["ds"].max() >= targets["ds"].min():
        raise ValueError("Benchmark history must precede the forecast origin.")
    if train["ds"].duplicated().any():
        raise ValueError("Benchmark history has duplicate timestamps.")
    observed = train.dropna(subset=["y"])
    if observed.empty or not np.isfinite(observed["y"].to_numpy(dtype=float)).all():
        raise ValueError("Benchmarks require finite observed training targets.")
    climatology = observed.groupby(["month", "hour"])["y"].mean()
    global_mean = float(observed["y"].mean())

    def seasonal_mean(timestamp):
        return float(climatology.get((timestamp.month, timestamp.hour), global_mean))

    fallback_count = 0
    if name == "month_hour":
        values = [seasonal_mean(timestamp) for timestamp in targets["ds"]]
    else:
        origin = targets["ds"].min()
        last_day_times = pd.date_range(origin - pd.Timedelta(hours=24), periods=24, freq="h")
        last_day = train.set_index("ds")["y"].reindex(last_day_times)
        fallback_count = int(last_day.isna().sum())
        for timestamp in last_day.index[last_day.isna()]:
            last_day.loc[timestamp] = seasonal_mean(timestamp)
        # Timestamp arithmetic keeps phase correct, including future frames >24h.
        offsets = ((targets["ds"] - origin) / pd.Timedelta(hours=1)).to_numpy()
        if not np.equal(offsets, np.floor(offsets)).all():
            raise ValueError("Benchmark timestamps must be exact elapsed hours.")
        values = last_day.to_numpy()[offsets.astype(int) % 24]
    predictions = targets[["ds"]].copy()
    predictions["yhat"] = values
    predictions.attrs["last_day_fallback_hours"] = fallback_count
    return predictions
