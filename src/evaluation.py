"""Shared evaluation; missing labels are allowed, missing predictions are errors."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_mase_scale(frame: pd.DataFrame, lag: int = 24) -> float:
    """Mean training error of 24-hour differences on the full hourly grid.

    This is a historical scaling reference, not the held-out error of a
    repeat-last-day forecast. Use the same reference history for all candidates.
    """
    if not {"ds", "y"}.issubset(frame.columns):
        raise ValueError("MASE requires ds and y columns.")
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("MASE lag must be a positive integer.")
    data = frame.sort_values("ds").copy()
    data["ds"] = pd.to_datetime(data["ds"], utc=True, errors="raise")
    if data["ds"].isna().any() or not data["ds"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("MASE requires a complete consecutive hourly timeline.")
    y = pd.to_numeric(data["y"], errors="raise")
    if np.isinf(y.to_numpy(dtype=float)).any():
        raise ValueError("Infinite training targets are invalid.")
    scale = (y - y.shift(lag)).abs().mean()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MASE scale is zero or invalid.")
    return float(scale)


def calculate_metrics(y_true, y_pred, mase_scale: float) -> dict:
    """Score positional pairs, excluding only missing actual targets.

    Use evaluate_forecast for timestamp alignment. This positional helper is
    retained for the existing Prophet experiment notebook.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Targets and predictions must be equal-length 1D arrays.")
    if not np.isfinite(predicted).all():
        raise ValueError("Every requested hour must have a finite prediction.")
    if np.isinf(actual).any():
        raise ValueError("Infinite actual temperatures are invalid.")
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise ValueError("MASE scale must be finite and positive.")
    observed = ~np.isnan(actual)
    if not observed.any():
        raise ValueError("No observed targets to evaluate.")
    scored_actual, scored_predicted = actual[observed], predicted[observed]
    mae = mean_absolute_error(scored_actual, scored_predicted)
    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mean_squared_error(scored_actual, scored_predicted))),
        "mase": float(mae / mase_scale),
        "r2": float(r2_score(scored_actual, scored_predicted)) if observed.sum() > 1 else np.nan,
        "bias": float((scored_predicted - scored_actual).mean()),
        "hours_expected": len(actual),
        "hours_observed": int(observed.sum()),
        "hours_evaluated": int(observed.sum()),
    }


def validate_predictions(predictions: pd.DataFrame, expected_times) -> pd.DataFrame:
    """Require one finite prediction per expected timestamp; return grid order."""
    if not {"ds", "yhat"}.issubset(predictions.columns):
        raise ValueError("Predictions require ds and yhat columns.")
    data = predictions.copy()
    data["ds"] = pd.to_datetime(data["ds"], utc=True, errors="raise")
    expected = pd.DatetimeIndex(pd.to_datetime(expected_times, utc=True))
    if expected.hasnans or expected.has_duplicates:
        raise ValueError("Expected timestamps must be nonmissing and unique.")
    if data["ds"].isna().any() or data["ds"].duplicated().any():
        raise ValueError("Prediction timestamps must be nonmissing and unique.")
    received = pd.DatetimeIndex(data["ds"])
    if len(received) != len(expected) or not received.sort_values().equals(expected.sort_values()):
        raise ValueError("Prediction timestamps must exactly match the requested grid.")
    data["yhat"] = pd.to_numeric(data["yhat"], errors="raise")
    if not np.isfinite(data["yhat"].to_numpy(dtype=float)).all():
        raise ValueError("Every requested hour must have a finite prediction.")
    return data.set_index("ds").reindex(expected).rename_axis("ds").reset_index()


def evaluate_forecast(actual: pd.DataFrame, predictions: pd.DataFrame, mase_scale: float):
    """Align on timestamps, check completeness, and return metrics plus pairs."""
    truth = actual[["ds", "y"]].copy()
    truth["ds"] = pd.to_datetime(truth["ds"], utc=True, errors="raise")
    forecast = validate_predictions(predictions, truth["ds"])
    if "y" in forecast.columns:
        raise ValueError("Prediction frames must not contain actual targets.")
    comparison = truth.merge(forecast, on="ds", how="left", validate="one_to_one")
    metrics = calculate_metrics(comparison["y"], comparison["yhat"], mase_scale)
    return metrics, comparison
