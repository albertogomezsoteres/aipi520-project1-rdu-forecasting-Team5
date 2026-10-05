"""Shared forecasting evaluation helpers for all project models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_mase_scale(frame: pd.DataFrame, lag: int = 24) -> float:
    """Compute a seasonal-naive MASE denominator on a complete hourly timeline."""
    required = {"ds", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = frame.sort_values("ds").copy()
    data["ds"] = pd.to_datetime(data["ds"])

    if not data["ds"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("MASE requires a complete consecutive hourly timeline.")

    y = pd.to_numeric(data["y"], errors="coerce")
    scale = (y - y.shift(lag)).abs().mean()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MASE scale is zero or invalid.")
    return float(scale)


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mase_scale: float,
) -> dict[str, float]:
    """Calculate MAE, RMSE, MASE, R^2, bias, and evaluated row count."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    mask = y_true.notna() & y_pred.notna() & np.isfinite(y_pred)
    actual = y_true[mask]
    predicted = y_pred[mask]

    if actual.empty:
        raise ValueError("No valid prediction/target pairs to evaluate.")

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mase": float(mae / mase_scale),
        "r2": float(r2_score(actual, predicted)),
        "bias": float((predicted - actual).mean()),
        "hours_evaluated": int(mask.sum()),
    }
