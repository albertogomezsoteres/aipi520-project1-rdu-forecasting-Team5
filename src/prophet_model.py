
"""
This final Prophet evaluation script uses the locked parameters selected in
prophet_experiments.ipynb:
- changepoint_prior_scale = 0.005
- seasonality_prior_scale = 1.0
- daily seasonality: custom period=1 day, fourier_order=16
- yearly seasonality: custom period=365.25 days, fourier_order=20
- weekly_seasonality = True
- seasonality_mode = additive, Prophet default

Final held-out test metrics from this locked scheme:
- MAE = 5.446 °F
- RMSE = 6.632 °F
- MASE = 0.930
- R^2 = 0.331
- Bias = -4.648 °F
- Hours evaluated = 336

It trains on train + validation data, predicts the held-out test set, prints
MAE, RMSE, MASE, R^2, bias, and saves test predictions to
outputs/predictions/prophet_test_predictions.csv.

"""

import contextlib
import io
import logging
import os
import tempfile
import warnings
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "prophet-model-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(CACHE_DIR / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import numpy as np
import pandas as pd

with contextlib.redirect_stderr(io.StringIO()):
    from prophet import Prophet

from evaluation import calculate_metrics, make_mase_scale


warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
PREDICTION_PATH = PREDICTION_DIR / "prophet_test_predictions.csv"

BEST_PROPHET_PARAMS = {
    "changepoint_prior_scale": 0.005,
    "seasonality_prior_scale": 1.0,
    "weekly_seasonality": True,
    "daily_fourier_order": 16,
    "yearly_fourier_order": 20,
}


def load_split(name: str) -> pd.DataFrame:
    """Load one processed split and convert UTC timestamps for Prophet."""
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
    return df.sort_values("ds").reset_index(drop=True)


def build_model() -> Prophet:
    """Build the final tuned Prophet model."""
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=False,
        weekly_seasonality=BEST_PROPHET_PARAMS["weekly_seasonality"],
        changepoint_prior_scale=BEST_PROPHET_PARAMS["changepoint_prior_scale"],
        seasonality_prior_scale=BEST_PROPHET_PARAMS["seasonality_prior_scale"],
        seasonality_mode="additive",
        interval_width=0.8,
    )
    model.add_seasonality(
        name="daily",
        period=1,
        fourier_order=BEST_PROPHET_PARAMS["daily_fourier_order"],
    )
    model.add_seasonality(
        name="yearly",
        period=365.25,
        fourier_order=BEST_PROPHET_PARAMS["yearly_fourier_order"],
    )
    return model


def main() -> None:
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    train_val = (
        pd.concat([train, val], ignore_index=True)
        .sort_values("ds")
        .reset_index(drop=True)
    )
    train_val_prophet = train_val[["ds", "y"]].dropna(subset=["y"]).copy()
    future = test[["ds"]].copy()

    mase_scale = make_mase_scale(train_val, lag=24)

    model = build_model()
    model.fit(train_val_prophet)

    forecast = model.predict(future)
    comparison = test[["ds", "y"]].merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left",
        validate="one_to_one",
    )

    metrics = calculate_metrics(
        comparison["y"],
        comparison["yhat"],
        mase_scale,
    )

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(PREDICTION_PATH, index=False)

    print("Final Prophet model parameters")
    print("--------------------------------")
    print(f"changepoint_prior_scale: {BEST_PROPHET_PARAMS['changepoint_prior_scale']}")
    print(f"seasonality_prior_scale: {BEST_PROPHET_PARAMS['seasonality_prior_scale']}")
    print(f"weekly_seasonality: {BEST_PROPHET_PARAMS['weekly_seasonality']}")
    print(f"daily_fourier_order: {BEST_PROPHET_PARAMS['daily_fourier_order']}")
    print(f"yearly_fourier_order: {BEST_PROPHET_PARAMS['yearly_fourier_order']}")
    print("seasonality_mode: additive")
    print()

    print("Data summary")
    print("------------")
    print(f"Train hours: {len(train):,}")
    print(f"Validation hours: {len(val):,}")
    print(f"Train + validation hours: {len(train_val):,}")
    print(f"Test hours: {len(test):,}")
    print(f"Missing train + validation targets: {train_val['y'].isna().sum():,}")
    print(f"Missing test targets: {test['y'].isna().sum():,}")
    print(f"Train range: {train['ds'].min()} to {train['ds'].max()}")
    print(f"Validation range: {val['ds'].min()} to {val['ds'].max()}")
    print(f"Test range: {test['ds'].min()} to {test['ds'].max()}")
    print()

    print("Final test metrics")
    print("------------------")
    print(f"MAE:  {metrics['mae']:.3f} °F")
    print(f"RMSE: {metrics['rmse']:.3f} °F")
    print(f"MASE: {metrics['mase']:.3f}")
    print(f"R^2:  {metrics['r2']:.3f}")
    print(f"Bias: {metrics['bias']:.3f} °F")
    print(f"Hours evaluated: {metrics['hours_evaluated']:,}")
    print()
    print(f"Saved test predictions to: {PREDICTION_PATH}")


if __name__ == "__main__":
    main()
