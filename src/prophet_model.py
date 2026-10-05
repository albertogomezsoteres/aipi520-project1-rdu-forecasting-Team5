
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

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Preserve python src/prophet_model.py as well as python -m src.prophet_model.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AUDIT_WINDOW, DATA_DIR, OUTPUT_DIR
from src.evaluation import evaluate_forecast, make_mase_scale

BEST_PROPHET_PARAMS = {
    "changepoint_prior_scale": 0.005,
    "seasonality_prior_scale": 1.0,
    "weekly_seasonality": True,
    "daily_fourier_order": 16,
    "yearly_fourier_order": 20,
}


def build_model(params=None):
    """Build the original additive daily/yearly Prophet model."""
    config = dict(BEST_PROPHET_PARAMS)
    if params is not None:
        if set(params).difference(config):
            raise ValueError("Unsupported Prophet configuration keys.")
        config.update(params)
    try:
        from prophet import Prophet
    except ImportError as error:
        raise ImportError("Install requirements.txt before running Prophet.") from error
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=False,
        weekly_seasonality=config["weekly_seasonality"],
        changepoint_prior_scale=config["changepoint_prior_scale"],
        seasonality_prior_scale=config["seasonality_prior_scale"],
        seasonality_mode="additive",
        interval_width=0.8,
    )
    model.add_seasonality(name="daily", period=1, fourier_order=config["daily_fourier_order"])
    model.add_seasonality(name="yearly", period=365.25, fourier_order=config["yearly_fourier_order"])
    return model


def predict_prophet(history: pd.DataFrame, future: pd.DataFrame, params=None) -> pd.DataFrame:
    """Fit on observed pre-origin labels; remove UTC markers only inside Prophet."""
    if "y" in future.columns:
        raise ValueError("Future inputs must not contain actual temperatures.")
    train = history[["ds", "y"]].copy()
    target = future[["ds"]].copy()
    train["ds"] = pd.to_datetime(train["ds"], utc=True, errors="raise")
    target["ds"] = pd.to_datetime(target["ds"], utc=True, errors="raise")
    if train.empty or target.empty or train["ds"].max() >= target["ds"].min():
        raise ValueError("Prophet training observations must precede the forecast origin.")
    train = train.dropna(subset=["y"]).sort_values("ds")
    train["ds"] = train["ds"].dt.tz_localize(None)
    target["ds"] = target["ds"].dt.tz_localize(None)
    model = build_model(params)
    model.fit(train)
    predictions = model.predict(target)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    predictions["ds"] = pd.to_datetime(predictions["ds"], utc=True)
    return predictions


def main():
    from src.backtesting import make_fold, select_history
    from src.data_pipeline import load_hourly_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "hourly.csv")
    parser.add_argument("--lookback-years", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "predictions/prophet_test_predictions.csv")
    args = parser.parse_args()
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    hourly = load_hourly_data(args.data)
    history, future, actual = make_fold(hourly, AUDIT_WINDOW, args.lookback_years)
    reference = select_history(hourly, AUDIT_WINDOW.start)
    metrics, comparison = evaluate_forecast(actual, predict_prophet(history, future), make_mase_scale(reference))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)
    metadata = {"role": "historical_audit", "params": BEST_PROPHET_PARAMS,
                "lookback_years": args.lookback_years, "metrics": metrics}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()