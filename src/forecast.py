"""Refit an explicitly selected configuration and generate the 336 submission hours."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from .backtesting import get_predictor, load_selection, select_history
from .config import DATA_DIR, FORECAST_START, OBSERVATION_POLICY, OUTPUT_DIR, SUBMISSION_WINDOW, TARGET_UNIT
from .data_pipeline import load_hourly_data, validate_hourly_data
from .evaluation import validate_predictions
from .features import add_time_features


def generate_forecast(hourly: pd.DataFrame, selection: dict, predictor=None, *, include_weather=False):
    """Use all eligible observations within the selected lookback; no scoring.

    The saved ds values label the containing UTC hours. For example 04:00 UTC
    on September 17 predicts the routine 04:51 UTC / 00:51 Eastern report.
    Optional weather is retained in history only; future observations are unknown.
    """
    data = validate_hourly_data(hourly, include_weather=include_weather)
    history = select_history(data, FORECAST_START, selection.get("lookback_years"))
    future = add_time_features(pd.DataFrame({"ds": SUBMISSION_WINDOW.timestamps()}))
    if predictor is None:
        predictor = get_predictor(selection["model"], selection.get("params"), selection.get("feature_columns"))
    predictions = predictor(add_time_features(history), future)
    forecast = validate_predictions(predictions, SUBMISSION_WINDOW.timestamps())
    return forecast, history


def save_forecast(forecast, history, selection, output):
    """Save the simple submission CSV and reusable configuration/provenance JSON."""
    output = Path(output)
    forecast = validate_predictions(forecast, SUBMISSION_WINDOW.timestamps())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep submission schema simple; full Prophet intervals are not submission columns.
    forecast[["ds", "yhat"]].to_csv(output, index=False)
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "prophet", "cmdstanpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "selection": selection,
        "origin_utc": SUBMISSION_WINDOW.start.isoformat(),
        "end_exclusive_utc": SUBMISSION_WINDOW.end.isoformat(),
        "observation_policy": OBSERVATION_POLICY,
        "target_unit": TARGET_UNIT,
        "train_start_utc": history["ds"].min().isoformat(),
        "train_last_hour_utc": history["ds"].max().isoformat(),
        "train_hours": len(history),
        "train_observed_targets": int(history["y"].notna().sum()),
        "prediction_hours": len(forecast),
        "package_versions": versions,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="Configuration selected using validation, never audit labels.")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "hourly.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "predictions/final_predictions.csv")
    parser.add_argument("--include-weather", action="store_true", help="Retain historical weather from hourly.csv for model inputs.")
    args = parser.parse_args()
    selection = load_selection(args.selection)
    hourly = load_hourly_data(args.data, include_weather=args.include_weather)
    forecast, history = generate_forecast(hourly, selection, include_weather=args.include_weather)
    save_forecast(forecast, history, selection, args.output)
    print(f"Saved {len(forecast)} finite hourly predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
