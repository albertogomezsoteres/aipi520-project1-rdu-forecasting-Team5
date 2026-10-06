"""Small fixed-origin backtesting runner for the shared project forecast task."""

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .baselines import BASELINE_NAMES, predict_baseline
from .config import AUDIT_WINDOW, DATA_DIR, OUTPUT_DIR, VALIDATION_WINDOWS, ForecastWindow
from .data_pipeline import load_hourly_data, validate_hourly_data
from .evaluation import evaluate_forecast, make_mase_scale
from .features import add_time_features
from .models import MODEL_NAMES, get_predictor, model_config

Predictor = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def select_history(hourly: pd.DataFrame, origin: pd.Timestamp, lookback_years=None):
    """Select a complete pre-origin timeline, optionally a rolling N-year span."""
    if lookback_years is not None and (
        not isinstance(lookback_years, int) or isinstance(lookback_years, bool) or lookback_years < 1
    ):
        raise ValueError("lookback_years must be a positive integer or None.")
    origin = pd.to_datetime(origin, utc=True)
    history = hourly.loc[hourly["ds"] < origin].copy()
    if lookback_years is not None:
        requested_start = origin - pd.DateOffset(years=lookback_years)
        if history.empty or history["ds"].min() > requested_start:
            raise ValueError("Not enough data for the requested training-history length.")
        history = history.loc[history["ds"] >= requested_start].copy()
    if history.empty or history["ds"].max() != origin - pd.Timedelta(hours=1):
        raise ValueError("History must extend to the hour immediately before the origin.")
    return history.reset_index(drop=True)


def make_fold(hourly: pd.DataFrame, window: ForecastWindow, lookback_years=None):
    """Return training history, target-free future features, and separate labels."""
    history = select_history(hourly, window.start, lookback_years)
    actual = hourly.loc[
        (hourly["ds"] >= window.start) & (hourly["ds"] < window.end), ["ds", "y"]
    ].reset_index(drop=True)
    if not pd.DatetimeIndex(actual["ds"]).equals(window.timestamps()):
        raise ValueError(f"Data does not cover the complete {window.name} target grid.")
    future = add_time_features(pd.DataFrame({"ds": window.timestamps()}))
    return add_time_features(history), future, actual


def run_backtests(hourly, predictors: dict[str, Predictor], lookbacks=(None,), windows=VALIDATION_WINDOWS,
                  *, include_weather=False):
    """Refit each candidate at each origin; return metrics and hourly predictions.

    MASE uses all available pre-origin history for every candidate in a fold.
    Labels are held separately and never passed into the predictor's future frame.
    Optional weather is passed only in pre-origin history, never future inputs.
    """
    data = validate_hourly_data(hourly, include_weather=include_weather)
    rows, comparisons = [], []
    for window in windows:
        reference_history = select_history(data, window.start)
        scale = make_mase_scale(reference_history, lag=24)
        for lookback in lookbacks:
            history, future, actual = make_fold(data, window, lookback)
            for name, predict in predictors.items():
                print(f"{window.name}: {name}, history={lookback or 'all'} years", flush=True)
                predictions = predict(history.copy(), future.copy())
                metrics, comparison = evaluate_forecast(actual, predictions, scale)
                metadata = {
                    "model": name,
                    "lookback": "all" if lookback is None else str(lookback),
                    "window": window.name,
                    "origin_utc": window.start.isoformat(),
                    "train_start_utc": history["ds"].min().isoformat(),
                    "train_hours": len(history),
                    "train_observed": int(history["y"].notna().sum()),
                    "mase_scale": scale,
                    "last_day_fallback_hours": predictions.attrs.get("last_day_fallback_hours", 0),
                }
                rows.append({**metadata, **metrics})
                comparison["forecast_hour"] = range(1, len(comparison) + 1)
                comparison["forecast_day"] = (comparison["forecast_hour"] - 1) // 24 + 1
                for key in ("model", "lookback", "window", "origin_utc"):
                    comparison[key] = metadata[key]
                comparisons.append(comparison)
    if not rows:
        raise ValueError("Backtesting requires windows, lookbacks, and predictors.")
    return pd.DataFrame(rows), pd.concat(comparisons, ignore_index=True)


def summarize_results(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank by equally weighted window MAE; expose variability and coverage."""
    if metrics.duplicated(["model", "lookback", "window"]).any():
        raise ValueError("Duplicate model/history/window results.")
    expected_windows = set(metrics["window"])
    if any(set(group["window"]) != expected_windows for _, group in metrics.groupby(["model", "lookback"])):
        raise ValueError("All candidates must be evaluated on the same windows.")
    summary = metrics.groupby(["model", "lookback"], as_index=False).agg(
        mean_mae=("mae", "mean"), worst_mae=("mae", "max"),
        std_mae=("mae", "std"), mean_rmse=("rmse", "mean"),
        mean_mase=("mase", "mean"), mean_bias=("bias", "mean"),
        windows=("window", "nunique"), scored_hours=("hours_evaluated", "sum"),
    )
    return summary.sort_values(["mean_mae", "worst_mae", "model", "lookback"]).reset_index(drop=True)


def candidate_config(model: str, lookback_years=None) -> dict:
    """Serializable configuration reused for audit and submission forecasting."""
    return {**model_config(model), "lookback_years": lookback_years}


def load_selection(path: Path) -> dict:
    selection = json.loads(Path(path).read_text())
    if selection.get("model") not in MODEL_NAMES:
        raise ValueError("Selection contains an unsupported model.")
    lookback = selection.get("lookback_years")
    if lookback is not None and (type(lookback) is not int or lookback < 1):
        raise ValueError("Selected lookback_years must be positive or null.")
    if not isinstance(selection.get("params", {}), dict):
        raise ValueError("Selected model parameters must be an object.")
    resolved = model_config(selection["model"], selection.get("params"), selection.get("feature_columns"))
    if "features" in selection and selection["features"] != resolved["features"]:
        raise ValueError("Saved feature metadata does not match the selected configuration.")
    return {**selection, **resolved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "hourly.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "backtests")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--lookbacks", nargs="+", default=["all", "5", "3"])
    parser.add_argument("--audit", action="store_true", help="Evaluate a locked selection; do not select from audit results.")
    parser.add_argument("--selection", type=Path, help="Required with --audit.")
    parser.add_argument("--include-weather", action="store_true", help="Retain historical weather from hourly.csv for model inputs.")
    args = parser.parse_args()
    if args.audit:
        if args.selection is None:
            parser.error("--audit requires --selection selected_config.json")
        selection = load_selection(args.selection)
        model = selection["model"]
        predictors = {name: get_predictor(name) for name in BASELINE_NAMES}
        predictors[model] = get_predictor(model, selection.get("params"), selection.get("feature_columns"))
        lookbacks = [selection.get("lookback_years")]
        windows, prefix = (AUDIT_WINDOW,), "audit"
    else:
        if args.selection is not None:
            parser.error("--selection is only used with --audit")
        try:
            lookbacks = [None if value == "all" else int(value) for value in args.lookbacks]
        except ValueError:
            parser.error("--lookbacks accepts all or positive integer years")
        if any(value is not None and value < 1 for value in lookbacks):
            parser.error("--lookbacks accepts all or positive integer years")
        predictors = {name: get_predictor(name) for name in dict.fromkeys(args.models)}
        windows, prefix = VALIDATION_WINDOWS, "validation"
    lookbacks = list(dict.fromkeys(lookbacks))
    hourly = load_hourly_data(args.data, include_weather=args.include_weather)
    metrics, predictions = run_backtests(hourly, predictors, lookbacks, windows,
                                        include_weather=args.include_weather)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / f"{prefix}_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / f"{prefix}_predictions.csv", index=False)
    summary = summarize_results(metrics)
    summary.to_csv(args.output_dir / f"{prefix}_summary.csv", index=False)
    if not args.audit:
        best = summary.iloc[0]
        selected = candidate_config(best["model"], None if best["lookback"] == "all" else int(best["lookback"]))
        selected.update({
            "selection_metric": "equal_window_mean_mae",
            "validation_windows": [window.name for window in windows],
            "mean_validation_mae": float(best["mean_mae"]),
        })
        (args.output_dir / "selected_config.json").write_text(json.dumps(selected, indent=2) + "\n")
        configs = [candidate_config(name, lookback) for name in predictors for lookback in lookbacks]
        (args.output_dir / "candidate_configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
