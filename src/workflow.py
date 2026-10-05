"""Prepare data, tune on historical windows, audit, and refit a final forecast.

Run from the repository root: python -m src.workflow
Each run saves its inputs, candidates, metrics, selection, and predictions to a
new folder. Audit results never change the validation-selected configuration.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

import pandas as pd

from .backtesting import make_fold, run_backtests, summarize_results
from .baselines import BASELINE_NAMES
from .config import AUDIT_WINDOW, OUTPUT_DIR, PROJECT_ROOT, RAW_PATH, VALIDATION_WINDOWS
from .data_pipeline import prepare_data
from .forecast import generate_forecast, save_forecast
from .models import get_predictor, model_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiment.json"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment(path):
    """Read the small JSON search specification; validate before fitting."""
    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite value in experiment JSON: {value}")
    experiment = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    expand_candidates(experiment)
    return experiment


def expand_candidates(experiment):
    """Build deterministic parameter/feature candidates and shared lookbacks.

    Each model's param_grid is a Cartesian product. Omitted parameters resolve
    to that adapter's defaults. Null feature sets mean the model's fixed inputs.
    """
    if not isinstance(experiment, dict) or set(experiment) != {"models", "lookbacks"}:
        raise ValueError("Experiment must contain exactly models and lookbacks.")
    lookbacks = experiment["lookbacks"]
    if not isinstance(lookbacks, list) or not lookbacks or any(
        value is not None and (type(value) is not int or value < 1) for value in lookbacks
    ) or len(set(lookbacks)) != len(lookbacks):
        raise ValueError("lookbacks must be unique positive integer years or null (all history).")
    models = experiment["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list.")
    candidates, names, seen = {}, set(), set()
    for spec in models:
        if not isinstance(spec, dict) or "name" not in spec or set(spec).difference(
            {"name", "param_grid", "feature_sets"}
        ):
            raise ValueError("Each model needs name, with optional param_grid and feature_sets.")
        name = spec["name"]
        if not isinstance(name, str) or name in names:
            raise ValueError("Each model name must occur once.")
        names.add(name)
        grid, feature_sets = spec.get("param_grid", {}), spec.get("feature_sets", [None])
        if not isinstance(grid, dict) or any(
            not isinstance(key, str) or not isinstance(values, list) or not values
            for key, values in grid.items()
        ):
            raise ValueError("param_grid must map parameter names to nonempty value lists.")
        if not isinstance(feature_sets, list) or not feature_sets:
            raise ValueError("feature_sets must be a nonempty list.")
        keys = sorted(grid)
        for values in product(*(grid[key] for key in keys)):
            for features in feature_sets:
                config = model_config(name, dict(zip(keys, values)), features)
                fingerprint = json.dumps(config, sort_keys=True, allow_nan=False)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                candidate_id = f"{name}_{sum(item['model'] == name for item in candidates.values()) + 1:03d}"
                candidates[candidate_id] = config
    # TODO(linear-regression): Add its name, parameter grid, and feature_sets to
    # configs/experiment.json after registering the adapter in src/models.py.
    # TODO(gradient-boosting): Add its search specification in the same way.
    # Keep grids small initially: fits = candidates * lookbacks * windows.
    return candidates, lookbacks


def attach_model_names(frame, candidates):
    """Keep candidate IDs distinct while making saved tables readable."""
    result = frame.rename(columns={"model": "candidate_id"}).copy()
    result.insert(0, "model", result["candidate_id"].map(lambda key: candidates[key]["model"]))
    return result


def tune_models(hourly, experiment, windows=VALIDATION_WINDOWS):
    """Compare all candidates on identical windows; choose by equal-window MAE.

    No audit or submission targets participate in selection. Each backtest fits
    fresh model state. A failed candidate stops the run rather than selecting
    from an incomplete comparison.
    """
    candidates, lookbacks = expand_candidates(experiment)
    predictors = {
        key: get_predictor(config["model"], config["params"], config["feature_columns"])
        for key, config in candidates.items()
    }
    metrics, predictions = run_backtests(hourly, predictors, lookbacks, windows)
    summary = summarize_results(metrics)
    best = summary.iloc[0]
    selected = {
        **candidates[best["model"]],
        "candidate_id": best["model"],
        "lookback_years": None if best["lookback"] == "all" else int(best["lookback"]),
        "selection_metric": "equal_window_mean_mae",
        "tie_breakers": ["worst_window_mae", "candidate_id", "lookback"],
        "validation_windows": [window.name for window in windows],
        "mean_validation_mae": float(best["mean_mae"]),
    }
    return (attach_model_names(metrics, candidates), attach_model_names(predictions, candidates),
            attach_model_names(summary, candidates), selected)


def run_workflow(config_path=DEFAULT_CONFIG, input_path=RAW_PATH, output_dir=None):
    """Run the full workflow and return the saved run-summary dictionary.

    Existing output directories are refused to avoid mixing runs. Input raw
    data is never modified; processed data is saved inside this run's folder.
    """
    experiment = load_experiment(config_path)
    candidates, lookbacks = expand_candidates(experiment)
    if any(config["model"] == "prophet" for config in candidates.values()):
        try:
            import prophet  # noqa: F401; check before spending time on other candidates
        except ImportError as error:
            raise ImportError("Prophet is configured. Install requirements.txt or use the baselines config.") from error
    input_path = Path(input_path).resolve()
    input_hash = file_sha256(input_path)
    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = OUTPUT_DIR / "runs" / run_id
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "experiment.json", experiment)
    expanded = [dict(config, candidate_id=key, lookback_years=lookback)
                for key, config in candidates.items() for lookback in lookbacks]
    write_json(output_dir / "candidate_configs.json", expanded)

    print(f"Run folder: {output_dir}", flush=True)
    print("Preparing hourly data...", flush=True)
    hourly, _, cleaning = prepare_data(input_path, output_dir / "data_processed")
    # Check historical coverage before starting expensive model fits.
    for window in (*VALIDATION_WINDOWS, AUDIT_WINDOW):
        for lookback in lookbacks:
            make_fold(hourly, window, lookback)
    fits = len(candidates) * len(lookbacks) * len(VALIDATION_WINDOWS)
    print(f"Tuning {len(expanded)} candidate/history combinations: {fits} validation fits.", flush=True)
    metrics, predictions, summary, selected = tune_models(hourly, experiment)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    summary.to_csv(output_dir / "validation_summary.csv", index=False)
    write_json(output_dir / "selected_config.json", selected)

    print("Auditing the locked selection and baselines...", flush=True)
    audit_predictors = {name: get_predictor(name) for name in BASELINE_NAMES}
    audit_predictors[selected["model"]] = get_predictor(
        selected["model"], selected["params"], selected["feature_columns"]
    )
    audit_metrics, audit_predictions = run_backtests(
        hourly, audit_predictors, [selected["lookback_years"]], [AUDIT_WINDOW]
    )
    audit_metrics.to_csv(output_dir / "audit_metrics.csv", index=False)
    audit_predictions.to_csv(output_dir / "audit_predictions.csv", index=False)
    summarize_results(audit_metrics).to_csv(output_dir / "audit_summary.csv", index=False)

    print("Refitting the selected configuration and forecasting...", flush=True)
    forecast, history = generate_forecast(hourly, selected)
    metadata = save_forecast(forecast, history, selected, output_dir / "final_predictions.csv")
    audit_row = audit_metrics.loc[audit_metrics["model"] == selected["model"]].iloc[0]
    run_summary = {
        "status": "complete", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input": str(input_path), "raw_input_sha256": input_hash,
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                          for path in sorted((PROJECT_ROOT / "src").glob("*.py"))},
        "experiment": experiment, "candidate_history_combinations": len(expanded),
        "validation_fits": fits, "cleaning": cleaning,
        "selection": selected, "selected_audit_mae": float(audit_row["mae"]),
        "audit_used_for_selection": False, "forecast": metadata,
        "outputs": {path.name: str(path) for path in sorted(output_dir.glob("*")) if path.is_file()},
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print(f"Selected model: {selected['model']}")
    print(f"Parameters: {json.dumps(selected['params'], sort_keys=True)}")
    print(f"Features: {json.dumps(selected['features'])}")
    print(f"Training history: {selected['lookback_years'] or 'all'} years")
    print(f"Mean validation MAE: {selected['mean_validation_mae']:.4f}")
    print(f"Audit MAE: {run_summary['selected_audit_mae']:.4f}")
    print(f"Forecast: {output_dir / 'final_predictions.csv'}")
    print(f"Run details: {output_dir / 'run_summary.json'}")
    return run_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, help="New run folder; default is a UTC-timestamped outputs/runs folder.")
    args = parser.parse_args()
    run_workflow(args.config, args.input, args.output_dir)


if __name__ == "__main__":
    main()
