"""Calendar, recursive-temperature, and direct-context gradient boosting.

Calendar-only is the default. Lag features select recursive forecasting;
forecast-hour and origin-context features select a pooled direct forecast.
Notebooks and workflow adapters can use the same fitting/prediction function.
"""

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data_pipeline import WEATHER_COLUMNS
from .features import CALENDAR_FEATURES, add_time_features
from .gradient_boosting_features import (
    CONTEXT_FEATURES, LAG_FEATURES, MIN_ROLLING_OBSERVATIONS, SUPPORTED_FEATURES,
    TEMPERATURE_CONTEXT, add_temperature_lags, make_direct_future_data,
    make_direct_training_data, require_hourly_grid, required_weather_columns,
)

# Selected by time-window validation in notebooks/gradient_boosting_analysis.ipynb.
# The bounded follow-up favored min_samples_leaf=100 by <0.001 F. Grids can override.
DEFAULT_GRADIENT_BOOSTING_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 7,
    "max_depth": None,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 520,
}


def resolve_config(params=None, feature_columns=None):
    """Validate estimator settings and infer a strategy from explicit features."""
    if params is None:
        params = {}
    controls = {"forecast_strategy", "origin_stride_hours", "horizon_hours"}
    if not isinstance(params, dict) or set(params).difference(
        set(DEFAULT_GRADIENT_BOOSTING_PARAMS) | controls
    ):
        raise ValueError("Unsupported gradient boosting parameter configuration.")
    config = {**DEFAULT_GRADIENT_BOOSTING_PARAMS, **params}
    if config["loss"] not in ("squared_error", "absolute_error"):
        raise ValueError("Boosting loss must be squared_error or absolute_error.")
    for key in ("learning_rate", "l2_regularization"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number.")
    if config["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive.")
    for key in ("max_iter", "min_samples_leaf"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key, minimum in (("max_leaf_nodes", 2), ("max_depth", 1)):
        value = config[key]
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{key} must be null or an integer >= {minimum}.")
    if config["early_stopping"] is not False:
        raise ValueError("Use early_stopping=False; tune iterations using the shared time windows.")
    seed = config["random_state"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_state must be an integer from 0 through 2**32 - 1.")
    columns = list(CALENDAR_FEATURES) if feature_columns is None else feature_columns
    if not isinstance(columns, list) or not columns or any(
        not isinstance(column, str) or column not in SUPPORTED_FEATURES for column in columns
    ) or len(set(columns)) != len(columns):
        raise ValueError(f"Boosting feature columns must be unique names from {SUPPORTED_FEATURES}.")
    has_lags = bool(set(columns).intersection(LAG_FEATURES))
    has_context = bool(set(columns).intersection(["forecast_hour", *CONTEXT_FEATURES]))
    if has_lags and has_context:
        raise ValueError("Recursive lags and direct origin context require separate feature sets.")
    strategy = "direct" if has_context else "recursive" if has_lags else "calendar"
    if "forecast_strategy" in params and params["forecast_strategy"] != strategy:
        raise ValueError("forecast_strategy does not match the selected feature columns.")
    if strategy != "calendar":
        config["forecast_strategy"] = strategy
    if strategy == "recursive" and not {"lag_1", "lag_24"}.issubset(columns):
        raise ValueError("Recursive feature sets must include lag_1 and lag_24.")
    if strategy == "direct":
        if "forecast_hour" not in columns or not set(columns).intersection(TEMPERATURE_CONTEXT):
            raise ValueError("Direct feature sets need forecast_hour and temperature context.")
        config.setdefault("origin_stride_hours", 168)
        config.setdefault("horizon_hours", 336)
        stride = config["origin_stride_hours"]
        if type(stride) is not int or stride < 24 or stride % 24:
            raise ValueError("origin_stride_hours must be a positive multiple of 24.")
        if config["horizon_hours"] != 336 or type(config["horizon_hours"]) is not int:
            raise ValueError("Direct horizon_hours must be 336 for this project.")
    elif set(params).intersection({"origin_stride_hours", "horizon_hours"}):
        raise ValueError("Origin sampling and horizon settings apply only to direct forecasts.")
    return config, list(columns)


def build_model(params=None):
    """Build an unfitted estimator from estimator parameters (no feature controls).

    HistGradientBoostingRegressor handles missing numeric inputs directly. Its
    internal random validation split stays disabled; tune on time-based folds.
    """
    config, _ = resolve_config(params)
    estimator_params = {key: config[key] for key in DEFAULT_GRADIENT_BOOSTING_PARAMS}
    return HistGradientBoostingRegressor(**estimator_params, categorical_features=None)


def predict_gradient_boosting(history, future, params=None, feature_columns=None):
    """Fit fresh on observed pre-origin targets; return UTC ds/yhat predictions.

    Recompute calendar features from timestamps. Preserve missing hours when
    constructing lags/context, then exclude missing training labels at fitting.
    Recursive forecasts feed predictions into later lags. Direct forecasts
    simulate historical origins and reuse one fixed pre-origin context across
    the live horizon. No future observed temperature or weather is consumed.
    """
    config, columns = resolve_config(params, feature_columns)
    if "y" in future.columns:
        raise ValueError("Future inputs must not contain actual temperatures.")
    if not {"ds", "y"}.issubset(history.columns) or "ds" not in future.columns:
        raise ValueError("Boosting requires history ds/y and future ds columns.")
    if set(WEATHER_COLUMNS).intersection(future.columns):
        raise ValueError("Future inputs must not contain observed weather.")
    train = add_time_features(history).sort_values("ds").reset_index(drop=True)
    target = add_time_features(future[["ds"]]).sort_values("ds").reset_index(drop=True)
    if train.empty or target.empty or train["ds"].max() >= target["ds"].min():
        raise ValueError("Boosting training observations must precede the forecast origin.")
    if train["ds"].duplicated().any() or target["ds"].duplicated().any():
        raise ValueError("Boosting timestamps must be unique.")
    require_hourly_grid(target)
    train["y"] = pd.to_numeric(train["y"], errors="raise")
    if np.isinf(train["y"].to_numpy(dtype=float)).any():
        raise ValueError("Infinite training targets are invalid.")
    for column in required_weather_columns(columns):
        if column not in train:
            raise ValueError(
                f"Weather context requires processed column {column}. "
                "Re-run the data pipeline with --include-weather."
            )
        train[column] = pd.to_numeric(train[column], errors="raise")
        if np.isinf(train[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite training weather is invalid: {column}.")
    strategy = config.get("forecast_strategy", "calendar")
    if strategy in ("recursive", "direct"):
        require_hourly_grid(train)
        if train["ds"].iloc[-1] + pd.Timedelta(hours=1) != target["ds"].iloc[0]:
            raise ValueError("Lag/context history must end immediately before the origin.")
        if strategy == "recursive" and len(train) < 24:
            raise ValueError("Recursive forecasts need at least 24 hourly history rows.")
    estimator_params = {key: config[key] for key in DEFAULT_GRADIENT_BOOSTING_PARAMS}
    model = build_model(estimator_params)
    training_origins = 0
    if strategy == "direct":
        if len(target) > config["horizon_hours"]:
            raise ValueError("Direct prediction exceeds the trained horizon.")
        inputs, labels, metadata = make_direct_training_data(
            train, columns, target["ds"].iloc[0].hour,
            config["horizon_hours"], config["origin_stride_hours"],
        )
        training_origins = metadata["origin_ds"].nunique()
        prediction_inputs = make_direct_future_data(train, target, columns)
    else:
        prepared = add_temperature_lags(train) if strategy == "recursive" else train
        observed = prepared.dropna(subset=["y"])
        inputs, labels = observed[columns], observed["y"]
        prediction_inputs = target[columns] if strategy == "calendar" else None
    if len(labels) < 2:
        raise ValueError("Boosting requires at least two observed training targets.")
    model.fit(inputs, labels)
    if strategy == "recursive":
        predicted = predict_recursive(model, train, target, columns)
    else:
        predicted = model.predict(prediction_inputs)
    if not np.isfinite(predicted).all():
        raise ValueError("Boosting must produce a finite prediction for every hour.")
    result = target[["ds"]].assign(yhat=predicted)
    result.attrs.update(forecast_strategy=strategy, fit_rows=len(labels),
                        training_origins=int(training_origins))
    return result


def predict_recursive(model, history, future, columns):
    """Advance hour by hour using prior predictions after the fixed origin."""
    require_hourly_grid(history)
    require_hourly_grid(future)
    if len(history) < 24 or history["ds"].iloc[-1] + pd.Timedelta(hours=1) != future["ds"].iloc[0]:
        raise ValueError("Recursive forecasts need 24 hourly rows immediately before the origin.")
    values = history["y"].to_numpy(dtype=float).tolist()
    predicted = []
    for position in range(len(future)):
        row = future.iloc[[position]].copy()
        row["lag_1"], row["lag_24"] = values[-1], values[-24]
        recent = np.asarray(values[-24:])
        row["rolling_mean_24"] = (
            np.nanmean(recent) if np.isfinite(recent).sum() >= MIN_ROLLING_OBSERVATIONS else np.nan
        )
        value = float(model.predict(row[columns])[0])
        if not np.isfinite(value):
            raise ValueError("Recursive boosting produced a nonfinite prediction.")
        predicted.append(value)
        values.append(value)
    return np.asarray(predicted)
