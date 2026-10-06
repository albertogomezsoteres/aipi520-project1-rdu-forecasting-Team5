"""Small model registry shared by tuning, historical audits, and final refits.

Adapters fit a fresh model using history and predict a target-free future frame.
Keep model-specific preprocessing inside the adapter and fit it on history only.
"""

from .baselines import BASELINE_NAMES, predict_baseline

MODEL_NAMES = (*BASELINE_NAMES, "prophet", "linear_regression", "gradient_boosting")


def model_config(name, params=None, feature_columns=None):
    """Resolve defaults and record inputs/components actually used by the model.

    feature_columns selects gradient boosting inputs; null uses its default
    calendar features. Other models use fixed inputs instead.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {name}. Available models: {MODEL_NAMES}")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("Model parameters must be an object.")
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or not feature_columns or any(
            not isinstance(column, str) or not column for column in feature_columns
        ) or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("Feature columns must be a nonempty list of unique names.")
        if {"y", "yhat", "ds"}.intersection(feature_columns):
            raise ValueError("Feature columns cannot include labels, predictions, or ds.")
        if name in (*BASELINE_NAMES, "prophet", "linear_regression"):
            raise ValueError(f"{name} uses fixed inputs; its feature_sets must contain only null.")

    if name in BASELINE_NAMES:
        if params:
            raise ValueError("Baselines do not accept model parameters.")
        components = ["training UTC month/hour temperature means", "training global mean fallback"]
        if name == "repeat_last_day":
            components.insert(0, "previous 24 elapsed-hour temperatures, repeated without future updates")
        features = {"input_columns": ["ds"], "components": components}
    elif name == "prophet":
        from .prophet_model import BEST_PROPHET_PARAMS
        if set(params).difference(BEST_PROPHET_PARAMS):
            raise ValueError("Unsupported Prophet configuration keys.")
        params = {**BEST_PROPHET_PARAMS, **params}
        for key in ("changepoint_prior_scale", "seasonality_prior_scale"):
            value = params[key]
            if type(value) not in (int, float) or not 0 < value < float("inf"):
                raise ValueError(f"{key} must be a finite positive number.")
        for key in ("daily_fourier_order", "yearly_fourier_order"):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if type(params["weekly_seasonality"]) is not bool:
            raise ValueError("weekly_seasonality must be true or false.")
        components = [
            "piecewise linear trend with Prophet automatic changepoints",
            f"daily Fourier seasonality, order {params['daily_fourier_order']}",
            f"yearly Fourier seasonality, order {params['yearly_fourier_order']}",
        ]
        if params["weekly_seasonality"]:
            components.append("weekly Fourier seasonality, Prophet default order 3")
        features = {"input_columns": ["ds"], "components": components}
    elif name == "linear_regression":
        # LR builds its own features from ds and past y, so feature_columns stays null
        from .linear_regression_model import describe_linear_regression, resolve_linear_regression_params
        params = resolve_linear_regression_params(params)
        features = {"input_columns": ["ds", "y (history only)"], "components": describe_linear_regression(params)}
    elif name == "gradient_boosting":
        from .features import CALENDAR_FEATURES
        from .gradient_boosting_features import required_weather_columns
        from .gradient_boosting_model import resolve_config
        params, feature_columns = resolve_config(params, feature_columns)
        strategy = params.get("forecast_strategy", "calendar")
        components = ["numeric inputs; missing values handled by the estimator"]
        if set(feature_columns).intersection(CALENDAR_FEATURES):
            components.append("shared UTC calendar/cyclic definitions")
        if strategy == "recursive":
            components.extend([
                "elapsed-hour temperature lags built before dropping missing targets",
                "post-origin temperature inputs updated using predictions",
            ])
        elif strategy == "direct":
            components.extend([
                "one fixed pre-origin context across the forecast",
                f"pooled historical origins every {params['origin_stride_hours']} hours; 336-hour horizon",
            ])
        features = {
            "input_columns": feature_columns,
            "historical_input_columns": ["ds", "y", *required_weather_columns(feature_columns)],
            "forecast_strategy": strategy,
            "components": components,
        }
    else:
        raise ValueError(f"No configuration resolver registered for {name}.")
    return {"model": name, "params": dict(params), "feature_columns": feature_columns,
            "features": features}


def get_predictor(name, params=None, feature_columns=None):
    """Return a fresh-fit adapter with resolved settings, never fitted state."""
    config = model_config(name, params, feature_columns)
    if name in BASELINE_NAMES:
        return lambda history, future: predict_baseline(history, future, name)
    if name == "prophet":
        from .prophet_model import predict_prophet
        return lambda history, future: predict_prophet(history, future, config["params"])
    if name == "linear_regression":
        from .linear_regression_model import predict_linear_regression
        return lambda history, future: predict_linear_regression(
            history, future, config["params"], config["feature_columns"])
    if name == "gradient_boosting":
        from .gradient_boosting_model import predict_gradient_boosting
        return lambda history, future: predict_gradient_boosting(
            history, future, config["params"], config["feature_columns"])
    raise ValueError(f"No predictor registered for {name}.")
