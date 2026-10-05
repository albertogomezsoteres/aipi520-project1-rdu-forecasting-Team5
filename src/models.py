"""Small model registry shared by tuning, historical audits, and final refits.

Adapters fit a fresh model using history and predict a target-free future frame.
Keep model-specific preprocessing inside the adapter and fit it on history only.
"""

from .baselines import BASELINE_NAMES, predict_baseline

MODEL_NAMES = (*BASELINE_NAMES, "prophet")
# TODO(linear-regression): Add "linear_regression" after its adapter is ready.
# TODO(gradient-boosting): Add "gradient_boosting" after its adapter is ready.


def model_config(name, params=None, feature_columns=None):
    """Resolve defaults and record inputs/components actually used by the model.

    feature_columns is an optional explicit predictor-column list for future
    regression/boosting adapters. Current models use fixed inputs instead.
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
        if name in (*BASELINE_NAMES, "prophet"):
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
    else:
        # TODO(linear-regression): Resolve estimator defaults, validate feature_columns,
        # and describe the actual columns/transforms used by predict_linear_regression.
        # TODO(gradient-boosting): Do the same for predict_gradient_boosting. Describe
        # any lag/rolling features and recursive updates in the saved components.
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
    # TODO(linear-regression): Import its adapter here and pass config["params"]
    # and config["feature_columns"]. Fit scaling/imputation on history only.
    # TODO(gradient-boosting): Import its adapter here and pass the same settings.
    # Build lags on the complete hourly grid before dropping missing training y;
    # after the origin, recursive features must use predictions, never actual y.
    raise ValueError(f"No predictor registered for {name}.")
