"""Temperature lags and forecast-origin context for gradient boosting.

All context excludes the origin hour. Direct forecasts repeat that context
across horizons; they never read weather or temperatures from the future.
"""

import numpy as np
import pandas as pd

from .features import CALENDAR_FEATURES, add_time_features

LAG_FEATURES = ["lag_1", "lag_24", "rolling_mean_24"]
TEMPERATURE_CONTEXT = ["temp_last", "temp_mean_24", "temp_change_24"]
WEATHER_GROUPS = {
    "dew_point": ["dewpoint_last", "dewpoint_mean_24"],
    "wind_speed": ["wind_speed_mean_24"],
    "humidity": ["humidity_mean_24"],
    "pressure": ["pressure_last", "pressure_change_24"],
    "cloud_cover": ["cloud_cover_mean_24"],
    "wind_direction": ["wind_u_mean_24", "wind_v_mean_24"],
    "precipitation": ["precip_sum_24", "precip_trace_hours_24"],
}
WEATHER_REQUIREMENTS = {
    "dew_point": ["dewpoint"], "wind_speed": ["wind_speed"],
    "humidity": ["humidity"], "pressure": ["pressure"],
    "cloud_cover": ["cloud_cover"],
    "wind_direction": ["wind_speed", "wind_direction"],
    "precipitation": ["precip", "precip_trace"],
}
CONTEXT_FEATURES = TEMPERATURE_CONTEXT + [
    column for group in WEATHER_GROUPS.values() for column in group
]
SUPPORTED_FEATURES = CALENDAR_FEATURES + LAG_FEATURES + ["forecast_hour"] + CONTEXT_FEATURES
MIN_ROLLING_OBSERVATIONS = 18  # At least 75% of the preceding 24 hours.


def required_weather_columns(columns):
    """Return the observed weather inputs needed by an explicit feature set."""
    return sorted({source for group, features in WEATHER_GROUPS.items()
                   if set(columns or []).intersection(features)
                   for source in WEATHER_REQUIREMENTS[group]})


def require_hourly_grid(frame):
    """Validate positional lag alignment without dropping missing target hours."""
    times = frame["ds"]
    if times.empty or not times.eq(times.dt.floor("h")).all() or not times.diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Temperature/context features require a complete ordered hourly grid.")


def add_temperature_lags(history):
    """Use elapsed-hour shifts and a rolling mean that excludes the target."""
    require_hourly_grid(history)
    result = add_time_features(history[["ds", "y"]])
    result["lag_1"] = history["y"].shift(1)
    result["lag_24"] = history["y"].shift(24)
    result["rolling_mean_24"] = history["y"].shift(1).rolling(
        24, min_periods=MIN_ROLLING_OBSERVATIONS
    ).mean()
    return result


def origin_context(history, columns):
    """Build context for each row, using observations strictly before its ds.

    No forward/backward fill or interpolation. Latest means origin minus one
    hour, even when that value is missing. Changes compare origin-1 to origin-25.
    Means need 18 valid hours; precipitation totals need all 24 hours.
    """
    require_hourly_grid(history)
    needed = {"y"} if set(columns).intersection(TEMPERATURE_CONTEXT) else set()
    needed.update(required_weather_columns(columns))
    missing = needed.difference(history.columns)
    if missing:
        raise ValueError(
            f"Weather context requires processed columns {sorted(missing)}. "
            "Re-run the data pipeline with --include-weather."
        )
    context = pd.DataFrame(index=history.index)

    def mean24(values):
        return values.shift(1).rolling(24, min_periods=MIN_ROLLING_OBSERVATIONS).mean()

    if "y" in needed:
        context["temp_last"] = history["y"].shift(1)
        context["temp_mean_24"] = mean24(history["y"])
        context["temp_change_24"] = history["y"].shift(1) - history["y"].shift(25)
    if "dewpoint" in needed:
        context["dewpoint_last"] = history["dewpoint"].shift(1)
        context["dewpoint_mean_24"] = mean24(history["dewpoint"])
    for source, feature in (("wind_speed", "wind_speed_mean_24"),
                            ("humidity", "humidity_mean_24"), ("cloud_cover", "cloud_cover_mean_24")):
        if source in needed:
            context[feature] = mean24(history[source])
    if "pressure" in needed:
        context["pressure_last"] = history["pressure"].shift(1)
        context["pressure_change_24"] = history["pressure"].shift(1) - history["pressure"].shift(25)
    if "wind_direction" in needed:
        radians = np.deg2rad(history["wind_direction"])
        # Meteorological direction is where the wind comes FROM.
        u = (-history["wind_speed"] * np.sin(radians)).mask(history["wind_speed"].eq(0), 0.)
        v = (-history["wind_speed"] * np.cos(radians)).mask(history["wind_speed"].eq(0), 0.)
        context["wind_u_mean_24"], context["wind_v_mean_24"] = mean24(u), mean24(v)
    if "precip" in needed:
        context["precip_sum_24"] = history["precip"].shift(1).rolling(24, min_periods=24).sum()
        context["precip_trace_hours_24"] = history["precip_trace"].shift(1).rolling(24, min_periods=24).sum()
    return context[list(columns)]


def make_direct_training_data(history, columns, origin_hour, horizon_hours=336, origin_stride_hours=168):
    """Return pooled (origin, horizon) examples and target/origin timestamp metadata.

    Sample origins at the forecast's UTC hour, spaced weekly by default. Every
    sampled origin has 25 prior hours and its entire target horizon in history.
    Training labels therefore all precede the real validation forecast origin.
    Overlapping training horizons are intentional; validation remains chronological.
    """
    require_hourly_grid(history)
    if type(origin_hour) is not int or not 0 <= origin_hour < 24:
        raise ValueError("origin_hour must be a UTC hour from 0 through 23.")
    if type(horizon_hours) is not int or horizon_hours < 1:
        raise ValueError("horizon_hours must be a positive integer.")
    if type(origin_stride_hours) is not int or origin_stride_hours < 24 or origin_stride_hours % 24:
        raise ValueError("origin_stride_hours must be a positive multiple of 24.")
    context_columns = [column for column in columns if column in CONTEXT_FEATURES]
    context = origin_context(history, context_columns)
    positions = np.arange(len(history))
    eligible = positions[
        (positions >= 25) & (positions + horizon_hours <= len(history))
        & history["ds"].dt.hour.eq(origin_hour).to_numpy()
    ]
    origins = eligible[::origin_stride_hours // 24]
    if not len(origins):
        raise ValueError("Direct boosting needs 25 prior hours plus a complete training forecast horizon.")
    origin_positions = np.repeat(origins, horizon_hours)
    offsets = np.tile(np.arange(horizon_hours), len(origins))
    targets = origin_positions + offsets
    calendar = add_time_features(history[["ds"]]).iloc[targets].reset_index(drop=True)
    inputs = calendar.copy()
    inputs["forecast_hour"] = offsets + 1
    for column in context_columns:
        inputs[column] = context[column].to_numpy()[origin_positions]
    labels = history["y"].to_numpy()[targets]
    observed = np.isfinite(labels)
    metadata = pd.DataFrame({"origin_ds": history["ds"].iloc[origin_positions].to_numpy(),
                             "target_ds": history["ds"].iloc[targets].to_numpy()})
    return (inputs.loc[observed, columns].reset_index(drop=True), labels[observed],
            metadata.loc[observed].reset_index(drop=True))


def make_direct_future_data(history, future, columns):
    """Use the same origin-context definitions for historical and live forecasts."""
    context_columns = [column for column in columns if column in CONTEXT_FEATURES]
    # Append a timestamp-only row: shifting at the origin sees only history.
    extended = pd.concat([history, pd.DataFrame({"ds": [future["ds"].iloc[0]]})], ignore_index=True)
    context = origin_context(extended, context_columns).iloc[-1]
    inputs = add_time_features(future[["ds"]])
    inputs["forecast_hour"] = np.arange(1, len(inputs) + 1)
    for column in context_columns:
        inputs[column] = context[column]
    return inputs[columns]
