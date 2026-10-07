"""
Linear regression model for the RDU hourly temperature forecast (Team 5).

The idea is simple: temperature = climatology + what is left of the current anomaly.

- Stage 1 (climatology): linear regression on calendar features (daily and
  annual harmonics, their interaction and a linear trend). We know these
  features for any future hour, so they work for the whole 14 days.
- Stage 2 (anomaly): we check how much warmer or colder than normal the last
  hours/days were, and the model learns how fast that anomaly fades.

We choose everything with time series cross-validation (36 windows in
September 2017-2025) and the final settings are in FINAL_CONFIG.
The analysis behind each choice is in notebooks/linear_regression_analysis.ipynb.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Works with python -m src.linear_regression_model and from the notebook
if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, FORECAST_HOURS, FORECAST_START, LOCAL_TIMEZONE, OUTPUT_DIR

warnings.filterwarnings("ignore")

PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"

# Same conventions as src/config.py
HORIZON = FORECAST_HOURS
LOCAL_TZ = LOCAL_TIMEZONE
FINAL_ORIGIN = FORECAST_START  # Sep 17 00:00 EDT = 04:00 UTC
TREND_START = pd.Timestamp("2015-01-01", tz="UTC")
DECAYS_H = (6, 24, 72, 168)
CV_YEARS = range(2017, 2026)
CV_WINDOW_STARTS = ("09-03", "09-10", "09-17", "09-24")


# ---------------------------------------------------------------------------
# Data and metrics (same as in prophet_model.py)
# ---------------------------------------------------------------------------
def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    return df.sort_values("ds").reset_index(drop=True)


def load_history() -> pd.DataFrame:
    """All the hours we can use before the real forecast starts."""
    # We use the canonical hourly.csv from the pipeline (it goes up to 03:00 UTC
    # on Sep 17), otherwise we just join train, val and test
    if (DATA_DIR / "hourly.csv").is_file():
        df = load_split("hourly")
    else:
        df = pd.concat([load_split(n) for n in ("train", "val", "test")], ignore_index=True)
    df = df.drop_duplicates("ds").sort_values("ds").reset_index(drop=True)
    return df.loc[df["ds"] < FINAL_ORIGIN, ["ds", "y"]]


def make_mase_scale(df: pd.DataFrame, lag: int = 24) -> float:
    data = df.sort_values("ds")
    assert data["ds"].diff().dropna().eq(pd.Timedelta(hours=1)).all(), \
        "MASE requires a complete hourly timeline."
    y = data.set_index("ds")["y"]
    scale = (y - y.shift(lag)).abs().mean()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MASE scale is zero or invalid.")
    return float(scale)


def calculate_metrics(y_true, y_pred, mase_scale: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # We only score the hours that have a real temperature
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    a, p = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(a, p)
    out = {
        "mae": float(mae),
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "r2": float(r2_score(a, p)),
        "bias": float((p - a).mean()),
        "hours_evaluated": int(mask.sum()),
    }
    if mase_scale is not None:
        out["mase"] = float(mae / mase_scale)
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _angles(ds):
    # We use UTC hours on purpose: UTC follows the sun better than local time,
    # which jumps 1 hour with daylight saving (CV MAE 5.229 UTC vs 5.243 local)
    ds = pd.DatetimeIndex(ds)
    hour = ds.hour.to_numpy()
    days_in_year = np.where(ds.is_leap_year, 366, 365)
    daily = 2 * np.pi * hour / 24
    annual = 2 * np.pi * (ds.dayofyear.to_numpy() - 1 + hour / 24) / days_in_year
    return ds, daily, annual


def shared_features(ds) -> pd.DataFrame:
    """The 4 cyclic features from data_pipeline.py."""
    _, daily, annual = _angles(ds)
    return pd.DataFrame({"hour_sin": np.sin(daily), "hour_cos": np.cos(daily),
                         "year_sin": np.sin(annual), "year_cos": np.cos(annual)})


def calendar_features(ds, k_daily: int, k_annual: int, k_interact: int, trend: bool) -> pd.DataFrame:
    """Calendar features for stage 1."""
    ds, daily, annual = _angles(ds)
    X = {}
    # First we add the daily harmonics (the daily cycle is not a perfect sine)
    for k in range(1, k_daily + 1):
        X[f"day_sin{k}"], X[f"day_cos{k}"] = np.sin(k * daily), np.cos(k * daily)
    # Then the annual harmonics for the seasons
    for k in range(1, k_annual + 1):
        X[f"year_sin{k}"], X[f"year_cos{k}"] = np.sin(k * annual), np.cos(k * annual)
    # The daily range changes with the season, so we multiply both cycles
    for k in range(1, min(k_interact, k_daily) + 1):
        for d in ("sin", "cos"):
            for a in ("sin", "cos"):
                X[f"day_{d}{k}_x_year_{a}1"] = X[f"day_{d}{k}"] * X[f"year_{a}1"]
    # Finally a linear trend in years for the long term warming
    if trend:
        X["trend_years"] = (ds - TREND_START).days.to_numpy() / 365.25
    return pd.DataFrame(X)


ANOMALY_SUMMARIES = {"last": 1, "1d": 24, "3d": 72, "7d": 168}


def anomaly_states(hist: pd.DataFrame, climatology) -> pd.DataFrame:
    """Recent anomaly (last hour, 1, 3 and 7 days) for every hour."""
    s = hist.set_index("ds")["y"]
    # The anomaly is how far the real temperature is from the climatology
    anom = s - climatology(s.index)
    states = {}
    for name, hours in ANOMALY_SUMMARIES.items():
        if hours == 1:
            states[name] = anom.ffill(limit=6)
        else:
            states[name] = anom.rolling(hours, min_periods=hours // 2).mean()
    return pd.DataFrame(states, index=s.index)


def state_at(states: pd.DataFrame, origin, max_gap_h: int = 24):
    """Last known state before the origin (we never look at the future)."""
    avail = states.loc[states.index < origin].dropna()
    if avail.empty or origin - avail.index[-1] > pd.Timedelta(hours=max_gap_h):
        return None, None
    return avail.iloc[-1].to_dict(), avail.index[-1]


def anomaly_features(ds, state: dict, state_time, feature_set: str) -> pd.DataFrame:
    """Recent anomaly x exp(-h/tau), so its effect fades with the horizon h."""
    h = (pd.DatetimeIndex(ds) - state_time).total_seconds().to_numpy() / 3600
    if feature_set == "simple":
        pairs = [("1d", tau) for tau in (24, 72, 168)]
    elif feature_set == "rich":
        pairs = [(s, tau) for s in ANOMALY_SUMMARIES for tau in DECAYS_H]
    else:
        raise ValueError(feature_set)
    return pd.DataFrame({f"anom_{s}_x_exp_h/{tau}": state[s] * np.exp(-h / tau) for s, tau in pairs})


# ---------------------------------------------------------------------------
# Models
# All of them have fit(hist) and predict(hist, ds). In predict, hist can go
# up to the forecast start, we only use it to know the current state.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LRConfig:
    k_daily: int = 4
    k_annual: int = 3
    k_interact: int = 2
    trend: bool = True
    anomaly: str = "rich"            # "none", "simple" or "rich"
    regularization: str = "none"     # "none", "ridge" or "lasso"
    alpha: float = 0.0
    origin_step_days: int = 2
    season_window_days: int | None = None  # only use origins close to the target date

    @property
    def name(self) -> str:
        return (f"LR(Kd={self.k_daily},Ka={self.k_annual},Ki={self.k_interact},trend={self.trend},"
                f"anom={self.anomaly},reg={self.regularization},alpha={self.alpha},season={self.season_window_days})")


class LinearTemperatureForecaster:
    def __init__(self, config: LRConfig):
        self.config = config

    # ----- stage 1 -----
    def _cal(self, ds):
        c = self.config
        return calendar_features(ds, c.k_daily, c.k_annual, c.k_interact, c.trend)

    def climatology(self, ds) -> np.ndarray:
        return self.stage1.predict(self._cal(ds))

    def _fit_stage1(self, hist):
        obs = hist.dropna(subset=["y"])
        self.stage1 = LinearRegression().fit(self._cal(obs["ds"]), obs["y"])
        self.train_mae_stage1 = mean_absolute_error(obs["y"], self.stage1.predict(self._cal(obs["ds"])))

    # ----- stage 2 -----
    def stage2_training_set(self, hist, target_doy: int | None = None):
        """Build the stage 2 training data by simulating past forecasts."""
        c = self.config
        states = anomaly_states(hist, self.climatology)
        anom = hist.set_index("ds")["y"] - self.climatology(hist["ds"])

        # First we pick one forecast start every few days (at 04:00 UTC like the real one)
        first = hist["ds"].min().normalize() + pd.Timedelta(days=8, hours=4)
        last = hist["ds"].max() - pd.Timedelta(hours=HORIZON)
        origins = pd.date_range(first, last, freq=f"{c.origin_step_days}D")
        if c.season_window_days is not None and target_doy is not None:
            dist = np.abs(((origins.dayofyear - target_doy) + 182) % 365 - 182)
            origins = origins[dist <= c.season_window_days]

        # Then for each start we compute the state with past data only
        # and the next 14 days of anomalies are the target
        X_parts, y_parts = [], []
        for origin in origins:
            state, t_state = state_at(states, origin)
            if state is None:
                continue
            ds = pd.date_range(origin, periods=HORIZON, freq="h")
            X_parts.append(anomaly_features(ds, state, t_state, c.anomaly))
            y_parts.append(anom.reindex(ds).to_numpy())
        X = pd.concat(X_parts, ignore_index=True)
        y = np.concatenate(y_parts)
        keep = np.isfinite(y)
        return X[keep].reset_index(drop=True), y[keep]

    def _make_stage2(self, alpha=None):
        c = self.config
        alpha = c.alpha if alpha is None else alpha
        if c.regularization == "none":
            return LinearRegression()
        # Ridge and Lasso need scaled features, the scaler is fitted on train only
        reg = Ridge(alpha=alpha) if c.regularization == "ridge" else Lasso(alpha=alpha, max_iter=5000)
        return Pipeline([("scaler", StandardScaler()), ("model", reg)])

    def fit(self, hist: pd.DataFrame, target_doy: int | None = None):
        hist = hist[["ds", "y"]]
        # First we fit the climatology, then the anomaly model on its residuals
        self._fit_stage1(hist)
        self.stage2 = None
        if self.config.anomaly != "none":
            X, y = self.stage2_training_set(hist, target_doy)
            self.stage2 = self._make_stage2().fit(X, y)
            self.stage2_feature_names = list(X.columns)
        return self

    def predict(self, hist: pd.DataFrame, ds, return_parts: bool = False):
        ds = pd.DatetimeIndex(ds)
        clim = self.climatology(ds)
        corr = np.zeros(len(ds))
        if self.stage2 is not None:
            # We measure the current anomaly with data before the forecast start
            states = anomaly_states(hist.loc[hist["ds"] < ds[0], ["ds", "y"]], self.climatology)
            state, t_state = state_at(states, ds[0])
            if state is None:
                raise ValueError("Not enough recent data to compute the anomaly state.")
            corr = self.stage2.predict(anomaly_features(ds, state, t_state, self.config.anomaly))
        if return_parts:
            return clim + corr, clim, corr
        return clim + corr


class NaiveLastDay:
    """Baseline: repeat the last observed day."""
    name = "Naive: repeat last day"

    def fit(self, hist, target_doy=None):
        return self

    def predict(self, hist, ds):
        last = hist.loc[hist["ds"] < ds[0]].dropna(subset=["y"]).tail(24)
        by_hour = dict(zip(last["ds"].dt.hour, last["y"]))
        return np.array([by_hour[h] for h in pd.DatetimeIndex(ds).hour], dtype=float)


class MonthHourMean:
    """Baseline: average temperature for each month and hour."""
    name = "Climatology: month-hour mean"

    def fit(self, hist, target_doy=None):
        obs = hist.dropna(subset=["y"])
        self.means = obs.groupby([obs["ds"].dt.month, obs["ds"].dt.hour])["y"].mean()
        return self

    def predict(self, hist, ds):
        ds = pd.DatetimeIndex(ds)
        return np.array([self.means[(m, h)] for m, h in zip(ds.month, ds.hour)], dtype=float)


class SharedFeaturesLR:
    """Simple LR with the 4 shared features, our starting point."""
    name = "LR: shared features"

    def fit(self, hist, target_doy=None):
        obs = hist.dropna(subset=["y"])
        self.model = LinearRegression().fit(shared_features(obs["ds"]), obs["y"])
        return self

    def predict(self, hist, ds):
        return self.model.predict(shared_features(ds))


# ---------------------------------------------------------------------------
# Time series cross-validation
# ---------------------------------------------------------------------------
def cv_folds(years=CV_YEARS, starts=CV_WINDOW_STARTS):
    """For each year we train before Sep 1 and forecast 4 windows of 14 days."""
    for year in years:
        cutoff = pd.Timestamp(f"{year}-09-01", tz="UTC")
        origins = [pd.Timestamp(f"{year}-{s} 04:00", tz="UTC") for s in starts]
        yield cutoff, origins


def cross_validate(make_model, history: pd.DataFrame, years=CV_YEARS) -> pd.DataFrame:
    """Returns one row per forecast hour with the real value and the prediction."""
    y_all = history.set_index("ds")["y"]
    rows = []
    for cutoff, origins in cv_folds(years):
        # We train only with data before the cutoff, so no leakage
        model = make_model().fit(history.loc[history["ds"] < cutoff], target_doy=origins[0].dayofyear)
        for origin in origins:
            ds = pd.date_range(origin, periods=HORIZON, freq="h")
            pred = model.predict(history.loc[history["ds"] < origin], ds)
            rows.append(pd.DataFrame({"fold": cutoff.year, "origin": origin, "ds": ds,
                                      "horizon_h": np.arange(1, HORIZON + 1),
                                      "y": y_all.reindex(ds).to_numpy(), "yhat": pred}))
    out = pd.concat(rows, ignore_index=True)
    out["error"] = out["yhat"] - out["y"]
    return out


def summarize_cv(cv: pd.DataFrame) -> dict:
    per_window = cv.dropna().groupby("origin")["error"].apply(lambda e: e.abs().mean())
    m = calculate_metrics(cv["y"], cv["yhat"])
    m.update({"mae_window_std": float(per_window.std()), "n_windows": int(per_window.size)})
    return m


# ---------------------------------------------------------------------------
# Final configuration, chosen with CV in the analysis notebook:
#   k_daily=4     CV error improves until 4 harmonics, then it is flat
#   k_annual=3    with more harmonics the CV error goes up (overfitting)
#   k_interact=1  more interaction terms do not help
#   trend=True    small gain once we add the anomaly stage
#   anomaly=rich  the biggest improvement (CV MAE 5.22 -> 4.94)
#   lasso 0.01    best CV MAE, it keeps 8 of the 16 features in the final fit
#                 (between 8 and 12 in the CV folds)
# ---------------------------------------------------------------------------
FINAL_CONFIG = LRConfig(k_daily=4, k_annual=3, k_interact=1, trend=True,
                        anomaly="rich", regularization="lasso", alpha=0.01)


# ---------------------------------------------------------------------------
# Adapter for the shared workflow (src/models.py and python -m src.workflow)
# ---------------------------------------------------------------------------
LR_PARAM_TYPES = {
    "k_daily": int, "k_annual": int, "k_interact": int, "trend": bool,
    "anomaly": str, "regularization": str, "alpha": (int, float),
    "origin_step_days": int, "season_window_days": (int, type(None)),
}


def resolve_linear_regression_params(params=None) -> dict:
    """Start from FINAL_CONFIG and check the values from experiment.json."""
    params = dict(params or {})
    unknown = set(params).difference(LR_PARAM_TYPES)
    if unknown:
        raise ValueError(f"Unsupported linear regression keys: {sorted(unknown)}")
    resolved = {key: getattr(FINAL_CONFIG, key) for key in LR_PARAM_TYPES}
    resolved.update(params)
    for key, kind in LR_PARAM_TYPES.items():
        value = resolved[key]
        if type(value) is bool and kind is not bool or not isinstance(value, kind):
            raise ValueError(f"{key} has an invalid type.")
    if resolved["anomaly"] not in ("none", "simple", "rich"):
        raise ValueError("anomaly must be none, simple or rich.")
    if resolved["regularization"] not in ("none", "ridge", "lasso"):
        raise ValueError("regularization must be none, ridge or lasso.")
    if min(resolved["k_daily"], resolved["k_annual"], resolved["origin_step_days"]) < 1 or resolved["k_interact"] < 0:
        raise ValueError("Harmonic counts and origin_step_days must be positive.")
    return resolved


def describe_linear_regression(params: dict) -> list[str]:
    """Short description of the features we really use (saved by the workflow)."""
    parts = [
        f"stage 1: daily Fourier harmonics, order {params['k_daily']} (UTC hour)",
        f"stage 1: annual Fourier harmonics, order {params['k_annual']}",
        f"stage 1: daily x annual interactions up to daily order {params['k_interact']}",
    ]
    if params["trend"]:
        parts.append("stage 1: linear trend in years since 2015")
    if params["anomaly"] != "none":
        reg = params["regularization"] if params["regularization"] != "none" else "OLS"
        parts.append(f"stage 2 ({reg}, alpha={params['alpha']}): recent anomaly vs stage 1 "
                     f"({params['anomaly']} set) x exp(-h/tau), only pre-origin data")
    return parts


def predict_linear_regression(history: pd.DataFrame, future: pd.DataFrame,
                              params=None, feature_columns=None) -> pd.DataFrame:
    """Fit on history and predict every future hour (contract from src/models.py)."""
    if "y" in future.columns:
        raise ValueError("Future inputs must not contain actual temperatures.")
    if feature_columns is not None:
        raise ValueError("Linear regression builds its own features; use feature_sets [null].")
    train = history[["ds", "y"]].copy()
    train["ds"] = pd.to_datetime(train["ds"], utc=True)
    ds = pd.DatetimeIndex(pd.to_datetime(future["ds"], utc=True)).sort_values()
    if train.empty or train["ds"].max() >= ds[0]:
        raise ValueError("Training history must precede the forecast origin.")

    # First we fit both stages on history, then we predict the requested hours
    config = LRConfig(**resolve_linear_regression_params(params))
    model = LinearTemperatureForecaster(config).fit(train, target_doy=ds[0].dayofyear)
    return pd.DataFrame({"ds": ds, "yhat": model.predict(train, ds)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-cv", action="store_true", help="Skip the 36-window cross-validation.")
    args = parser.parse_args()

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    train_val = pd.concat([train, val], ignore_index=True)
    history = load_history()
    models = {
        "naive_last_day": NaiveLastDay,
        "month_hour_mean": MonthHourMean,
        "lr_shared_features": SharedFeaturesLR,
        "lr_climatology_only": lambda: LinearTemperatureForecaster(replace(FINAL_CONFIG, anomaly="none")),
        "lr_final": lambda: LinearTemperatureForecaster(FINAL_CONFIG),
    }
    print(f"Final configuration: {FINAL_CONFIG.name}\n")

    # First we use the same protocol as Prophet:
    # fit on train and score val, then fit on train + val and score test
    rows, test_preds = [], test[["ds", "y"]].copy()
    for split, fit_df, eval_df in (("val", train, val), ("test", train_val, test)):
        scale = make_mase_scale(fit_df)
        for name, make in models.items():
            model = make().fit(fit_df, target_doy=eval_df["ds"].iloc[0].dayofyear)
            pred = model.predict(fit_df, eval_df["ds"])
            rows.append({"split": split, "model": name, **calculate_metrics(eval_df["y"], pred, scale)})
            if split == "test":
                test_preds[name] = pred
    table = pd.DataFrame(rows)
    cols = ["model", "mae", "rmse", "mase", "r2", "bias"]
    for split in ("val", "test"):
        print(f"{split.upper()} metrics\n" + table.loc[table["split"] == split, cols].round(3).to_string(index=False) + "\n")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    test_preds.to_csv(PREDICTION_DIR / "linear_regression_test_predictions.csv", index=False)
    table.to_csv(METRICS_DIR / "linear_regression_val_test_metrics.csv", index=False)

    # Then we run the cross-validation, which is more reliable than one window
    if not args.skip_cv:
        print("Time-series CV: 36 windows (Sep 3/10/17/24 of 2017-2025)")
        cv_rows = []
        for name, make in models.items():
            cv_rows.append({"model": name, **summarize_cv(cross_validate(make, history))})
        cv_table = pd.DataFrame(cv_rows)
        cv_table.to_csv(METRICS_DIR / "linear_regression_cv_metrics.csv", index=False)
        print(cv_table[["model", "mae", "mae_window_std", "rmse", "bias"]].round(3).to_string(index=False) + "\n")

    # Finally we train on all the history and forecast Sep 17-30
    final_ds = pd.date_range(FINAL_ORIGIN, periods=HORIZON, freq="h")
    model = LinearTemperatureForecaster(FINAL_CONFIG).fit(history, target_doy=FINAL_ORIGIN.dayofyear)
    yhat, clim, corr = model.predict(history, final_ds, return_parts=True)
    final = pd.DataFrame({"ds_utc": final_ds, "ds_local": final_ds.tz_convert(LOCAL_TZ),
                          "yhat": yhat, "climatology": clim, "anomaly_correction": corr})
    final.to_csv(PREDICTION_DIR / "linear_regression_final_forecast.csv", index=False)
    print(f"History ends at {history['ds'].max()} | forecast {final['ds_local'].iloc[0]} -> "
          f"{final['ds_local'].iloc[-1]} ({len(final)} h)")


if __name__ == "__main__":
    main()
