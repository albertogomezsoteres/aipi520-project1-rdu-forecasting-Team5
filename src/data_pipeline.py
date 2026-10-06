"""
Shared data pipeline for RDU hourly temperature forecasting.

1. Load raw weather observations, parse missing values, convert timestamps
   to UTC, and convert temperature values to numbers.
2. Keep RDU observations within the requested historical period.
3. Remove identical duplicate rows and check for conflicting temperatures
   at the same timestamp.
4. Select one valid temperature per hour: prefer the :51 observation;
   otherwise use the valid observation closest to :51 within that hour.
5. Restore the complete hourly timeline, preserving missing temperatures
   as NaN rather than filling them.
6. Save the canonical hourly dataset as hourly.csv with ds/y columns and,
   when requested, numeric historical weather inputs from the same reports.
7. Create the original chronological training, validation, and test splits,
   add shared calendar and cyclic features, and save each split as a CSV.
8. Save a cleaning report and print cleaning and split coverage summaries.

Default historical period:
    January 1, 2015 00:00 UTC through September 17, 2026 04:00 UTC,
    with the end boundary excluded.

Legacy validation window:
    August 20, 2026 04:00 UTC through September 3, 2026 04:00 UTC.

Legacy test window:
    September 3, 2026 04:00 UTC through September 17, 2026 04:00 UTC.

All stored timestamps and shared calendar features use UTC.
Each hourly label represents the selected observation within that hour,
normally recorded at :51. Temperatures are in degrees Fahrenheit.

Optional weather inputs retain source units: dew point (Fahrenheit), wind
speed (knots), direction (degrees), humidity (percent), pressure (inches Hg),
and precipitation (inches, with a separate trace flag). Cloud cover is a
fractional proxy for the highest reported coverage. Missing values remain
NaN. Weather is retained even when an hour has a report but no temperature;
legacy split exports keep their original temperature/calendar columns.

Historical multi-window validation and model fitting are handled separately
by backtesting.py. This pipeline does not fit models or generate forecasts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Keep the original direct-script invocation working; prefer python -m src.data_pipeline.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    DATA_DIR, DATA_START, FORECAST_START, OBSERVATION_POLICY, PREFERRED_MINUTE,
    RAW_PATH, STATION, TEST_START, VAL_START,
)
from src.features import add_time_features

WEATHER_COLUMNS = (
    "dewpoint", "wind_speed", "wind_direction", "humidity", "pressure",
    "cloud_cover", "precip", "precip_trace",
)


def _parse_weather(raw):
    """Parse available ASOS fields; leave missing/invalid values unfilled."""
    parsed = pd.DataFrame(index=raw.index)
    invalid = {}
    sources = {"dwpf": "dewpoint", "sknt": "wind_speed", "drct": "wind_direction",
               "relh": "humidity", "alti": "pressure"}
    for source, name in sources.items():
        if source not in raw:
            continue
        original = raw[source].mask(raw[source].eq("M").fillna(False))
        values = pd.to_numeric(original, errors="coerce").astype(float)
        bad = ~np.isfinite(values) & values.notna()
        if name == "wind_speed":
            bad |= values.lt(0)
        elif name == "pressure":
            bad |= values.le(0)
        elif name == "humidity":
            bad |= ~values.between(0, 100) & values.notna()
        elif name == "wind_direction":
            bad |= ~values.between(0, 360) & values.notna()
        invalid[name] = int(((original.notna() & values.isna()) | bad).sum())
        parsed[name] = values.mask(bad)

    layers = [name for name in ("skyc1", "skyc2", "skyc3", "skyc4") if name in raw]
    if layers:
        fractions = {"CLR": 0., "SKC": 0., "FEW": .25, "SCT": .5,
                     "BKN": .875, "OVC": 1., "VV": 1.}
        encoded = pd.DataFrame({
            name: raw[name].astype("string").str.strip().str.upper().map(fractions)
            for name in layers
        })
        # A coverage proxy, not summed layers; missing layers do not mean clear.
        parsed["cloud_cover"] = encoded.max(axis=1)
    if "p01i" in raw:
        text = raw["p01i"].astype("string").str.strip().str.upper().replace("M", pd.NA)
        trace = text.eq("T").fillna(False)
        amount = pd.to_numeric(text.mask(trace, "0"), errors="coerce").astype(float)
        bad = amount.lt(0) | (~np.isfinite(amount) & amount.notna())
        invalid["precip"] = int(((text.notna() & amount.isna()) | bad).sum())
        parsed["precip"] = amount.mask(bad)
        parsed["precip_trace"] = trace.astype(float).where(parsed["precip"].notna())
    return parsed, invalid


def clean_hourly_data(raw, start, end, *, include_weather=False):
    """Select the valid observation nearest :51 and restore the hourly grid.

    Opt-in weather comes from that same report. For hours without any valid
    temperature, use the nearest report for weather only, keeping y missing.
    """
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if pd.isna(start) or pd.isna(end) or start >= end or any(
        boundary != boundary.floor("h") for boundary in (start, end)
    ):
        raise ValueError("Cleaning boundaries must be ordered exact UTC hours.")
    if end > FORECAST_START:
        raise ValueError("Cleaning cannot include observations after the submission cutoff.")
    required = {"station", "valid", "tmpf"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = raw.copy()
    data["valid"] = pd.to_datetime(data["valid"], utc=True, errors="raise")
    if data["valid"].isna().any():
        raise ValueError("Missing observation timestamps in valid column.")
    data = data.loc[
        (data["station"] == STATION)
        & (data["valid"] >= start)
        & (data["valid"] < end)
    ].copy()
    if data.empty:
        raise ValueError("No RDU observations within the requested date range.")

    temperature = pd.to_numeric(data["tmpf"], errors="coerce")
    invalid_values = int((data["tmpf"].notna() & temperature.isna()).sum())
    data["tmpf"] = temperature
    if np.isinf(data["tmpf"].to_numpy(dtype=float)).any():
        raise ValueError("Infinite temperature values found.")

    identical_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    data["ds"] = data["valid"].dt.floor("h")
    duplicate_timestamps = data.loc[
        data.duplicated("valid", keep=False), "valid"
    ].nunique()
    multiple_record_hours = data.loc[
        data.duplicated("ds", keep=False), "ds"
    ].nunique()

    usable = data.dropna(subset=["tmpf"]).copy()
    counts = usable.groupby("valid")["tmpf"].nunique()
    conflicts = counts[counts > 1].index
    if len(conflicts):
        details = usable.loc[
            usable["valid"].isin(conflicts), ["valid", "tmpf"]
        ].to_string(index=False)
        raise ValueError(f"Conflicting temperatures at the same timestamp:\n{details}")

    preferred_time = usable["ds"] + pd.Timedelta(minutes=PREFERRED_MINUTE)
    usable["distance_to_51"] = (usable["valid"] - preferred_time).abs()
    usable = usable.sort_values(
        ["ds", "distance_to_51", "valid"], ascending=[True, True, False]
    )
    selected = usable.drop_duplicates("ds", keep="first").copy()
    expected_hours = pd.date_range(start, end, freq="h", inclusive="left")
    reported_hours = pd.DatetimeIndex(data["ds"].unique())
    usable_hours = pd.DatetimeIndex(selected["ds"])

    clean = (
        selected[["ds", "tmpf"]]
        .rename(columns={"tmpf": "y"})
        .set_index("ds")
        .reindex(expected_hours)
        .rename_axis("ds")
        .reset_index()
    )
    selected_51 = int(selected["valid"].dt.minute.eq(PREFERRED_MINUTE).sum())
    report = {
        "Non-numeric temperatures converted to NaN": invalid_values,
        "Identical duplicate rows removed": identical_duplicates,
        "Repeated exact timestamps": int(duplicate_timestamps),
        "Hours with multiple records": int(multiple_record_hours),
        "Hours using :51 observations": selected_51,
        "Hours using other-minute observations": len(selected) - selected_51,
        "Hours without any report": len(expected_hours.difference(reported_hours)),
        "Reported hours without valid temperature": len(reported_hours.difference(usable_hours)),
        "Total missing targets": int(clean["y"].isna().sum()),
        "Total hours": len(clean),
        "Observed targets": int(clean["y"].notna().sum()),
    }
    if include_weather:
        weather, invalid_weather = _parse_weather(data)
        data = data.join(weather)
        missing_target = data.loc[~data["ds"].isin(selected["ds"])].copy()
        preferred = missing_target["ds"] + pd.Timedelta(minutes=PREFERRED_MINUTE)
        missing_target["distance_to_51"] = (missing_target["valid"] - preferred).abs()
        fallback = missing_target.sort_values(
            ["ds", "distance_to_51", "valid"], ascending=[True, True, False]
        ).drop_duplicates("ds")
        # Never substitute a different report just because its weather is fuller.
        weather_rows = pd.concat([data.loc[selected.index], fallback])
        clean = clean.merge(weather_rows[["ds", *weather.columns]], on="ds", how="left")
        report["Weather values converted to NaN"] = invalid_weather
        report["Observed weather values"] = {
            name: int(clean[name].notna().sum()) for name in weather.columns
        }
    return clean, report


def split_data(clean, val_start, test_start, end):
    """Split chronologically using exclusive upper boundaries."""
    splits = {
        "train": clean.loc[clean["ds"] < val_start].copy(),
        "val": clean.loc[(clean["ds"] >= val_start) & (clean["ds"] < test_start)].copy(),
        "test": clean.loc[(clean["ds"] >= test_start) & (clean["ds"] < end)].copy(),
    }
    for name in ("val", "test"):
        if len(splits[name]) != 14 * 24:
            raise ValueError(f"{name} must contain 336 hours.")
    if splits["train"].empty or sum(map(len, splits.values())) != len(clean):
        raise ValueError("Invalid split boundaries or empty training data.")
    return splits


def validate_hourly_data(frame, *, include_weather=False):
    """Validate the hourly timeline, optionally retaining supported weather."""
    if not {"ds", "y"}.issubset(frame.columns):
        raise ValueError("Hourly data requires ds and y columns.")
    weather = [name for name in WEATHER_COLUMNS if name in frame] if include_weather else []
    data = frame[["ds", "y", *weather]].copy()
    data["ds"] = pd.to_datetime(data["ds"], utc=True, errors="raise")
    data["y"] = pd.to_numeric(data["y"], errors="raise")
    data = data.sort_values("ds").reset_index(drop=True)
    if data.empty or data["ds"].isna().any() or not data["ds"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all() or not data["ds"].eq(data["ds"].dt.floor("h")).all():
        raise ValueError("Hourly data must have a nonempty complete unique hourly grid.")
    if data["ds"].max() >= FORECAST_START:
        raise ValueError("Canonical data must end before the submission cutoff.")
    if np.isinf(data["y"].to_numpy(dtype=float)).any():
        raise ValueError("Infinite target values are invalid.")
    for name in weather:
        data[name] = pd.to_numeric(data[name], errors="raise")
        if np.isinf(data[name].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite weather values are invalid: {name}.")
    return data


def load_hourly_data(path=DATA_DIR / "hourly.csv", *, include_weather=False):
    return validate_hourly_data(pd.read_csv(path), include_weather=include_weather)


def prepare_data(input_path=RAW_PATH, output_dir=DATA_DIR, start=DATA_START, *, include_weather=False):
    """Clean once, save hourly.csv and a report, and preserve legacy exports."""
    raw = pd.read_csv(input_path, na_values=["M"], low_memory=False)
    clean, report = clean_hourly_data(raw, start, FORECAST_START, include_weather=include_weather)
    clean = validate_hourly_data(clean, include_weather=include_weather)
    splits = split_data(clean[["ds", "y"]], VAL_START, TEST_START, FORECAST_START)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_dir / "hourly.csv", index=False)
    for name, frame in splits.items():
        add_time_features(frame).to_csv(output_dir / f"{name}.csv", index=False)
    metadata = {
        "input": str(Path(input_path).resolve()),
        "start_utc": clean["ds"].min().isoformat(),
        "end_exclusive_utc": FORECAST_START.isoformat(),
        "observation_policy": OBSERVATION_POLICY,
        "cleaning": report,
        "splits": {
            name: {"hours": len(frame), "observed_targets": int(frame["y"].notna().sum())}
            for name, frame in splits.items()
        },
    }
    (output_dir / "cleaning_report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return clean, splits, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--start", default="2015-01-01", help="Inclusive UTC start hour.")
    parser.add_argument("--include-weather", action="store_true", help="Retain available historical weather in hourly.csv; legacy splits are unchanged.")
    args = parser.parse_args()
    clean, splits, report = prepare_data(args.input, args.output_dir, args.start, include_weather=args.include_weather)
    print(f"Saved canonical hourly data and legacy splits to {args.output_dir.resolve()}")
    for label, value in report.items():
        print(f"{label}: {value}")
    for name, frame in splits.items():
        print(f"{name}: {len(frame)} hours; {frame['y'].isna().sum()} missing targets")


if __name__ == "__main__":
    main()
