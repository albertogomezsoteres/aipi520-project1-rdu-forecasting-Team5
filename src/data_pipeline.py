"""
Shared data pipeline for RDU hourly temperature forecasting.

1. Load raw weather observations and convert timestamps to UTC.
2. Remove identical duplicates and select one valid temperature per hour:
   prefer the :51 observation; otherwise use the observation closest to :51.
3. Restore the complete hourly timeline, leaving missing temperatures as NaN.
4. Split the data chronologically into training, validation, and test sets.
5. Add calendar features and cyclic encodings for daily and annual patterns.
6. Save the processed datasets and print duplicate and missing-data summaries.

Default period: January 1, 2015 to September 17, 2026 (exclusive).
Validation: August 20-September 2, 2026.
Test: September 3-16, 2026.
All time boundaries use UTC.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# All boundaries are UTC; upper boundaries are exclusive
# same thing as forecasting EDT [2026-9-17 00:00:00 -  2026.10.01 00:00:00) 
FORECAST_START = pd.Timestamp("2026-09-17 04:00:00", tz="UTC")
FORECAST_END = pd.Timestamp("2026-10-01 04:00:00", tz="UTC")

TEST_START = FORECAST_START - pd.Timedelta(days=14)
VAL_START = TEST_START - pd.Timedelta(days=14)


def clean_hourly_data(raw, start, end):
    """Select the valid observation nearest :51 and restore the hourly grid."""
    required = {"station", "valid", "tmpf"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = raw.copy()
    data["valid"] = pd.to_datetime(data["valid"], utc=True, errors="raise")
    if data["valid"].isna().any():
        raise ValueError("Missing observation timestamps in valid column.")
    data = data.loc[
        (data["station"] == "RDU")
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
    data = data.drop_duplicates().copy()
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

    preferred_time = usable["ds"] + pd.Timedelta(minutes=51)
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
    selected_51 = int(selected["valid"].dt.minute.eq(51).sum())
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


def add_time_features(frame):
    """Add the notebook's shared UTC calendar and cyclic features."""
    data = frame.copy()
    data["hour"] = data["ds"].dt.hour
    data["month"] = data["ds"].dt.month
    data["dayofyear"] = data["ds"].dt.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    days_in_year = np.where(data["ds"].dt.is_leap_year, 366, 365)
    annual_position = (data["dayofyear"] - 1 + data["hour"] / 24) / days_in_year
    data["year_sin"] = np.sin(2 * np.pi * annual_position)
    data["year_cos"] = np.cos(2 * np.pi * annual_position)
    return data


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name in {"src", "scripts", "notebooks"} else script_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "data/raw/RDU_starting2015_UTC.csv",
                        help="Raw CSV path; explicit relative paths are relative to the current directory.")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data/processed",
                        help="Destination for train.csv, val.csv and test.csv (overwritten on each run).")
    parser.add_argument("--start", default="2015-01-01", help="Inclusive UTC start date, e.g. 2021-01-01.")
    parser.add_argument("--export-2021", action="store_true",
                        help="Also export RDU_starting2021_UTC.csv next to the input file.")
    args = parser.parse_args()

    start = pd.to_datetime(args.start, utc=True)
    val_start = VAL_START
    test_start = TEST_START
    end = FORECAST_START
    if pd.isna(start) or start != start.floor("h") or start >= val_start:
        parser.error("--start must be an exact hour before 2026-08-20 UTC.")
    if not args.input.is_file():
        parser.error(f"Input CSV not found: {args.input}. Place the file there or use --input PATH.")

    raw = pd.read_csv(args.input, na_values=["M"], low_memory=False)
    print(f"Input: {args.input.resolve()}\nRaw shape: {raw.shape}")
    clean, report = clean_hourly_data(raw, start, end)
    print("\nCleaning summary")
    for label, value in report.items():
        print(f"{label}: {value}")

    splits = split_data(clean, val_start, test_start, end)
    summary = pd.DataFrame([
        {"Split": name, "Start": frame["ds"].min(), "End": frame["ds"].max(),
         "Total hours": len(frame), "Observed targets": int(frame["y"].notna().sum()),
         "Missing targets": int(frame["y"].isna().sum())}
        for name, frame in splits.items()
    ])
    print("\nSplit summary\n" + summary.to_string(index=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        path = args.output_dir / f"{name}.csv"
        add_time_features(frame).to_csv(path, index=False)
        print(f"Saved: {path.resolve()}")

    if args.export_2021:
        timestamps = pd.to_datetime(raw["valid"], utc=True)
        subset = raw.loc[timestamps >= pd.Timestamp("2021-01-01", tz="UTC")]
        path = args.input.parent / "RDU_starting2021_UTC.csv"
        if path.resolve() == args.input.resolve():
            print("Skipped 2021 export: destination is the input file.")
        else:
            subset.to_csv(path, index=False, na_rep="M")
            print(f"Saved 2021 raw subset: {path.resolve()} ({len(subset)} rows)")


if __name__ == "__main__":
    main()
