"""Project conventions and fixed forecast windows; stored timestamps are UTC."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/RDU_starting2015_UTC.csv"
DATA_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STATION = "RDU"
LOCAL_TIMEZONE = "America/New_York"
TARGET_UNIT = "degF"
PREFERRED_MINUTE = 51
FORECAST_HOURS = 14 * 24
DATA_START = pd.Timestamp("2015-01-01", tz="UTC")


def eastern_midnight(date: str) -> pd.Timestamp:
    """Convert a local midnight to an unambiguous UTC boundary."""
    return pd.Timestamp(date, tz=LOCAL_TIMEZONE).tz_convert("UTC")


FORECAST_START = eastern_midnight("2026-09-17")
FORECAST_END = eastern_midnight("2026-10-01")
TEST_START = FORECAST_START - pd.Timedelta(days=14)
VAL_START = TEST_START - pd.Timedelta(days=14)


@dataclass(frozen=True)
class ForecastWindow:
    """A 336-hour fixed-origin forecast, with an exclusive end boundary."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self):
        for boundary in (self.start, self.end):
            if boundary.tzinfo is None or boundary != boundary.floor("h"):
                raise ValueError("Window boundaries must be timezone-aware exact hours.")
        if self.end - self.start != pd.Timedelta(hours=FORECAST_HOURS):
            raise ValueError("Each project forecast window must contain 336 hours.")
        object.__setattr__(self, "start", self.start.tz_convert("UTC"))
        object.__setattr__(self, "end", self.end.tz_convert("UTC"))

    def timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="h", inclusive="left")


VALIDATION_WINDOWS = tuple(
    ForecastWindow(
        f"september_{year}",
        eastern_midnight(f"{year}-09-17"),
        eastern_midnight(f"{year}-10-01"),
    )
    for year in (2022, 2023, 2024, 2025)
) + (ForecastWindow("recent_2026", VAL_START, TEST_START),)
AUDIT_WINDOW = ForecastWindow("audit_2026", TEST_START, FORECAST_START)
SUBMISSION_WINDOW = ForecastWindow("submission_2026", FORECAST_START, FORECAST_END)

# An hourly label denotes the selected observation in that hour, normally :51.
# Example: 2026-09-17 04:00 UTC predicts the 04:51 UTC (00:51 Eastern) report.
OBSERVATION_POLICY = "valid temperature nearest :51 within the containing UTC hour"
