"""Check optional weather cleaning, compatibility, and forecast-origin boundaries."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtesting import run_backtests
from src.config import AUDIT_WINDOW, FORECAST_START, VAL_START, VALIDATION_WINDOWS
from src.data_pipeline import (
    WEATHER_COLUMNS, clean_hourly_data, load_hourly_data, prepare_data, validate_hourly_data,
)
from src.models import get_predictor
from src.workflow import run_workflow


class WeatherPipelineTests(unittest.TestCase):
    def test_weather_uses_temperature_report_and_keeps_target_gaps(self):
        raw = pd.DataFrame({
            "station": ["RDU"] * 7,
            "valid": ["2024-01-01 00:50", "2024-01-01 00:51", "2024-01-01 01:51",
                      "2024-01-01 01:49", "2024-01-01 02:40", "2024-01-01 02:51",
                      "2024-01-01 03:51"],
            "tmpf": [60, 61, "M", 62, "M", "M", 64],
            "dwpf": [45, "M", 99, 46, 47, 48, 49],
        })
        default, default_report = clean_hourly_data(raw, "2024-01-01", "2024-01-01 05:00")
        weather, report = clean_hourly_data(raw, "2024-01-01", "2024-01-01 05:00",
                                           include_weather=True)
        pd.testing.assert_frame_equal(default, weather[["ds", "y"]])
        self.assertTrue(pd.isna(weather.loc[0, "dewpoint"]))  # No switching for fuller weather.
        self.assertEqual(weather.loc[1, "dewpoint"], 46)  # Nearest valid temperature report.
        self.assertEqual(weather.loc[2, "dewpoint"], 48)  # Weather remains when y is missing.
        self.assertTrue(pd.isna(weather.loc[2, "y"]))
        self.assertTrue(weather.loc[4, ["y", "dewpoint"]].isna().all())  # No report.
        self.assertEqual(default_report, {key: report[key] for key in default_report})

    def test_weather_units_codes_missing_values_and_invalid_values(self):
        raw = pd.DataFrame({
            "station": "RDU", "valid": pd.date_range("2024-01-01 00:51", periods=4, freq="h"),
            "tmpf": 60, "dwpf": [40, "M", "bad", np.inf],
            "sknt": [10, 0, -1, "M"], "drct": [360, 0, 361, -1],
            "relh": [80, 100, 101, -1], "alti": [30.1, 29.9, 0, np.inf],
            "skyc1": ["FEW", "CLR", "M", "VV"], "skyc2": ["BKN", "M", "unknown", "M"],
            "p01i": ["T", "0.1", "M", "-1"],
        })
        clean, report = clean_hourly_data(raw, "2024-01-01", "2024-01-01 04:00",
                                         include_weather=True)
        self.assertEqual(clean.columns.tolist(), ["ds", "y", *WEATHER_COLUMNS])
        self.assertEqual(clean.loc[0, "dewpoint"], 40)
        self.assertEqual(clean.loc[0, "wind_speed"], 10)
        self.assertEqual(clean.loc[0, "pressure"], 30.1)
        self.assertEqual(clean["cloud_cover"].tolist()[:2], [.875, 0.])
        self.assertTrue(pd.isna(clean.loc[2, "cloud_cover"]))
        self.assertEqual(clean.loc[3, "cloud_cover"], 1.)
        self.assertEqual(clean["precip"].tolist()[:2], [0., .1])
        self.assertEqual(clean["precip_trace"].tolist()[:2], [1., 0.])
        self.assertTrue(clean.loc[2:, ["wind_speed", "wind_direction", "humidity", "pressure",
                                      "precip", "precip_trace"]].isna().all().all())
        self.assertTrue(clean.loc[1:, "dewpoint"].isna().all())
        self.assertEqual(report["Weather values converted to NaN"]["dewpoint"], 2)

    def test_weather_selection_ignores_other_stations_and_cutoff_reports(self):
        raw = pd.DataFrame({
            "station": ["RDU", "OTHER", "RDU"],
            "valid": [FORECAST_START - pd.Timedelta(minutes=9),
                      FORECAST_START - pd.Timedelta(minutes=9), FORECAST_START],
            "tmpf": [60, 999, 999], "sknt": [5, 999, 999],
        })
        clean, _ = clean_hourly_data(raw, FORECAST_START - pd.Timedelta(hours=1),
                                     FORECAST_START, include_weather=True)
        self.assertEqual(clean["y"].tolist(), [60])
        self.assertEqual(clean["wind_speed"].tolist(), [5])

    def test_loading_weather_is_opt_in_and_temperature_only_files_still_work(self):
        frame = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
            "y": 60., "wind_speed": 5., "unrelated": "ignore",
        })
        self.assertEqual(validate_hourly_data(frame).columns.tolist(), ["ds", "y"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hourly.csv"
            frame.to_csv(path, index=False)
            loaded = load_hourly_data(path, include_weather=True)
            pd.testing.assert_frame_equal(loaded, frame[["ds", "y", "wind_speed"]])
            self.assertEqual(load_hourly_data(path).columns.tolist(), ["ds", "y"])
        pd.testing.assert_frame_equal(validate_hourly_data(frame[["ds", "y"]], include_weather=True),
                                      frame[["ds", "y"]])
        for value in (np.inf, "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_hourly_data(frame.assign(wind_speed=value), include_weather=True)

    def test_optional_weather_does_not_change_legacy_exports(self):
        times = pd.date_range(VAL_START - pd.Timedelta(days=2), FORECAST_START,
                              freq="h", inclusive="left")
        raw = pd.DataFrame({"station": "RDU", "valid": times + pd.Timedelta(minutes=51),
                            "tmpf": 60. + times.hour, "sknt": 5.})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw.csv"
            raw.to_csv(input_path, index=False)
            default, splits, _ = prepare_data(input_path, root / "default", times[0])
            weather, weather_splits, _ = prepare_data(input_path, root / "weather", times[0],
                                                      include_weather=True)
            pd.testing.assert_frame_equal(default, weather[["ds", "y"]])
            for name in splits:
                pd.testing.assert_frame_equal(splits[name], weather_splits[name])
                self.assertEqual((root / "default" / f"{name}.csv").read_bytes(),
                                 (root / "weather" / f"{name}.csv").read_bytes())
            self.assertIn("wind_speed", pd.read_csv(root / "weather/hourly.csv"))
            self.assertNotIn("wind_speed", pd.read_csv(root / "default/hourly.csv"))

    def test_backtesting_never_passes_future_weather_or_uses_later_weather(self):
        window = VALIDATION_WINDOWS[0]
        times = pd.date_range(window.start - pd.Timedelta(hours=72), window.end,
                              freq="h", inclusive="left")
        hourly = pd.DataFrame({"ds": times, "y": 60. + times.hour + np.arange(len(times)) / 1000,
                               "wind_speed": 5.})
        def predict(history, future):
            self.assertIn("wind_speed", history)
            self.assertLess(history["ds"].max(), window.start)
            self.assertFalse(set(WEATHER_COLUMNS).intersection(future.columns))
            self.assertNotIn("y", future)
            return future[["ds"]].assign(yhat=history["wind_speed"].iloc[-1])
        changed = hourly.copy()
        changed.loc[changed["ds"] >= window.start, "wind_speed"] = 999.
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, predictions = run_backtests(hourly, {"spy": predict}, windows=[window],
                                                 include_weather=True)
            changed_metrics, changed_predictions = run_backtests(changed, {"spy": predict},
                                                                 windows=[window], include_weather=True)
        pd.testing.assert_frame_equal(metrics, changed_metrics)
        pd.testing.assert_frame_equal(predictions, changed_predictions)
        def default_predict(history, future):
            self.assertNotIn("wind_speed", history)
            return future[["ds"]].assign(yhat=60.)
        with contextlib.redirect_stdout(io.StringIO()):
            run_backtests(hourly, {"spy": default_predict}, windows=[window])

    def test_full_workflow_preserves_weather_through_tuning_audit_and_final_refit(self):
        times = pd.date_range(pd.Timestamp("2015-01-01", tz="UTC"), FORECAST_START,
                              freq="h", inclusive="left")
        raw = pd.DataFrame({"station": "RDU", "valid": times + pd.Timedelta(minutes=51),
                            "tmpf": 60. + times.hour + times.month / 10, "sknt": 5.})
        origins = []
        def factory(name, params=None, feature_columns=None):
            original = get_predictor(name, params, feature_columns)
            def predict(history, future):
                self.assertIn("wind_speed", history)
                self.assertTrue(history["wind_speed"].eq(5.).all())
                self.assertLess(history["ds"].max(), future["ds"].min())
                self.assertFalse(set(WEATHER_COLUMNS).intersection(future.columns))
                self.assertNotIn("y", future)
                origins.append(future["ds"].min())
                return original(history, future)
            return predict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, config = root / "raw.csv", root / "config.json"
            raw.to_csv(input_path, index=False)
            config.write_text(json.dumps({"lookbacks": [None], "models": [{"name": "month_hour"}]}))
            with patch("src.workflow.get_predictor", side_effect=factory), \
                    patch("src.forecast.get_predictor", side_effect=factory), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = run_workflow(config, input_path, root / "run", include_weather=True)
            self.assertTrue(report["include_weather"])
            self.assertEqual(set(origins), {window.start for window in VALIDATION_WINDOWS}
                             | {AUDIT_WINDOW.start, FORECAST_START})
            forecast = pd.read_csv(root / "run/final_predictions.csv")
            self.assertEqual(forecast.columns.tolist(), ["ds", "yhat"])
            self.assertEqual(len(forecast), 336)
            self.assertTrue(np.isfinite(forecast["yhat"]).all())


if __name__ == "__main__":
    unittest.main()
