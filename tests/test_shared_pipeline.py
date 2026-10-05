"""Focused checks for temporal boundaries, forecast completeness, and missing data."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtesting import get_predictor, make_fold, run_backtests, select_history, summarize_results
from src.baselines import predict_baseline
from src.config import FORECAST_START, SUBMISSION_WINDOW, VALIDATION_WINDOWS
from src.data_pipeline import clean_hourly_data
from src.evaluation import calculate_metrics, evaluate_forecast, make_mase_scale, validate_predictions
from src.features import add_time_features
from src.forecast import generate_forecast
from src.prophet_model import BEST_PROPHET_PARAMS, predict_prophet


class SharedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = pd.date_range(pd.Timestamp("2015-01-01", tz="UTC"), FORECAST_START, freq="h", inclusive="left")
        cls.hourly = pd.DataFrame({"ds": times, "y": 50 + times.hour + times.month / 10})

    def test_local_midnight_and_submission_grid(self):
        times = SUBMISSION_WINDOW.timestamps()
        self.assertEqual(times[0], pd.Timestamp("2026-09-17 04:00", tz="UTC"))
        self.assertEqual(times[-1], pd.Timestamp("2026-10-01 03:00", tz="UTC"))
        self.assertEqual(len(times), 336)
        self.assertEqual(times[0].tz_convert("America/New_York").hour, 0)

    def test_cleaner_prefers_valid_51_report_and_preserves_gap(self):
        start = FORECAST_START - pd.Timedelta(hours=3)
        raw = pd.DataFrame({
            "station": ["RDU"] * 5,
            "valid": [start + pd.Timedelta(minutes=31), start + pd.Timedelta(minutes=51),
                      start + pd.Timedelta(hours=1, minutes=51), FORECAST_START,
                      FORECAST_START - pd.Timedelta(minutes=9)],
            "tmpf": [60, 61, "M", 999, 62],
        })
        clean, report = clean_hourly_data(raw, start, FORECAST_START)
        np.testing.assert_allclose(clean["y"], [61, np.nan, 62], equal_nan=True)
        self.assertEqual(report["Total missing targets"], 1)

    def test_conflicting_same_timestamp_temperatures_raise(self):
        raw = pd.DataFrame({"station": ["RDU", "RDU"], "valid": ["2025-01-01 00:51"] * 2,
                            "tmpf": [50, 51]})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            clean_hourly_data(raw, "2025-01-01", "2025-01-02")

    def test_fold_has_no_future_labels_or_later_training_rows(self):
        window = VALIDATION_WINDOWS[2]
        history, future, actual = make_fold(self.hourly, window, 3)
        self.assertEqual(history["ds"].min(), window.start - pd.DateOffset(years=3))
        self.assertLess(history["ds"].max(), window.start)
        self.assertNotIn("y", future)
        self.assertEqual(len(actual), 336)

    def test_missing_target_does_not_compress_calendar_grid(self):
        data = self.hourly.copy()
        window = VALIDATION_WINDOWS[3]
        data.loc[data["ds"] == window.start + pd.Timedelta(hours=10), "y"] = np.nan
        _, future, actual = make_fold(data, window)
        self.assertEqual(len(future), 336)
        self.assertEqual(actual["y"].notna().sum(), 335)

    def test_insufficient_lookback_and_missing_pre_origin_hour_raise(self):
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_history(self.hourly, VALIDATION_WINDOWS[0].start, 10)
        with self.assertRaisesRegex(ValueError, "immediately before"):
            select_history(self.hourly.iloc[:-1], FORECAST_START)

    def test_scoring_masks_only_missing_actuals(self):
        metrics = calculate_metrics([10, np.nan, 30], [12, 99, 28], 2)
        self.assertEqual(metrics["mae"], 2)
        self.assertEqual(metrics["hours_expected"], 3)
        self.assertEqual(metrics["hours_evaluated"], 2)
        for values in ([10, np.nan, 30], [10, np.inf, 30]):
            with self.assertRaisesRegex(ValueError, "finite prediction"):
                calculate_metrics([10, np.nan, 30], values, 2)

    def test_predictions_align_by_time_and_reject_wrong_grid(self):
        actual = self.hourly.iloc[:3].copy()
        predictions = actual.rename(columns={"y": "yhat"}).iloc[::-1]
        metrics, comparison = evaluate_forecast(actual, predictions, 1)
        self.assertEqual(metrics["mae"], 0)
        self.assertTrue(comparison["ds"].equals(actual["ds"]))
        with self.assertRaises(ValueError):
            validate_predictions(predictions.iloc[:2], actual["ds"])
        duplicate = pd.concat([predictions.iloc[:2], predictions.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_predictions(duplicate, actual["ds"])

    def test_mase_keeps_elapsed_hour_pairs_across_gaps(self):
        frame = self.hourly.iloc[:49].copy()
        frame["y"] = np.arange(49, dtype=float)
        frame.loc[1, "y"] = np.nan
        self.assertEqual(make_mase_scale(frame), 24)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            make_mase_scale(frame.drop(index=1))

    def test_repeat_day_preserves_hour_alignment_when_history_has_gap(self):
        origin = VALIDATION_WINDOWS[0].start
        history = select_history(self.hourly, origin)
        history.loc[history["ds"] == origin - pd.Timedelta(hours=23), "y"] = np.nan
        future = pd.DataFrame({"ds": pd.date_range(origin, periods=48, freq="h")})
        predictions = predict_baseline(history, future, "repeat_last_day")
        self.assertEqual(predictions.attrs["last_day_fallback_hours"], 1)
        self.assertTrue(np.isfinite(predictions["yhat"]).all())
        self.assertEqual(predictions.loc[2, "yhat"], history.iloc[-22]["y"])
        np.testing.assert_array_equal(predictions["yhat"][:24], predictions["yhat"][24:])

    def test_backtesting_uses_same_mase_reference_for_different_histories(self):
        calls = []
        def spy(history, future):
            self.assertNotIn("y", future)
            self.assertLess(history["ds"].max(), future["ds"].min())
            calls.append(len(history))
            return predict_baseline(history, future, "month_hour")
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, predictions = run_backtests(self.hourly, {"spy": spy}, [None, 3], VALIDATION_WINDOWS[:1])
        self.assertEqual(metrics["mase_scale"].nunique(), 1)
        self.assertGreater(calls[0], calls[1])
        self.assertEqual(len(predictions), 672)

    def test_summary_weights_windows_equally_and_rejects_missing_folds(self):
        metrics = pd.DataFrame({
            "model": ["a", "a"], "lookback": ["all", "all"], "window": ["one", "two"],
            "mae": [2, 8], "rmse": [3, 9], "mase": [1, 4], "bias": [0, 1],
            "hours_evaluated": [336, 334],
        })
        self.assertEqual(summarize_results(metrics).iloc[0]["mean_mae"], 5)
        incomplete = pd.concat([metrics, metrics.iloc[:1].assign(model="b")])
        with self.assertRaisesRegex(ValueError, "same windows"):
            summarize_results(incomplete)

    def test_final_refit_uses_eligible_audit_data_and_exact_grid(self):
        captured = {}
        def spy(history, future):
            captured["last"] = history["ds"].max()
            self.assertNotIn("y", future)
            return predict_baseline(history, future, "month_hour")
        forecast, history = generate_forecast(self.hourly, {"model": "month_hour", "lookback_years": 3}, spy)
        self.assertEqual(captured["last"], FORECAST_START - pd.Timedelta(hours=1))
        self.assertEqual(len(forecast), 336)
        self.assertEqual(history["ds"].min(), FORECAST_START - pd.DateOffset(years=3))

    def test_prophet_adapter_keeps_utc_clock_and_drops_missing_training_labels(self):
        history, future, _ = make_fold(self.hourly, VALIDATION_WINDOWS[0], 3)
        history.loc[0, "y"] = np.nan
        class FakeProphet:
            def fit(inner, frame):
                self.assertIsNone(frame["ds"].dt.tz)
                self.assertFalse(frame["y"].isna().any())
            def predict(inner, frame):
                self.assertIsNone(frame["ds"].dt.tz)
                self.assertEqual(frame["ds"].iloc[0].hour, 4)
                return frame.assign(yhat=70., yhat_lower=60., yhat_upper=80.)
        with patch("src.prophet_model.build_model", return_value=FakeProphet()) as builder:
            forecast = predict_prophet(history, future)
        builder.assert_called_once_with(None)
        self.assertEqual(str(forecast["ds"].dt.tz), "UTC")
        self.assertEqual(BEST_PROPHET_PARAMS["changepoint_prior_scale"], 0.005)
        with self.assertRaisesRegex(ValueError, "actual temperatures"):
            predict_prophet(history, future.assign(y=0))

    def test_shared_feature_values_are_unchanged(self):
        frame = pd.DataFrame({"ds": pd.to_datetime(["2024-03-01 06:00"], utc=True)})
        features = add_time_features(frame)
        self.assertAlmostEqual(features.loc[0, "hour_sin"], 1)
        self.assertAlmostEqual(features.loc[0, "year_sin"], np.sin(2 * np.pi * (60 + 6/24) / 366))


if __name__ == "__main__":
    unittest.main()
