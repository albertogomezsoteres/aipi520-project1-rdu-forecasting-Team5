"""Check forecast completeness, fresh fitting, and leakage-safe boosting features."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.backtesting import load_selection
from src.config import FORECAST_START, SUBMISSION_WINDOW
from src.features import CALENDAR_FEATURES
from src.gradient_boosting_features import (
    LAG_FEATURES, TEMPERATURE_CONTEXT, WEATHER_GROUPS, add_temperature_lags,
    make_direct_future_data, make_direct_training_data, origin_context,
)
from src.gradient_boosting_model import predict_gradient_boosting
from src.models import get_predictor, model_config


class GradientBoostingTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range(FORECAST_START - pd.Timedelta(days=60), FORECAST_START,
                              freq="h", inclusive="left")
        # Increasing targets make elapsed-hour alignment easy to check.
        self.history = pd.DataFrame({
            "ds": times, "y": np.arange(len(times), dtype=float),
            "dewpoint": 45., "wind_speed": 5., "wind_direction": 350.,
            "humidity": 70., "pressure": 30., "cloud_cover": .5,
            "precip": 0., "precip_trace": 0.,
        })
        self.future = pd.DataFrame({"ds": SUBMISSION_WINDOW.timestamps()})
        self.direct = CALENDAR_FEATURES + ["forecast_hour"] + TEMPERATURE_CONTEXT

    def test_invalid_settings_and_feature_sets_fail_before_fitting(self):
        invalid = [
            ({"unknown": 1}, None), ({"early_stopping": True}, None),
            ({"learning_rate": np.nan}, None), ({"max_iter": 0}, None),
            ({"random_state": True}, None), ({}, ["y"]), ({}, ["hour", "hour"]),
            ({}, ["lag_1"]), ({}, self.direct + LAG_FEATURES),
            ({}, ["forecast_hour"]), ({}, ["dewpoint_last"]),
            ({"forecast_strategy": "recursive"}, None),
            ({"origin_stride_hours": 25}, self.direct), ({"horizon_hours": 337}, self.direct),
        ]
        with patch("src.gradient_boosting_model.build_model") as builder:
            for params, columns in invalid:
                with self.subTest(params=params, columns=columns), self.assertRaises(ValueError):
                    get_predictor("gradient_boosting", params, columns)(self.history, self.future)
        builder.assert_not_called()

    def test_adapter_refits_and_respects_selected_inputs(self):
        self.history.loc[10, "y"] = np.nan
        self.future["hour"] = 999  # Calendar inputs must be recomputed from ds.
        before_history, before_future = self.history.copy(), self.future.copy()
        estimators = []

        class SpyEstimator:
            def fit(inner, inputs, labels):
                self.assertEqual(inputs.columns.tolist(), ["hour", "month"])
                self.assertEqual(len(labels), len(self.history) - 1)
                self.assertFalse(labels.isna().any())

            def predict(inner, inputs):
                self.assertEqual(inputs.iloc[0]["hour"], 4)
                return np.full(len(inputs), 65.)

        def build(params):
            self.assertEqual(params["max_iter"], 3)
            estimators.append(SpyEstimator())
            return estimators[-1]

        predict = get_predictor("gradient_boosting", {"max_iter": 3}, ["hour", "month"])
        with patch("src.gradient_boosting_model.build_model", side_effect=build):
            for _ in range(2):
                result = predict(self.history, self.future)
        self.assertEqual(len(estimators), 2)
        self.assertIsNot(estimators[0], estimators[1])
        self.assertEqual(result.columns.tolist(), ["ds", "yhat"])
        self.assertEqual(result["ds"].tolist(), self.future["ds"].tolist())
        pd.testing.assert_frame_equal(self.history, before_history)
        pd.testing.assert_frame_equal(self.future, before_future)

    def test_invalid_forecast_data_is_rejected_before_fit(self):
        cases = [
            (self.history, self.future.assign(y=np.nan), None),
            (self.history, self.future.assign(dewpoint=45), self.direct),
            (pd.concat([self.history, self.future.iloc[:1].assign(y=1.)]), self.future, None),
            (self.history, pd.concat([self.future.iloc[:1], self.future]), None),
            (self.history, self.future.drop(index=10), None),
            (self.history.assign(y=np.inf), self.future, None),
            (self.history.drop(index=10), self.future, CALENDAR_FEATURES + LAG_FEATURES),
            (self.history.iloc[:-1], self.future, self.direct),
            (self.history.drop(columns="dewpoint"), self.future, self.direct + ["dewpoint_last"]),
        ]
        with patch("src.gradient_boosting_model.build_model") as builder:
            for history, future, columns in cases:
                with self.subTest(columns=columns), self.assertRaises(ValueError):
                    predict_gradient_boosting(history, future, feature_columns=columns)
        builder.return_value.fit.assert_not_called()

    def test_lags_preserve_elapsed_hours_and_missing_values(self):
        self.history.loc[5, "y"] = np.nan
        features = add_temperature_lags(self.history)
        self.assertTrue(pd.isna(features.loc[29, "lag_24"]))
        self.assertEqual(features.loc[30, "lag_24"], 6)
        self.assertAlmostEqual(features.loc[24, "rolling_mean_24"], self.history.loc[:23, "y"].mean())
        self.history.loc[:6, "y"] = np.nan
        self.assertTrue(pd.isna(add_temperature_lags(self.history).loc[24, "rolling_mean_24"]))

    def test_origin_context_excludes_origin_and_later_observations(self):
        columns = TEMPERATURE_CONTEXT + WEATHER_GROUPS["dew_point"] + WEATHER_GROUPS["pressure"]
        original = origin_context(self.history, columns)
        changed = self.history.copy()
        changed.loc[100:, ["y", "dewpoint", "pressure"]] = 9999
        pd.testing.assert_series_equal(original.loc[100], origin_context(changed, columns).loc[100])
        self.assertEqual(original.loc[100, "temp_last"], 99)
        self.assertEqual(original.loc[100, "temp_change_24"], 24)
        self.assertEqual(original.loc[100, "temp_mean_24"], np.arange(76, 100).mean())

    def test_weather_context_handles_direction_wraparound_and_missing_precipitation(self):
        history = self.history.iloc[:48].copy()
        history["wind_direction"] = np.tile([350., 10.], 24)
        columns = WEATHER_GROUPS["wind_direction"]
        wind = origin_context(history, columns)
        self.assertAlmostEqual(wind.loc[24, "wind_u_mean_24"], 0.)
        self.assertLess(wind.loc[24, "wind_v_mean_24"], -4.9)
        history["wind_speed"], history["wind_direction"] = 0., np.nan
        self.assertTrue(origin_context(history, columns).loc[24].eq(0.).all())
        history["wind_speed"] = 5.
        self.assertTrue(origin_context(history, columns).loc[24].isna().all())
        history.loc[23, ["precip", "precip_trace"]] = np.nan
        self.assertTrue(origin_context(history, WEATHER_GROUPS["precipitation"]).loc[24].isna().all())

    def test_direct_examples_use_pre_origin_context_and_historical_labels(self):
        inputs, labels, metadata = make_direct_training_data(self.history, self.direct, 4)
        positions = (metadata["origin_ds"] - self.history["ds"].iloc[0]).dt.total_seconds() / 3600
        np.testing.assert_array_equal(inputs["temp_last"], positions - 1)
        np.testing.assert_array_equal(labels, inputs["temp_last"] + inputs["forecast_hour"])
        self.assertTrue(metadata["target_ds"].lt(FORECAST_START).all())
        self.assertTrue(metadata["target_ds"].ge(metadata["origin_ds"]).all())
        origins = metadata["origin_ds"].drop_duplicates()
        self.assertTrue(origins.diff().dropna().eq(pd.Timedelta(days=7)).all())
        self.assertEqual(inputs["forecast_hour"].min(), 1)
        self.assertEqual(inputs["forecast_hour"].max(), 336)
        live = make_direct_future_data(self.history, self.future, self.direct)
        self.assertEqual(live["temp_last"].nunique(), 1)
        self.assertEqual(live.iloc[0]["temp_last"], self.history.iloc[-1]["y"])

    def test_recursive_forecasts_feed_predictions_into_lags(self):
        calls = []

        class SpyEstimator:
            def fit(inner, inputs, labels):
                pass

            def predict(inner, inputs):
                calls.append(inputs.iloc[0].copy())
                return np.array([inputs.iloc[0]["lag_1"] + 1])

        with patch("src.gradient_boosting_model.build_model", return_value=SpyEstimator()):
            result = predict_gradient_boosting(self.history, self.future,
                                              feature_columns=CALENDAR_FEATURES + LAG_FEATURES)
        expected = np.arange(len(self.history), len(self.history) + 336)
        np.testing.assert_array_equal(result["yhat"], expected)
        self.assertEqual(calls[24]["lag_24"], result.loc[0, "yhat"])
        self.assertAlmostEqual(calls[24]["rolling_mean_24"], result.loc[:23, "yhat"].mean())

    def test_real_estimators_predict_every_hour_with_missing_inputs(self):
        self.history["y"] = 60 + 8 * np.sin(2 * np.pi * self.history["ds"].dt.hour / 24)
        self.history.loc[20, "y"] = np.nan
        self.history.loc[self.history.index[-3:], ["y", "dewpoint"]] = np.nan
        self.history.loc[20, "wind_direction"] = np.nan
        params = {"max_iter": 3, "min_samples_leaf": 5, "random_state": 520}
        all_weather = [column for group in WEATHER_GROUPS.values() for column in group]
        # Bound test resources without changing production fitting behavior.
        with threadpool_limits(limits=2):
            for columns in (None, CALENDAR_FEATURES + LAG_FEATURES, self.direct + all_weather):
                with self.subTest(columns=columns):
                    result = get_predictor("gradient_boosting", params, columns)(self.history, self.future)
                    self.assertEqual(len(result), 336)
                    self.assertTrue(np.isfinite(result["yhat"]).all())
                    self.assertEqual(str(result["ds"].dt.tz), "UTC")
            first = predict_gradient_boosting(self.history, self.future, params)
            second = predict_gradient_boosting(self.history, self.future, params)
        pd.testing.assert_frame_equal(first, second)

    def test_saved_configurations_roundtrip_and_reject_stale_feature_metadata(self):
        direct_weather = self.direct + WEATHER_GROUPS["wind_speed"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            for columns in (None, CALENDAR_FEATURES + LAG_FEATURES, direct_weather):
                with self.subTest(columns=columns):
                    selected = dict(model_config("gradient_boosting", {"max_iter": 3}, columns),
                                    lookback_years=3)
                    path.write_text(json.dumps(selected))
                    self.assertEqual(load_selection(path), selected)
                    self.assertEqual(selected["feature_columns"], selected["features"]["input_columns"])
                    if columns == direct_weather:
                        self.assertEqual(selected["features"]["historical_input_columns"],
                                         ["ds", "y", "wind_speed"])
                    selected["features"]["input_columns"] = ["bogus"]
                    path.write_text(json.dumps(selected))
                    with self.assertRaisesRegex(ValueError, "feature metadata"):
                        load_selection(path)


if __name__ == "__main__":
    unittest.main()
