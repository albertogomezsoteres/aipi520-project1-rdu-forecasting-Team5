"""Checks for the linear regression adapter used by the shared workflow."""
import unittest

import numpy as np
import pandas as pd

from src.backtesting import make_fold
from src.config import FORECAST_START, VALIDATION_WINDOWS
from src.linear_regression_model import predict_linear_regression, resolve_linear_regression_params
from src.models import model_config


class LinearRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small synthetic history: 4 years with a daily and an annual cycle
        times = pd.date_range(FORECAST_START - pd.DateOffset(years=4), FORECAST_START, freq="h", inclusive="left")
        y = 60 + 10 * np.sin(2 * np.pi * times.hour / 24) + 15 * np.sin(2 * np.pi * times.dayofyear / 365.25)
        cls.hourly = pd.DataFrame({"ds": times, "y": y})

    def test_predicts_every_hour_without_future_labels(self):
        history, future, _ = make_fold(self.hourly, VALIDATION_WINDOWS[-1])
        history.loc[history.index[-3:], "y"] = np.nan  # last hours missing, like before the real forecast
        predictions = predict_linear_regression(history, future)
        self.assertEqual(predictions["ds"].tolist(), future["ds"].tolist())
        self.assertTrue(np.isfinite(predictions["yhat"]).all())
        with self.assertRaisesRegex(ValueError, "actual temperatures"):
            predict_linear_regression(history, future.assign(y=0))

    def test_params_are_checked_and_described(self):
        self.assertEqual(resolve_linear_regression_params()["regularization"], "lasso")
        for bad in ({"unknown": 1}, {"k_daily": 0}, {"anomaly": "big"}, {"trend": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_config("linear_regression", bad)
        with self.assertRaises(ValueError):
            model_config("linear_regression", None, ["hour_sin"])
        config = model_config("linear_regression", {"anomaly": "none"})
        self.assertFalse(any("stage 2" in part for part in config["features"]["components"]))


if __name__ == "__main__":
    unittest.main()
