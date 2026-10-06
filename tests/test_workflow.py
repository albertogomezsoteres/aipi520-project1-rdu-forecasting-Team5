"""Check candidate tuning, frozen selection, provenance, and full orchestration."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtesting import load_selection
from src.config import FORECAST_START, SUBMISSION_WINDOW, VALIDATION_WINDOWS
from src.models import model_config
from src.workflow import expand_candidates, file_sha256, run_workflow, tune_models


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = pd.date_range(pd.Timestamp("2015-01-01", tz="UTC"), FORECAST_START, freq="h", inclusive="left")
        cls.hourly = pd.DataFrame({"ds": times, "y": 60. + times.hour + times.month / 10})

    def test_default_grid_includes_original_prophet_and_resolves_features(self):
        config = Path(__file__).resolve().parents[1] / "configs/experiment.json"
        candidates, lookbacks = expand_candidates(json.loads(config.read_text()))
        self.assertEqual(len(candidates), 8)  # 2 baselines + 4 Prophet + linear regression + boosting
        self.assertEqual(lookbacks, [None, 5, 3])
        prophet = [candidate for candidate in candidates.values() if candidate["model"] == "prophet"]
        self.assertTrue(any(candidate["params"]["daily_fourier_order"] == 16 and
                            candidate["params"]["changepoint_prior_scale"] == .005 for candidate in prophet))
        self.assertTrue(all(candidate["features"]["input_columns"] == ["ds"] for candidate in prophet))
        self.assertTrue(all(candidate["params"]["yearly_fourier_order"] == 20 for candidate in prophet))

    def test_invalid_candidates_fail_before_fitting(self):
        invalid = [
            {"lookbacks": [True], "models": [{"name": "month_hour"}]},
            {"lookbacks": [None], "models": [{"name": "unregistered_model"}]},
            {"lookbacks": [None], "models": [{"name": "prophet", "param_grid": {"unknown": [1]}}]},
            {"lookbacks": [None], "models": [{"name": "prophet", "param_grid": {"daily_fourier_order": [0]}}]},
            {"lookbacks": [None], "models": [{"name": "prophet", "feature_sets": [["y"]]}]},
            {"lookbacks": [None], "models": [{"name": "month_hour", "param_grid": {"alpha": [1]}}]},
        ]
        for experiment in invalid:
            with self.subTest(experiment=experiment), self.assertRaises(ValueError):
                expand_candidates(experiment)

    def test_duplicate_grid_values_are_not_fit_twice(self):
        candidates, _ = expand_candidates({"lookbacks": [None], "models": [
            {"name": "prophet", "param_grid": {"daily_fourier_order": [6, 6]}}
        ]})
        self.assertEqual(len(candidates), 1)

    def test_parameter_candidates_reach_predictor_and_rank_across_same_windows(self):
        experiment = {"lookbacks": [None, 3], "models": [{
            "name": "prophet", "param_grid": {"changepoint_prior_scale": [.005, .05]}
        }]}
        calls = []
        def predict(history, future, params):
            self.assertNotIn("y", future)
            self.assertLess(history["ds"].max(), future["ds"].min())
            calls.append(params.copy())
            offset = 0 if params["changepoint_prior_scale"] == .005 else 5
            return future[["ds"]].assign(yhat=60. + future["ds"].dt.hour + future["ds"].dt.month / 10 + offset)
        with patch("src.prophet_model.predict_prophet", side_effect=predict), contextlib.redirect_stdout(io.StringIO()):
            metrics, predictions, summary, selected = tune_models(self.hourly, experiment, VALIDATION_WINDOWS[:2])
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(metrics), 8)
        self.assertEqual(len(predictions), 8 * 336)
        self.assertEqual(summary["windows"].tolist(), [2] * 4)
        self.assertEqual(selected["params"]["changepoint_prior_scale"], .005)
        self.assertAlmostEqual(selected["mean_validation_mae"], 0)
        self.assertEqual(metrics["model"].unique().tolist(), ["prophet"])
        self.assertEqual(metrics["candidate_id"].nunique(), 2)

    def test_full_workflow_keeps_selection_despite_bad_audit_and_refits_to_cutoff(self):
        experiment = {"lookbacks": [None], "models": [{
            "name": "prophet", "param_grid": {"changepoint_prior_scale": [.005, .05]}
        }]}
        calls = []
        def predict(history, future, params):
            calls.append((history["ds"].max(), future["ds"].min(), params.copy()))
            self.assertNotIn("y", future)
            offset = 0 if params["changepoint_prior_scale"] == .005 else 5
            # The selected candidate intentionally fails the audit comparison.
            if future["ds"].min().month == 9 and future["ds"].min().day == 3:
                offset = 100
            return future[["ds"]].assign(yhat=60. + future["ds"].dt.hour + future["ds"].dt.month / 10 + offset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.csv"
            pd.DataFrame({"station": "RDU", "valid": self.hourly["ds"] + pd.Timedelta(minutes=51),
                          "tmpf": self.hourly["y"]}).to_csv(raw, index=False)
            raw_hash = file_sha256(raw)
            config = root / "config.json"
            config.write_text(json.dumps(experiment))
            output = root / "run"
            with patch.dict("sys.modules", {"prophet": types.ModuleType("prophet")}), \
                    patch("src.prophet_model.predict_prophet", side_effect=predict), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = run_workflow(config, raw, output)
            self.assertEqual(len(calls), 12)  # 10 validation fits + selected audit + final refit
            self.assertEqual(report["selection"]["params"]["changepoint_prior_scale"], .005)
            self.assertAlmostEqual(report["selected_audit_mae"], 100)
            self.assertFalse(report["audit_used_for_selection"])
            self.assertEqual(calls[-1][0], FORECAST_START - pd.Timedelta(hours=1))
            self.assertEqual(calls[-1][1], FORECAST_START)
            self.assertEqual(calls[-1][2], report["selection"]["params"])
            self.assertEqual(file_sha256(raw), raw_hash)
            self.assertEqual(report["raw_input_sha256"], raw_hash)
            selected = load_selection(output / "selected_config.json")
            self.assertEqual(selected["features"], report["forecast"]["selection"]["features"])
            forecast = pd.read_csv(output / "final_predictions.csv")
            self.assertEqual(forecast.columns.tolist(), ["ds", "yhat"])
            self.assertTrue(pd.DatetimeIndex(pd.to_datetime(forecast["ds"], utc=True)).equals(SUBMISSION_WINDOW.timestamps()))
            self.assertTrue(np.isfinite(forecast["yhat"]).all())
            self.assertEqual(json.loads((output / "run_summary.json").read_text())["status"], "complete")
            self.assertEqual(report["validation_fits"], 10)
            with patch.dict("sys.modules", {"prophet": types.ModuleType("prophet")}), self.assertRaises(FileExistsError):
                run_workflow(config, raw, output)

    def test_selection_rejects_stale_feature_metadata(self):
        selected = dict(model_config("prophet", {"daily_fourier_order": 6}), lookback_years=3)
        selected["params"]["daily_fourier_order"] = 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(selected))
            with self.assertRaisesRegex(ValueError, "feature metadata"):
                load_selection(path)


if __name__ == "__main__":
    unittest.main()
