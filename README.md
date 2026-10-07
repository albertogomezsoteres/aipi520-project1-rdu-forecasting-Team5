# AIPI 520 Project 1 RDU Temperature Forecasting

Forecast the hourly temperature measured at Raleigh-Durham International Airport
(RDU) for September 17-30, 2026, using only information available before the
forecast period. The project compares forecasting models using historical
14-day forecasts and produces a reproducible set of 336 final predictions.

**Team:** Alberto Gomez, Yan Liu, and Nicholas Wang.  

## Models and deliverables
Linear regression, gradient boosting, and Prophet are the models chosen for comparison.
Two simple benchmarks are used for comparison: month/hour climatology and repetition of the
previous day's temperature profile.

The shared command-line workflow supports all three models and both benchmarks.

## Data and forecast conventions

**Source:** Iowa Environmental Mesonet NC ASOS observations for station RDU.  
**Input:** `data/raw/RDU_starting2015_UTC.csv`.  
**Target:** Temperature in degrees Fahrenheit (`tmpf` in the raw data, `y` after
cleaning). Predictions are named `yhat`.

All stored timestamps, joins, split boundaries, and shared calendar features use
UTC. The submission starts at midnight `America/New_York` on September 17, 2026:

| Boundary | UTC timestamp |
|---|---|
| First hourly label | September 17, 2026, 04:00 |
| Last hourly label | October 1, 2026, 03:00 |
| Exclusive end | October 1, 2026, 04:00 |

Each hourly label represents the selected observation within that hour, normally
at `:51`. For example, `ds=2026-09-17 04:00 UTC` predicts the 04:51 UTC report,
which is September 17, 00:51 Eastern. There are 336 hourly labels in total.

Cleaning selects the valid temperature nearest `:51` within each hour, preferring
the later report in a tie. Conflicting temperatures at the same timestamp raise
an error. The complete hourly grid is preserved, including missing temperatures
as `NaN`. Missing evaluation labels are excluded from scoring and are not filled.

## Repository structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original weather observations |
| `data/processed/` | Canonical hourly data, cleaning report, and split CSVs |
| `notebooks/` | Data inspection, exploratory analysis, and model experiments |
| `src/config.py` | Shared paths, forecast conventions, and evaluation windows |
| `src/data_pipeline.py` | Load, clean, validate, and export hourly data |
| `src/features.py` | Shared UTC calendar and cyclic features |
| `src/baselines.py` | Training-only benchmark forecasts |
| `src/evaluation.py` | Prediction checks, timestamp alignment, and metrics |
| `src/backtesting.py` | Historical model comparison, selection, and audit |
| `src/models.py` | Model registry, resolved parameters, and used-feature descriptions |
| `src/workflow.py` | One-command data preparation, tuning, selection, audit, and forecast |
| `src/prophet_model.py` | Prophet model and forecasting adapter |
| `src/linear_regression_model.py` | Two-stage linear regression and forecasting adapter |
| `src/gradient_boosting_model.py` | Gradient boosting and calendar/recursive/direct forecasting adapter |
| `src/gradient_boosting_features.py` | Past-temperature lags and fixed forecast-origin context |
| `src/forecast.py` | Selected-model refitting and final predictions |
| `configs/` | Small, editable experiment search specifications |
| `tests/` | Shared data and forecasting contract checks |
| `outputs/` | Evaluation results, predictions, figures, and model artifacts |
| `reports/` | Presentation and written report materials |

## Setup

Use Python 3.11 or 3.12. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.
Prophet is needed for Prophet model runs; the shared tests and baseline-only
workflows can run without it.

## Run the entire workflow

After setup, run from the repository root:

```bash
python -m src.workflow
```

The command prepares the checked-in raw observations, tunes configured candidates
on all five validation windows, selects the lowest equal-window mean MAE, audits
that locked selection alongside the baselines, then refits it through the final
cutoff and generates all 336 predictions. It does not download new observations.

The default `configs/experiment.json` compares both baselines, four Prophet
parameter combinations, one linear regression configuration, and one gradient
boosting configuration, each with all available, five-year, and three-year
training histories. The Prophet search varies `changepoint_prior_scale` between
0.005 and 0.05 and `daily_fourier_order` between 6 and 16. Other parameters retain
the existing adapter defaults; the teammate's original configuration is included.
This is 24 candidate/history combinations and 120 validation fits, including 60
Prophet fits. Runtime depends on your computer and the training histories.

For a faster full workflow using just the baselines:

```bash
python -m src.workflow --config configs/baselines.json
```

Each run creates a new folder under `outputs/runs/<UTC timestamp>/`. To choose
your own new folder or raw input:

```bash
python -m src.workflow --config configs/experiment.json \
  --input data/raw/RDU_starting2015_UTC.csv --output-dir outputs/runs/experiment_01
```

An existing output folder raises an error so results from different runs cannot
be mixed. Processed data is written to that run's `data_processed/` folder; the
standalone pipeline still writes to `data/processed/`.

| Run output | Contents |
|---|---|
| `experiment.json`, `candidate_configs.json` | Search specification and every resolved candidate/history configuration |
| `data_processed/` | Canonical hourly data, cleaning report, and legacy splits |
| `validation_metrics.csv`, `validation_predictions.csv`, `validation_summary.csv` | Per-window scores, aligned predictions, and candidate rankings |
| `selected_config.json` | Validation-selected model, parameters, actual feature description, and lookback |
| `audit_metrics.csv`, `audit_predictions.csv`, `audit_summary.csv` | Selected model and baseline results; do not change selection |
| `final_predictions.csv`, `final_predictions.json` | UTC `ds,yhat` predictions and forecast metadata |
| `run_summary.json` | Selected settings, validation/audit MAE, input/source hashes, package versions, and output paths |

The terminal reports the selected model, parameters, features/components,
training history, validation MAE, audit MAE, and forecast location. A baseline
can win; selection is always limited to the configured candidates.

### Gradient boosting experiments

`notebooks/gradient_boosting_analysis.ipynb` follows the calendar, recursive-lag,
weather, parameter, and training-history experiments in order. It includes visible
results and loads one compact `notebooks/gradient_boosting_results.json` cache by
default. Set its `RUN_STAGES` list to refit selected experiments through the shared
tuning function; a complete replay is 450 validation fits. The recorded input hash
guards against silently mixing results from different raw datasets.

For the smaller repeatable development search (90 fits), without audit scoring
or a final forecast:

```bash
python -m src.workflow --config configs/gradient_boosting.json --validation-only
```

It compares 100/150 iterations and 7/15 leaves with combined calendar inputs and
all/five/three-year histories. The main experiment uses the recommended default:
learning rate 0.05, 150 iterations, 7 leaves, minimum leaf size 50, and L2 penalty 1.
The workflow still selects the training-history length from validation.

### Configure experiments

Each `models` entry has a `name`, optional `param_grid` mapping parameter names
to lists of values, and optional `feature_sets`. The runner evaluates the
Cartesian product of parameter values and feature sets across every `lookbacks`
value. `null` in `lookbacks` means all available history.

Prophet and the baselines use fixed input rules, so their feature sets contain
only `null`. Prophet consumes timestamps and creates its trend/seasonality
components; it does not consume the shared calendar columns as regressors.
The saved feature description reports those actual components and Fourier
orders. Changing those orders is part of the Prophet search.

Gradient boosting accepts explicit feature-column lists, validates their names,
and describes their actual use in saved configurations. `null` selects the seven
shared calendar/cyclic features. Adding `lag_1` and `lag_24` selects recursive
forecasting; adding `forecast_hour` and temperature context selects a direct
forecast from fixed pre-origin conditions. Weather feature sets require
`--include-weather`. Linear regression uses its own Fourier and anomaly features;
its feature-set entry remains `null`. Each model's settings are resolved in
`src/models.py` and passed to its shared forecasting adapter.

The audit period has already been examined in earlier experiments. Disclose
any use of its results when discussing the final evaluation. The automated
workflow keeps it out of selection and never scores submission-period targets.

## Prepare the hourly data

```bash
python -m src.data_pipeline
```

This writes the following files to `data/processed/`:

- `hourly.csv`: the canonical complete UTC timeline with `ds,y` columns.
- `cleaning_report.json`: cleaning counts, cutoff, and data coverage.
- `train.csv`, `val.csv`, and `test.csv`: chronological splits with shared features
  for the notebook workflows.

Use `python -m src.data_pipeline --include-weather` to retain available historical
dew point, wind, humidity, pressure, cloud, and precipitation columns in `hourly.csv`.
The default is temperature only; legacy split columns stay unchanged. Weather is
passed only in pre-origin history. Means require 18 of 24 observations, precipitation
totals require 24, and missing inputs remain NaN for histogram boosting to handle.

`hourly.csv` is generated by this command. Prepare it before running backtesting
or final forecasting. The data pipeline notebook also calls the shared preparation
functions and displays their results.

To use another input location or historical start hour:

```bash
python -m src.data_pipeline --input data/raw/RDU_starting2015_UTC.csv --start 2015-01-01 --output-dir data/processed
```

## Compare models using historical forecasts

Each evaluation simulates one forecast made at the beginning of a 14-day window.
Training data must precede that origin. The model predicts all 336 hours without
receiving measured temperatures from inside the window.

| Window in Eastern time | Purpose | Observed target hours |
|---|---|---:|
| September 17-30, 2022 | Validation | 336 |
| September 17-30, 2023 | Validation | 336 |
| September 17-30, 2024 | Validation | 336 |
| September 17-30, 2025 | Validation | 334 |
| August 20-September 2, 2026 | Validation | 336 |
| September 3-16, 2026 | Historical audit | 336 |

Compare the supported models using all available history and rolling five-year
and three-year histories:

```bash
python -m src.backtesting --models month_hour repeat_last_day prophet linear_regression gradient_boosting --lookbacks all 5 3
```

For a fast baseline-only comparison:

```bash
python -m src.backtesting --models month_hour repeat_last_day --lookbacks all
```

The runner saves the following files under `outputs/backtests/`:

| Output | Contents |
|---|---|
| `validation_metrics.csv` | Metrics and training coverage for every model/history/window |
| `validation_predictions.csv` | Aligned actuals and predictions with forecast hour/day |
| `validation_summary.csv` | Aggregate metrics and variation across validation windows |
| `candidate_configs.json` | Configurations evaluated |
| `selected_config.json` | Configuration with the lowest mean validation-window MAE |

Selection is limited to the candidates included in that run. A baseline-only
comparison selects a baseline configuration. Use the complete intended candidate
set before choosing the final project model.

## Evaluate the selected configuration

Audit a locked selection on September 3-16, 2026, alongside both benchmarks:

```bash
python -m src.backtesting --audit --selection outputs/backtests/selected_config.json
```

This writes `audit_metrics.csv`, `audit_predictions.csv`, and `audit_summary.csv`
without creating a new selection. This historical period has already been examined;
report any use of its results in subsequent tuning.

To evaluate the configured Prophet model directly with all pre-audit history:

```bash
python -m src.prophet_model
```

Its comparison and metrics are saved to
`outputs/predictions/prophet_test_predictions.csv` and a JSON sidecar.

## Generate final predictions

After selecting the final model configuration using validation:

```bash
python -m src.forecast --selection outputs/backtests/selected_config.json
```

The model refits using all observations before September 17, 2026, 04:00 UTC
within the selected training-history length. Previously designated validation and
audit observations may be included because they precede the final cutoff.

Outputs are `outputs/predictions/final_predictions.csv`, containing UTC `ds,yhat`
columns, and `final_predictions.json`, documenting the configuration, cutoff,
units, training coverage, and package versions. The CSV uses hourly labels and
must contain exactly 336 finite predictions. No submission-period labels are
loaded or scored.

## Evaluation metrics and benchmarks

The primary model-selection metric is **mean validation-window MAE**, with each
window weighted equally. Secondary metrics are RMSE, MASE, R-squared, and bias
(`prediction - actual`). Every requested hour must have a finite prediction;
only missing actual temperatures are excluded from scoring.

MASE divides forecast MAE by the mean absolute 24-elapsed-hour training difference
on the complete hourly grid. All candidates in a window use the same pre-origin
reference history for this scale. Evaluate the actual benchmark forecasts
separately when determining whether a model improves on them.

Month/hour climatology uses training averages for each UTC month/hour combination.
Repeat-last-day repeats the exact preceding 24-hour profile. Missing inputs in
that profile fall back to the training month/hour mean, then the training global
mean. Evaluation results record the number of fallback hours.

## Run checks

```bash
python -m unittest discover -s tests -v
```

The checks cover time boundaries, cleaning, missing targets, training histories,
forecast completeness, benchmark alignment, metrics, and model adapter behavior.
The Prophet adapter test uses a stand-in model; live Prophet fitting is checked
separately by running its forecasting workflow. Gradient boosting checks include
real-estimator forecasts, pre-origin feature timing, recursive feedback, missing
inputs, saved settings, and validation-only/full workflow behavior.
