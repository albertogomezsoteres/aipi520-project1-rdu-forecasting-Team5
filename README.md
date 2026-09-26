# AIPI 520 | Project 1: RDU Temperature Forecasting

Hourly temperature forecasting at Raleigh-Durham International Airport (RDU) using historical data and machine learning.

## 1. Project Overview

The objective of this project is to predict the hourly temperature recorded at RDU Airport from **September 17 through September 30, 2026**.

All predictions must use information available before September 17.

The project requires at least two forecasting models:
- **Model 1:** Linear Regression
- **Model 2:** To be determined (TBD)

## 2. Team Members

- Alberto Gomez
- Yan Liu
- Nicholas Wang

## 3. Dataset

**Data source:** TBD

**Target:** Hourly temperature at RDU Airport.

The dataset and final input features will be selected following an initial review of available historical data.

Only information available before the forecasting period may be used to generate predictions.

## 4. Modeling Approach

### Model 1: Linear Regression

Linear Regression will be included as required by the project.

The specific preprocessing and feature engineering techniques will be determined during development.

### Model 2: TBD

The second model has not yet been selected.

Potential options include:
- Random Forest
- Gradient Boosting

The final choice will be made after exploring the data and discussing the modeling approach.

### Potential Features

Depending on data availability and exploratory analysis, potential features include:

- Time-based features
- Seasonal and cyclical features
- Historical temperature and lag features
- Additional historical weather variables

The final feature set is yet to be determined.

## 5. Potential Evaluation Approach & Metrics

The evaluation strategy and metrics have not yet been finalized.

Potential evaluation metrics include:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

Time-series validation will be considered to assess forecasting performance while avoiding future data leakage.

The final evaluation approach will be documented after team discussion.

## 6. Repository Structure

    aipi520-project1-rdu-forecasting/
    |
    |-- README.md
    |-- requirements.txt
    |-- .gitignore
    |
    |-- data/
    |   |-- raw/
    |   |-- processed/
    |
    |-- notebooks/
    |
    |-- src/
    |
    |-- outputs/
    |   |-- figures/
    |   |-- models/
    |
    |-- reports/

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Original datasets |
| `data/processed/` | Cleaned and processed datasets |
| `notebooks/` | Exploratory analysis and experimentation |
| `src/` | Reusable Python functions and executable scripts |
| `outputs/figures/` | Visualizations and evaluation plots |
| `outputs/models/` | Saved trained models |
| `reports/` | Final presentation and written report |

## 7. Reproducibility

The project will aim to provide a reproducible workflow using reusable Python functions and scripts.

Jupyter notebooks will be used primarily for exploratory analysis and experimentation.

Dependencies will be documented in `requirements.txt`.

Instructions for data preparation, model training, evaluation, and forecasting will be added as development progresses.

## 8. Collaboration and Git Workflow

We plan to use a simplified GitFlow strategy:

- **`main`:** Stable, reviewed project version.
- **`dev`:** Integration branch for ongoing development.
- **`feature/*`:** Individual development branches.

Development changes will be integrated into `dev` through pull requests. Reviewed and finalized changes will then be merged into `main`.

## 9. Deliverables

The final submission will include:

1. **Code repository:** Project code and supporting materials.
2. **Presentation:** Inputs, data pipeline, features, models, evaluation approach, and performance.
3. **Written report:** A 2–4 page description of the project methodology and application of course concepts.

**Submission deadline:** October 7, 2026.

---

**Course:** AIPI 520 - Modeling Process & Algorithms