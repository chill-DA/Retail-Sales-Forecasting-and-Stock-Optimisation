# Retail Demand Forecasting and Stock Optimisation for a Multi-Store Retailer

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy&logoColor=white">
  <img alt="Scikit-Learn" src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualisation-orange?logo=python&logoColor=white">
  <img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal?logo=python&logoColor=white">
  <img alt="Pytest" src="https://img.shields.io/badge/Pytest-Unit%20Tests-green?logo=pytest&logoColor=white">
</p>

## 1. Overview

This project develops a data-driven forecasting approach to improve weekly sales predictions for a multi-store retailer. Using historical store-level sales data, it compares a simple planning baseline against a machine learning model and translates the results into stock-risk and replenishment recommendations.

The project is now split into two layers:

1. **Notebook analysis** for EDA, modelling, evaluation and business storytelling.
2. **Reusable Python modules** for feature engineering, metrics, model training and reproducible execution.

This makes the repository stronger as a portfolio project because it shows both analytical thinking and production-adjacent project structure.

---

## 2. Problem Statement

The retailer experienced frequent stockouts in high-demand weeks and excess stock on slow-moving lines. Planners relied on manual, spreadsheet-based forecasts using simple averages that did not capture short-term patterns, volatility or seasonal effects.

This resulted in:

- Lost sales during peak periods
- High markdown costs
- Inefficient use of warehouse space
- Poor replenishment planning

A more robust, data-driven forecasting method was required.

---

## 3. Business Objectives

1. Improve weekly sales forecast accuracy at store level.
2. Identify stores with volatile or irregular demand that need special handling.
3. Provide visibility of store-level risk: over-forecasting, under-forecasting and volatility.
4. Translate model output into replenishment and safety-stock recommendations.

---

## 4. Data

### Source

- Historical weekly store-level sales data similar to the Walmart Sales Forecasting dataset.
- Includes external variables such as fuel price, temperature, CPI, unemployment and holiday flags.

Original data source:

- `Walmart.csv`
- Kaggle dataset: `yasserh/walmart-dataset`
- Licence: CC0 Public Domain

The full raw dataset is not committed to the repository. A small synthetic/sample file is included so the code can be run immediately:

```bash
data/sample/walmart_sample.csv
```

### Key Features Used in Modelling

- Store ID
- Weekly sales
- Calendar features: year, month, week of year, day of week
- Seasonality indicators: month start/end
- Lag features: 1, 2 and 4 weeks
- Rolling features: 4-week mean and standard deviation
- External variables: holiday flag, temperature, fuel price, CPI, unemployment

Rolling features are calculated from prior sales values only, so the current week's target does not leak into the predictor set.

---

## 5. Approach

### 5.1 Data Preparation

- Cleaned raw CSV data and validated column types.
- Standardised date formatting and sorted data by store and time.
- Engineered calendar, lag and rolling-window features.
- Created a model-ready dataset by removing rows without enough demand history.

### 5.2 Exploratory Analysis

- Examined sales trends, seasonality patterns and store-level variability.
- Identified stable stores versus highly volatile stores.
- Analysed model errors to understand operational risk.

### 5.3 Forecast Modelling

Forecasts were generated using:

- **Baseline model:** naive lag-1 forecast, equivalent to using last week's sales.
- **Machine learning model:** Random Forest Regressor using engineered features.

Models were evaluated using a time-based train-test split and the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

## 6. Key Results

From the notebook analysis:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Baseline lag-1 | 50,730.91 | 75,702.75 | 4.94% |
| Random Forest | 36,792.57 | 55,665.33 | 3.52% |

The Random Forest model improved MAPE from **4.94%** to **3.52%**, representing an approximate **29% reduction in percentage forecast error** compared with the naive baseline.

---

## 7. Business Recommendations

### 1. Deploy automated forecasting for stable stores

Stores with low forecast error can safely rely on automated predictions for weekly replenishment planning.

### 2. Apply targeted oversight to volatile stores

Stores with high MAPE or unstable demand should receive manual review, local context checks and potentially higher safety stock.

### 3. Correct systematic over-forecasting

Stores with persistent positive forecast bias should receive simple post-model correction factors to avoid inflated inventory.

### 4. Add richer operational features

Potential next features include:

- Promotions
- Local events
- Stockout flags
- Lead times
- Store clusters
- Product-level or department-level demand patterns

### 5. Convert outputs into a dashboard

A Power BI dashboard could monitor forecast accuracy, bias, volatile stores and replenishment risk. No `.pbix` file is included in this repository; this is listed as a future integration step.

---

## 8. Repository Structure

```text
project/
|
├── README.md
├── requirements.txt
├── run_forecast_pipeline.py
|
├── data/
│   └── sample/
│       └── walmart_sample.csv
|
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_modelling.ipynb
│   ├── 05_evaluation_reporting.ipynb
│   └── 06_business_recommendations.ipynb
|
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── metrics.py
│   └── model.py
|
├── tests/
│   ├── test_features.py
│   └── test_metrics.py
|
├── reports/       # generated, ignored
├── models/        # generated, ignored
└── visuals/       # generated, ignored
```

---

## 9. How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the tests

```bash
pytest
```

### 4. Run the reproducible sample pipeline

```bash
python run_forecast_pipeline.py
```

This uses:

```bash
data/sample/walmart_sample.csv
```

and creates generated outputs in:

```bash
models/
reports/
```

### 5. Run against the full Kaggle data

Download the Kaggle CSV and run:

```bash
python run_forecast_pipeline.py --input data/raw/Walmart.csv
```

The raw data folder is ignored by Git to avoid committing external datasets.

---

## 10. Notebook Workflow

The notebooks preserve the original analysis flow:

1. `01_data_cleaning.ipynb` - raw data checks, cleaning and validation
2. `02_feature_engineering.ipynb` - lag, rolling and calendar features
3. `03_eda.ipynb` - trends, seasonality and store-level behaviour
4. `04_modelling.ipynb` - baseline and Random Forest model comparison
5. `05_evaluation_reporting.ipynb` - forecast interpretation and store-level risk
6. `06_business_recommendations.ipynb` - operational recommendations

The reusable `src/` modules make the same core logic easier to test, maintain and rerun outside the notebooks.

---

## 11. Portfolio Summary

This project demonstrates:

- Forecasting model evaluation against a planning baseline
- Time-aware train/test splitting
- Feature engineering for retail demand
- Forecast accuracy measurement using MAE, RMSE and MAPE
- Store-level operational risk interpretation
- Practical stock-optimisation recommendations
- Reproducible code structure with tests and a runnable sample pipeline
