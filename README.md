# Retail Demand Forecasting and Stock Optimisation for a Multi-Store Retailer

## 1. Overview

This project develops a data-driven forecasting approach to improve weekly sales predictions 
for a multi-store retailer. Using historical store-level sales data, the goal is to replace 
manual spreadsheet forecasts with an accurate, automated model that supports smarter stock 
management and reduces operational inefficiencies.

---

## 2. Problem Statement

The retailer experienced frequent stockouts in high-demand weeks and excess stock on slow-moving 
lines. Planners relied on manual, spreadsheet-based forecasts using simple averages that did not 
capture short-term patterns, volatility or seasonal effects.

This resulted in:

- Lost sales during peak periods  
- High markdown costs  
- Inefficient use of warehouse space  
- Poor replenishment planning  

A more robust, data-driven forecasting method was required.

---

## 3. Business Context

- Multi-store retailer operating weekly replenishment cycles  
- Limited warehouse capacity and rising inventory holding costs  
- Increasing pressure to improve availability without increasing total stock  
- Decision makers required reliable forecasts and clear identification of high-risk stores  

### Stakeholders

- Head of Merchandising  
- Supply Chain Planning Team  
- Store Operations Managers  

---

## 4. Business Objectives

1. Improve weekly sales forecast accuracy at store level.
2. Identify stores with volatile or irregular demand that need special handling.
3. Provide visibility of store-level risk: over forecasting, under forecasting and volatility.
4. Quantify potential reductions in stockouts and excess inventory enabled by better forecasts.

---

## 5. Data

### Source
- Historical weekly store-level sales data (similar to the Walmart Sales Forecasting dataset).
- Includes external variables such as fuel price, temperature, CPI and holiday flags.

### Key Features Used in Modelling
- Store ID  
- Weekly sales  
- Calendar features (date, month, week of year, year)  
- Seasonality indicators (is month start/end, day of week)  
- Lag features (1, 2 and 4 weeks)  
- Rolling windows (4-week mean and standard deviation)  
- External factors: holiday flag, temperature, fuel price, CPI, unemployment  

All feature engineering was implemented in Python using a reproducible notebook pipeline.

---

## 6. Approach

### 6.1 Data Preparation

- Cleaned raw CSV data and validated column types.  
- Standardised date formatting and sorted data by store and time.  
- Engineered lag features and 4-week rolling statistics to capture demand patterns.  
- Created a “model-ready” dataset by removing rows without sufficient history for lag features.

### 6.2 Exploratory Analysis

- Examined sales trends, seasonality patterns and store-level variability.  
- Identified stable stores vs highly volatile stores.  
- Computed store-level metrics such as weekly volatility and historical MAPE.  
- Analysed distribution of forecasting errors to understand model reliability.

### 6.3 Forecast Modelling

Forecasts were generated using:

- **Baseline model:** naive lag-1 prediction  
- **Machine learning model:** Random Forest Regressor using engineered features  

Models were evaluated using a time-based train-test split and the following metrics:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  

### 6.4 Stock Optimisation Logic

Based on forecast behaviour and store-level error analysis:

- Identified high-risk stores where the model consistently over forecasts (inventory inflation risk).
- Identified volatile stores requiring higher safety stock.
- Provided store segmentation to guide planners in applying differentiated ordering rules.

---

## 7. Tools and Techniques Used

- **Python:** data cleaning, feature engineering, forecasting models  
- **Pandas, NumPy:** preprocessing and transformations  
- **scikit-learn:** Random Forest model and evaluation  
- **Matplotlib / Seaborn:** visualisation  
- **Excel:** exploratory checks, planner-friendly summary tables  
- **Power BI:** (optional) dashboard for forecast accuracy and store risk  

---

## 8. Key Insights

(Real numbers included where available.)

- The Random Forest model improved MAPE from **4.94 percent (baseline)** to **3.52 percent**, 
  representing a **29 percent improvement** in forecast accuracy.
- The 4-week rolling mean was the dominant predictor (importance: **0.94**), indicating weekly 
  sales are highly stable and strongly driven by short-term patterns.
- A small group of stores (e.g. **Stores 28, 14, 23, 17**) showed both high MAPE and strong 
  positive bias, marking them as high operational risk due to consistent over forecasting.
- Most stores achieved MAPE between **2 percent and 4 percent**, indicating strong model reliability.
- Error distribution was heavily right-skewed, with the vast majority of predictions within 
  **0–5 percent error**, and only occasional spikes due to volatile store behaviour.

---

## 9. Business Recommendations

### 1. Deploy the forecasting model across stable stores
Most stores achieve highly accurate forecasts and can safely rely on automated predictions.

### 2. Apply targeted oversight to high-variance stores
Stores with MAPE above 5 percent require:
- Manual review  
- Higher safety stock  
- Localised adjustments  

### 3. Correct systematic over forecasting bias
For stores showing consistent upward bias:
- Apply simple post-model correction factors  
- Avoid unnecessary overstocking  

### 4. Enhance the model with store-specific or external features
Potential improvements:
- Promotion indicators  
- Holiday-specific variables  
- Local event data  
- Store clustering to personalise models  

### 5. Integrate forecasts into a dashboard
Use Power BI or similar tooling to:
- Monitor forecast accuracy  
- Identify at-risk stores  
- Support weekly replenishment decision making  

---

## 10. Impact

(Values to be finalised after end-to-end evaluation.)

- Forecast accuracy improved by **~29 percent** vs the manual baseline.
- Improved inventory visibility across all stores.
- Potential to reduce excess stock by **meaningful operational margins**, while improving availability.
- Enabled planners to adopt a consistent, data-driven replenishment process.

---

## 11. Data source:

https://www.kaggle.com/datasets/yasserh/walmart-dataset [Walmart.csv(363.73 kB)]

## 12. Repository Structure

```markdown
project/
│ README.md
│ requirements.txt
│
├── data/ (ignored)
│ ├── raw/ (ignored)
│ ├── processed/ (ignored)
│ └── clean/ (ignored)
│
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_eda.ipynb
│ ├── 04_modelling.ipynb
│ ├── 05_evaluation_reporting.ipynb
│ └── 06_business_recommendations.ipynb
│
├── models/ (ignored)
└── visuals/ (ignored)
```