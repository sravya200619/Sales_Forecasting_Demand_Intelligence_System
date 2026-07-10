# 📈 End-to-End Sales Forecasting & Demand Intelligence System

## 📌 Project Overview

This project is an end-to-end retail sales forecasting system developed using Machine Learning and Time Series Forecasting techniques.

The application predicts future product demand, detects sales anomalies, segments products based on demand patterns, and provides an interactive business dashboard using Streamlit.

---

## Features

- Executive Dashboard
- Time Series Analysis
- SARIMA Forecasting
- Facebook Prophet Forecasting
- XGBoost Forecasting
- Forecast Comparison
- Isolation Forest Anomaly Detection
- Z-Score Anomaly Detection
- Product Demand Segmentation
- K-Means Clustering
- Interactive Streamlit Dashboard

---

## Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- XGBoost
- Prophet
- Statsmodels
- Scikit-Learn

---

## Folder Structure

```
SalesForecasting_Sravya/

│── app.py
│── train.csv
│── requirements.txt
│── README.md
│
├── pages/
│      1_Executive_Dashboard.py
│      2_Forecast_Explorer.py
│      3_Anomaly_Report.py
│      4_Demand_Segmentation.py
│      5_Model_Comparison.py
│
├── models/
│      xgboost_model.pkl
│
├── charts/
│
└── analysis.ipynb
```

---

## Installation

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

## Dashboard Pages

### Executive Dashboard

- KPI Cards
- Sales Trends
- Category Analysis
- Region Analysis

### Forecast Explorer

- XGBoost Forecast
- Forecast Horizon
- Model Metrics

### Anomaly Report

- Isolation Forest
- Z-Score Detection

### Product Segmentation

- K-Means
- PCA Visualization
- Stocking Strategy

### Model Comparison

- SARIMA
- Prophet
- XGBoost
- Performance Metrics

---
Streamlit Deployed link
https://sravya200619-sales-forecasting-demand-intelligence-s-app-yknaso.streamlit.app

## Author

**Sravya Velaga**
