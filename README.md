# 🏭 Supply Chain Demand Forecasting and Inventory Optimization

This project focuses on **demand forecasting** using machine learning and time series models, and incorporates **inventory optimization** techniques such as **Economic Order Quantity (EOQ)** and **Safety Stock** calculations to improve supply chain performance.

## 🚀 Project Overview

Efficient demand forecasting and inventory planning are vital for reducing costs and avoiding stockouts in supply chains. This project presents an end-to-end solution that:

- Uses historical sales data to forecast demand.
- Applies ARIMA, SARIMA, Prophet, and XGBoost for prediction.
- Implements inventory control with EOQ and safety stock methods.
- Provides visualizations and insights for decision-making.

## 📊 Key Features

- 📈 Time Series Forecasting using ARIMA/SARIMA/Prophet/XGBoost
- 📦 Inventory Management:
  - Economic Order Quantity (EOQ)
  - Safety Stock Calculation
- 📉 Forecast Evaluation: MAE, RMSE, MAPE
- 📊 Visual dashboards using Matplotlib/Seaborn

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Statsmodels (ARIMA, SARIMA)
- Facebook Prophet
- XGBoost
- Matplotlib, Seaborn


## 📦 Inventory Optimization Methods

- **EOQ Formula:**
  \[
  EOQ = \sqrt{\frac{{2DS}}{H}}
  \]
  - *D* = Annual demand
  - *S* = Ordering cost per order
  - *H* = Holding cost per unit per year

- **Safety Stock:**
  \[
  SS = Z \cdot \sigma_L
  \]
  - *Z* = Z-score (based on desired service level)
  - *σₗ* = Standard deviation during lead time

## 📈 Forecasting Models

- **ARIMA/SARIMA:** Traditional time series models for univariate forecasting.
- **Prophet:** Robust to missing data and trend changes.
- **XGBoost:** Gradient boosting-based model using time features and lag variables.

## 📌 Future Work

- 📊 Streamlit dashboard for real-time visualization
- 📦 Integrate MLOps with Docker & Airflow for automation
- 🔁 Incorporate more external features (e.g., promotions, seasonality)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

