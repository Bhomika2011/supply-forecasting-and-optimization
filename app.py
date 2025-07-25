
import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="📦 Supply Chain Optimizer", layout="wide")

st.title("📦 Supply Chain Forecasting + Inventory Optimization")

# Sidebar
st.sidebar.title("Upload Your Data")
train_file = st.sidebar.file_uploader("Upload train.csv", type="csv")
test_file = st.sidebar.file_uploader("Upload test.csv", type="csv")
holiday_file = st.sidebar.file_uploader("Upload holidays_events.csv", type="csv")

# Helper Functions
def create_features(df):
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

def calc_eoq(demand, setup_cost, holding_cost, cv):
    return np.sqrt((2 * demand * setup_cost / holding_cost) * (1 + cv**2 / 2))

def calc_safety_stock(service_level, std_dev, lead_time):
    z = norm.ppf(service_level)
    return z * std_dev * np.sqrt(lead_time)

# Main Logic
if train_file and test_file and holiday_file:
    # Load data
    train = pd.read_csv(train_file, parse_dates=['date'])
    test = pd.read_csv(test_file, parse_dates=['date'])
    holidays = pd.read_csv(holiday_file, parse_dates=['date'])

    # Filter holidays
    holidays = holidays[(holidays['transferred'] == False) & (holidays['locale'] == 'National')]
    holiday_dates = holidays['date'].unique()

    # Add holiday feature
    train["is_holiday"] = train["date"].isin(holiday_dates).astype(int)
    test["is_holiday"] = test["date"].isin(holiday_dates).astype(int)

    # Feature engineering
    train = create_features(train)
    test = create_features(test)

    # Encode category
    le = LabelEncoder()
    train['family_enc'] = le.fit_transform(train['family'])
    test['family_enc'] = le.transform(test['family'])

    # Lag features
    train = train.sort_values(['store_nbr', 'family', 'date'])
    train['lag_1'] = train.groupby(['store_nbr', 'family'])['sales'].shift(1)
    train['rolling_mean_7'] = train.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7).mean()
    train = train.dropna()

    # Train model
    features = ['onpromotion', 'is_holiday', 'dayofweek', 'month', 'lag_1', 'rolling_mean_7', 'family_enc']
    X = train[features]
    y = train['sales']

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.1)
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    st.success(f"✅ Model MAE on Validation: {mean_absolute_error(y_val, val_pred):.2f}")

    # Prepare test set for prediction
    latest = train.groupby(['store_nbr', 'family']).last().reset_index()
    test = test.merge(latest[['store_nbr', 'family', 'lag_1', 'rolling_mean_7']], on=['store_nbr', 'family'], how='left')

    test['sales'] = model.predict(test[features])
    forecast = test[['store_nbr', 'family', 'date', 'sales']]

    st.subheader("🔮 Sample Forecast Results")
    st.dataframe(forecast.head(20))

    # Inventory Optimization
    inventory_plan = []

    for (store, family), group in test.groupby(['store_nbr', 'family']):
        predicted_demand = group['sales'].sum()
        std_dev = group['sales'].std() or 1
        eoq = calc_eoq(predicted_demand, setup_cost=100, holding_cost=2, cv=0.3)
        safety = calc_safety_stock(service_level=0.95, std_dev=std_dev, lead_time=7)

        inventory_plan.append({
            'store_nbr': store,
            'family': family,
            'forecasted_demand': round(predicted_demand, 2),
            'EOQ': round(eoq, 2),
            'Safety_Stock': round(safety, 2)
        })

    inventory_df = pd.DataFrame(inventory_plan)

    st.subheader("📦 Inventory Plan (EOQ & Safety Stock)")
    st.dataframe(inventory_df.head(20))

    # Download buttons
    st.download_button("📥 Download Forecast CSV", forecast.to_csv(index=False), file_name="submission.csv")
    st.download_button("📥 Download Inventory Plan CSV", inventory_df.to_csv(index=False), file_name="inventory_plan.csv")

    # Visualization
    st.subheader("📊 EOQ by Product Family")
    family_eoq = inventory_df.groupby("family")["EOQ"].mean().sort_values()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    family_eoq.plot(kind='barh', ax=ax)
    plt.xlabel("EOQ")
    plt.title("Average EOQ by Product Family")
    st.pyplot(fig)

else:
    st.info("Please upload all 3 required CSV files from the dataset: `train.csv`, `test.csv`, and `holidays_events.csv`.")
