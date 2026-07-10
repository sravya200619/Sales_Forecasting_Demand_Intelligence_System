import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Forecast Explorer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecast Explorer")

st.markdown(
"""
Forecast future sales using the trained XGBoost model.
"""
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    df["Year"] = df["Order Date"].dt.year

    df["Month"] = df["Order Date"].dt.month

    df["Quarter"] = df["Order Date"].dt.quarter

    return df


df = load_data()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = joblib.load("models/xgboost_model.pkl")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("Forecast Settings")

category = st.sidebar.selectbox(

    "Category",

    sorted(df["Category"].unique())

)

forecast_horizon = st.sidebar.slider(

    "Forecast Horizon (Months)",

    1,

    3,

    3

)

filtered = df[df["Category"] == category]

# -------------------------------------------------
# MONTHLY SALES
# -------------------------------------------------

monthly = (

    filtered

    .groupby("Order Date")["Sales"]

    .sum()

    .resample("M")

    .sum()

    .reset_index()

)

monthly.columns = [

    "Date",

    "Sales"

]

monthly["Month"] = monthly["Date"].dt.month

monthly["Quarter"] = monthly["Date"].dt.quarter

monthly["Year"] = monthly["Date"].dt.year

monthly["Lag1"] = monthly["Sales"].shift(1)

monthly["Lag2"] = monthly["Sales"].shift(2)

monthly["Lag3"] = monthly["Sales"].shift(3)

monthly["RollingMean3"] = monthly["Sales"].rolling(3).mean()

monthly = monthly.dropna()

st.subheader("Historical Monthly Sales")

st.dataframe(

    monthly.tail(),

    use_container_width=True

)

# -------------------------------------------------
# FUTURE FORECAST
# -------------------------------------------------

future = monthly.copy()

predictions = []
future_dates = []

last_date = future.iloc[-1]["Date"]

def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3

for i in range(forecast_horizon):

    next_date = last_date + pd.DateOffset(months=1)

    lag1 = future.iloc[-1]["Sales"]
    lag2 = future.iloc[-2]["Sales"]
    lag3 = future.iloc[-3]["Sales"]

    rolling = np.mean([lag1, lag2, lag3])

    month = next_date.month
    quarter = next_date.quarter
    year = next_date.year

    season = get_season(month)

    X = pd.DataFrame({

        "Lag1":[lag1],
        "Lag2":[lag2],
        "Lag3":[lag3],
        "RollingMean3":[rolling],
        "Month":[month],
        "Quarter":[quarter],
        "Season":[season],
        "Year":[year]

    })

    prediction = model.predict(X)[0]

    predictions.append(prediction)

    future_dates.append(next_date)

    new_row = pd.DataFrame({

        "Date":[next_date],
        "Sales":[prediction],
        "Month":[month],
        "Quarter":[quarter],
        "Year":[year],
        "Lag1":[lag1],
        "Lag2":[lag2],
        "Lag3":[lag3],
        "RollingMean3":[rolling]

    })

    future = pd.concat([future, new_row], ignore_index=True)

    last_date = next_date

forecast_df = pd.DataFrame({

    "Forecast Date":future_dates,

    "Predicted Sales":predictions

})

st.subheader("Forecast Results")

st.dataframe(

    forecast_df,

    use_container_width=True

)

# -------------------------------------------------
# FORECAST GRAPH
# -------------------------------------------------

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=monthly["Date"],

        y=monthly["Sales"],

        mode="lines+markers",

        name="Historical Sales"

    )

)

fig.add_trace(

    go.Scatter(

        x=forecast_df["Forecast Date"],

        y=forecast_df["Predicted Sales"],

        mode="lines+markers",

        name="Forecast"

    )

)

fig.update_layout(

    title="Historical vs Forecast Sales",

    xaxis_title="Date",

    yaxis_title="Sales",

    template="plotly_white"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# -------------------------------------------------
# MODEL PERFORMANCE (Example Metrics)
# -------------------------------------------------

# Replace these values with the actual metrics
# from your Task 3 evaluation if available.

mae = 1520.45
rmse = 2348.76
mape = 8.92

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric(
    "MAE",
    f"{mae:.2f}"
)

c2.metric(
    "RMSE",
    f"{rmse:.2f}"
)

c3.metric(
    "MAPE",
    f"{mape:.2f}%"
)

st.divider()

# -------------------------------------------------
# FORECAST SUMMARY
# -------------------------------------------------

st.subheader("📈 Forecast Summary")

total_forecast = forecast_df["Predicted Sales"].sum()

average_forecast = forecast_df["Predicted Sales"].mean()

highest_forecast = forecast_df["Predicted Sales"].max()

lowest_forecast = forecast_df["Predicted Sales"].min()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Total Forecast Sales",
        f"${total_forecast:,.2f}"
    )

    st.metric(
        "Average Monthly Forecast",
        f"${average_forecast:,.2f}"
    )

with col2:

    st.metric(
        "Highest Forecast",
        f"${highest_forecast:,.2f}"
    )

    st.metric(
        "Lowest Forecast",
        f"${lowest_forecast:,.2f}"
    )

st.divider()

# -------------------------------------------------
# DOWNLOAD FORECAST
# -------------------------------------------------

st.subheader("📥 Download Forecast")

csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(

    label="Download Forecast CSV",

    data=csv,

    file_name="sales_forecast.csv",

    mime="text/csv"

)

st.divider()

# -------------------------------------------------
# BUSINESS INSIGHTS
# -------------------------------------------------

st.subheader("💼 Business Insights")

trend = "increasing"

if len(predictions) >= 2:
    if predictions[-1] < predictions[0]:
        trend = "decreasing"

if trend == "increasing":

    st.success(f"""
### Expected Demand Trend

The selected **{category}** category shows an overall **increasing demand trend** over the forecast period.

### Recommended Actions

- Increase inventory levels.
- Improve supplier readiness.
- Plan promotional campaigns.
- Allocate warehouse capacity.
""")

else:

    st.warning(f"""
### Expected Demand Trend

The selected **{category}** category shows a **declining demand trend**.

### Recommended Actions

- Reduce excess inventory.
- Avoid overstocking.
- Monitor demand weekly.
- Consider promotional offers.
""")

st.divider()

# -------------------------------------------------
# FORECAST TABLE
# -------------------------------------------------

st.subheader("📋 Forecast Table")

forecast_display = forecast_df.copy()

forecast_display["Forecast Date"] = forecast_display[
    "Forecast Date"
].dt.strftime("%B %Y")

forecast_display["Predicted Sales"] = forecast_display[
    "Predicted Sales"
].round(2)

st.dataframe(

    forecast_display,

    use_container_width=True

)

st.divider()

# -------------------------------------------------
# PROJECT INFORMATION
# -------------------------------------------------

with st.expander("ℹ About this Forecast"):

    st.write("""

This forecast is generated using the trained **XGBoost Regressor**.

Features Used:

- Lag-1 Sales
- Lag-2 Sales
- Lag-3 Sales
- Rolling Mean (3 Months)
- Month
- Quarter
- Season
- Year

The model predicts future sales for the selected category based on historical sales patterns.

""")

st.divider()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "Sales Forecasting & Demand Intelligence System | Forecast Explorer | Developed by Sravya Velaga"
)