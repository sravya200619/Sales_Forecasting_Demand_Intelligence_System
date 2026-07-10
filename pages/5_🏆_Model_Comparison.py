import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Forecasting Model Comparison")

st.markdown("""
Compare the performance of **SARIMA**, **Facebook Prophet**, and **XGBoost**
to determine the best forecasting model.
""")

# -------------------------------------------------------
# MODEL METRICS
# -------------------------------------------------------

# Replace these values with your actual metrics from Task 3

comparison = pd.DataFrame({

    "Model":[
        "SARIMA",
        "Prophet",
        "XGBoost"
    ],

    "MAE":[
        1984.26,
        1756.40,
        1520.45
    ],

    "RMSE":[
        2689.84,
        2456.71,
        2348.76
    ],

    "MAPE":[
        11.82,
        9.71,
        8.92
    ],

    "Month-1":[
        75432,
        76840,
        78265
    ],

    "Month-2":[
        78110,
        79450,
        81220
    ],

    "Month-3":[
        80325,
        81980,
        83510
    ]

})

st.subheader("📋 Model Comparison Table")

st.dataframe(

    comparison,

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------

best_model = comparison.sort_values(
    "RMSE"
).iloc[0]

col1,col2,col3,col4 = st.columns(4)

col1.metric(

    "Best Model",

    best_model["Model"]

)

col2.metric(

    "MAE",

    round(best_model["MAE"],2)

)

col3.metric(

    "RMSE",

    round(best_model["RMSE"],2)

)

col4.metric(

    "MAPE",

    f"{best_model['MAPE']:.2f}%"

)

st.divider()

# -------------------------------------------------------
# MAE COMPARISON
# -------------------------------------------------------

st.subheader("📊 Mean Absolute Error (MAE)")

fig_mae = px.bar(
    comparison,
    x="Model",
    y="MAE",
    color="Model",
    text_auto=".2f",
    title="MAE Comparison"
)

fig_mae.update_layout(template="plotly_white")

st.plotly_chart(
    fig_mae,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# RMSE COMPARISON
# -------------------------------------------------------

st.subheader("📊 Root Mean Squared Error (RMSE)")

fig_rmse = px.bar(
    comparison,
    x="Model",
    y="RMSE",
    color="Model",
    text_auto=".2f",
    title="RMSE Comparison"
)

fig_rmse.update_layout(template="plotly_white")

st.plotly_chart(
    fig_rmse,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# MAPE COMPARISON
# -------------------------------------------------------

st.subheader("📊 Mean Absolute Percentage Error (MAPE)")

fig_mape = px.bar(
    comparison,
    x="Model",
    y="MAPE",
    color="Model",
    text_auto=".2f",
    title="MAPE Comparison (%)"
)

fig_mape.update_layout(template="plotly_white")

st.plotly_chart(
    fig_mape,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# FORECAST COMPARISON
# -------------------------------------------------------

st.subheader("📈 3-Month Forecast Comparison")

forecast = comparison.melt(

    id_vars="Model",

    value_vars=["Month-1","Month-2","Month-3"],

    var_name="Forecast Month",

    value_name="Forecast Sales"

)

fig_forecast = px.line(

    forecast,

    x="Forecast Month",

    y="Forecast Sales",

    color="Model",

    markers=True,

    title="Forecast Comparison"

)

fig_forecast.update_layout(

    template="plotly_white"

)

st.plotly_chart(

    fig_forecast,

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# BEST MODEL
# -------------------------------------------------------

st.subheader("🏆 Best Performing Model")

winner = comparison.sort_values("RMSE").iloc[0]

st.success(f"""

## 🥇 {winner['Model']}

### Performance

• MAE : {winner['MAE']:.2f}

• RMSE : {winner['RMSE']:.2f}

• MAPE : {winner['MAPE']:.2f}%

This model achieved the lowest prediction error and is recommended for production deployment.

""")

st.divider()

# -------------------------------------------------------
# MODEL RANKING
# -------------------------------------------------------

st.subheader("🥇 Model Ranking")

ranking = comparison.sort_values("RMSE").reset_index(drop=True)

ranking.index = ranking.index + 1

st.dataframe(
    ranking,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# DOWNLOAD REPORT
# -------------------------------------------------------

st.subheader("📥 Download Model Comparison Report")

csv = comparison.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Comparison CSV",
    data=csv,
    file_name="Model_Comparison_Report.csv",
    mime="text/csv"
)

st.divider()

# -------------------------------------------------------
# EXECUTIVE BUSINESS RECOMMENDATION
# -------------------------------------------------------

st.subheader("💼 Executive Recommendation")

recommended = comparison.sort_values("RMSE").iloc[0]["Model"]

st.success(f"""

### Recommended Forecasting Model: **{recommended}**

Based on the evaluation metrics:

- Lowest MAE
- Lowest RMSE
- Lowest MAPE
- Most accurate 3-month forecast

This model is recommended for production because it provides the best balance
between prediction accuracy and business reliability.

""")

st.divider()

# -------------------------------------------------------
# WHY XGBOOST?
# -------------------------------------------------------

st.subheader("⭐ Why This Model?")

st.info("""

### Key Reasons

- Captures non-linear sales patterns effectively.
- Uses lag features and rolling averages to improve forecasting.
- Handles seasonal changes better than simple statistical models.
- Scales well for large retail datasets.
- Easy to retrain as new sales data becomes available.
- Suitable for real-world inventory planning and demand forecasting.

""")

st.divider()

# -------------------------------------------------------
# MODEL COMPARISON SUMMARY
# -------------------------------------------------------

st.subheader("📋 Model Comparison Summary")

summary = pd.DataFrame({
    "Criteria": [
        "Forecast Accuracy",
        "Handles Seasonality",
        "Machine Learning",
        "Interpretability",
        "Production Ready"
    ],
    "SARIMA": [
        "Good",
        "Excellent",
        "No",
        "High",
        "Yes"
    ],
    "Prophet": [
        "Very Good",
        "Excellent",
        "No",
        "High",
        "Yes"
    ],
    "XGBoost": [
        "Excellent",
        "Very Good",
        "Yes",
        "Medium",
        "Yes"
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# FINAL CONCLUSION
# -------------------------------------------------------

st.subheader("🎯 Final Conclusion")

st.success("""

### Project Outcome

✔ Successfully analyzed retail sales data.

✔ Built and compared three forecasting models.

✔ Identified the best-performing model using evaluation metrics.

✔ Developed an interactive Streamlit dashboard.

✔ Implemented anomaly detection.

✔ Performed product demand segmentation using K-Means.

✔ Generated business-ready insights for inventory planning and decision-making.

""")

st.divider()

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("---")

st.caption(
    "Sales Forecasting & Demand Intelligence System | Model Comparison | Developed by Sravya Velaga"
)