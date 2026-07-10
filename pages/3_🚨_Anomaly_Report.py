import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Anomaly Report",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Sales Anomaly Detection")

st.markdown("""
Detect unusual sales spikes and drops using **Isolation Forest** and **Z-Score Analysis**.
""")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    # Convert DD/MM/YYYY dates safely
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=["Order Date"])

    # Ensure Sales is numeric
    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    # Remove invalid sales values
    df = df.dropna(subset=["Sales"])

    # Sort by date
    df = df.sort_values("Order Date")

    return df


df = load_data()

# -----------------------------------------------------
# CHECK DATA
# -----------------------------------------------------

if df.empty:
    st.error("No valid records found in train.csv")
    st.stop()

# -----------------------------------------------------
# WEEKLY SALES
# -----------------------------------------------------

weekly_sales = (
    df
    .set_index("Order Date")
    .resample("W")
    .agg({"Sales": "sum"})
    .reset_index()
)

weekly_sales = weekly_sales.dropna()

if weekly_sales.empty:
    st.error("No weekly sales available.")
    st.stop()

# -----------------------------------------------------
# WEEKLY SALES PREVIEW
# -----------------------------------------------------

st.subheader("Weekly Sales")

st.dataframe(
    weekly_sales.tail(),
    use_container_width=True
)

# -----------------------------------------------------
# ISOLATION FOREST
# -----------------------------------------------------

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly_sales = weekly_sales.dropna(subset=["Sales"]).copy()

weekly_sales["Isolation"] = model.fit_predict(
    weekly_sales[["Sales"]]
)

normal = weekly_sales[
    weekly_sales["Isolation"] == 1
]

anomaly = weekly_sales[
    weekly_sales["Isolation"] == -1
]

# -----------------------------------------------------
# ISOLATION FOREST CHART
# -----------------------------------------------------

st.subheader("Isolation Forest Detection")

fig = px.line(
    weekly_sales,
    x="Order Date",
    y="Sales",
    title="Weekly Sales with Isolation Forest Anomalies"
)

fig.add_scatter(
    x=anomaly["Order Date"],
    y=anomaly["Sales"],
    mode="markers",
    marker=dict(
        size=12,
        color="red",
        symbol="x"
    ),
    name="Isolation Forest Anomaly"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# -----------------------------------------------------
# Z-SCORE DETECTION
# -----------------------------------------------------

weekly_sales["RollingMean"] = (
    weekly_sales["Sales"]
    .rolling(window=8, min_periods=1)
    .mean()
)

weekly_sales["RollingStd"] = (
    weekly_sales["Sales"]
    .rolling(window=8, min_periods=1)
    .std()
)

# Prevent division by zero
weekly_sales["RollingStd"] = (
    weekly_sales["RollingStd"]
    .replace(0, np.nan)
)

weekly_sales["ZScore"] = (
    weekly_sales["Sales"] -
    weekly_sales["RollingMean"]
) / weekly_sales["RollingStd"]

weekly_sales["Z_Anomaly"] = (
    weekly_sales["ZScore"].abs() > 2
)

z_anomaly = weekly_sales[
    weekly_sales["Z_Anomaly"]
].copy()

# -----------------------------------------------------
# Z-SCORE CHART
# -----------------------------------------------------

st.subheader("Z-Score Detection")

fig2 = px.line(
    weekly_sales,
    x="Order Date",
    y="Sales",
    title="Weekly Sales with Z-Score Anomalies"
)

fig2.add_scatter(
    x=z_anomaly["Order Date"],
    y=z_anomaly["Sales"],
    mode="markers",
    marker=dict(
        size=12,
        color="orange",
        symbol="diamond"
    ),
    name="Z-Score Anomaly"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.divider()

# -----------------------------------------------------
# COMPARISON TABLE
# -----------------------------------------------------

st.subheader("Detected Anomalies")

col1, col2 = st.columns(2)

with col1:

    st.write("### Isolation Forest")

    if anomaly.empty:
        st.info("No anomalies detected by Isolation Forest.")
    else:
        st.dataframe(
            anomaly[["Order Date", "Sales"]],
            use_container_width=True
        )

with col2:

    st.write("### Z-Score")

    if z_anomaly.empty:
        st.info("No anomalies detected by Z-Score.")
    else:
        st.dataframe(
            z_anomaly[["Order Date", "Sales"]],
            use_container_width=True
        )

st.divider()

# -----------------------------------------------------
# KPI CARDS
# -----------------------------------------------------

st.subheader("📊 Anomaly Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Isolation Forest",
    len(anomaly)
)

col2.metric(
    "Z-Score",
    len(z_anomaly)
)

common = len(
    set(anomaly["Order Date"]).intersection(
        set(z_anomaly["Order Date"])
    )
)

col3.metric(
    "Common Anomalies",
    common
)

st.divider()

# -----------------------------------------------------
# DOWNLOAD REPORTS
# -----------------------------------------------------

st.subheader("📥 Download Reports")

csv1 = anomaly.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Isolation Forest Report",
    data=csv1,
    file_name="IsolationForest_Anomalies.csv",
    mime="text/csv"
)

csv2 = z_anomaly.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Z-Score Report",
    data=csv2,
    file_name="ZScore_Anomalies.csv",
    mime="text/csv"
)

st.divider()

# -----------------------------------------------------
# BUSINESS INSIGHTS
# -----------------------------------------------------

st.subheader("💼 Business Interpretation")

if weekly_sales.empty:

    st.warning("No weekly sales data available.")

else:

    highest = weekly_sales.loc[
        weekly_sales["Sales"].idxmax()
    ]

    lowest = weekly_sales.loc[
        weekly_sales["Sales"].idxmin()
    ]

    st.success(f"""
### Key Findings

✅ **Highest Weekly Sales**

**Week Ending:** {highest['Order Date'].strftime('%d-%b-%Y')}

**Sales:** ${highest['Sales']:,.2f}

---

📉 **Lowest Weekly Sales**

**Week Ending:** {lowest['Order Date'].strftime('%d-%b-%Y')}

**Sales:** ${lowest['Sales']:,.2f}

---

### Possible Reasons

• Seasonal demand

• Festival sales

• Flash promotions

• Black Friday events

• Supply chain delays

• Inventory shortages

• Sudden changes in customer demand
""")

st.divider()

# -----------------------------------------------------
# ABOUT ANOMALY DETECTION
# -----------------------------------------------------

with st.expander("ℹ About These Models"):

    st.markdown("""
### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm that detects unusual observations by isolating them from normal data.

---

### Z-Score Analysis

The Z-Score method compares each week's sales with the rolling average and identifies values that are more than **2 standard deviations** away.

---

### Why Use Two Methods?

Using both techniques improves anomaly detection by combining:

- Machine Learning detection
- Statistical detection

This helps reduce false positives and provides more reliable business insights.

---

### Business Benefits

- Detect sudden sales spikes
- Identify unexpected sales drops
- Improve inventory planning
- Evaluate marketing campaigns
- Monitor supply chain issues
- Support forecasting decisions
""")

st.divider()

# -----------------------------------------------------
# FINAL RECOMMENDATIONS
# -----------------------------------------------------

st.subheader("📌 Business Recommendations")

st.info("""
### Recommended Actions

📦 Increase inventory before seasonal demand peaks.

📊 Continuously monitor weekly sales anomalies.

🚚 Improve supplier coordination during high-demand periods.

🎯 Schedule promotions based on historical demand spikes.

📉 Investigate sudden sales drops to identify operational issues.

📈 Combine anomaly detection with forecasting models for better demand planning.

💼 Review anomaly reports regularly to support executive decision-making.
""")

st.divider()

# -----------------------------------------------------
# PROJECT SUMMARY
# -----------------------------------------------------

st.subheader("📄 Project Summary")

summary = pd.DataFrame({
    "Metric": [
        "Total Weeks Analysed",
        "Isolation Forest Anomalies",
        "Z-Score Anomalies",
        "Common Anomalies"
    ],
    "Value": [
        len(weekly_sales),
        len(anomaly),
        len(z_anomaly),
        common
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Sravya Velaga | Sales Forecasting & Demand Intelligence System | Anomaly Report"
)
# -----------------------------------------------------
# BUSINESS INSIGHTS
# -----------------------------------------------------

st.subheader("💼 Business Interpretation")

if not weekly_sales.empty:

    highest = weekly_sales.loc[
        weekly_sales["Sales"].idxmax()
    ]

    lowest = weekly_sales.loc[
        weekly_sales["Sales"].idxmin()
    ]

    st.success(f"""
### Key Findings

✅ Highest Weekly Sales

**Date:** {highest['Order Date'].date()}

**Sales:** ${highest['Sales']:,.2f}

---

📉 Lowest Weekly Sales

**Date:** {lowest['Order Date'].date()}

**Sales:** ${lowest['Sales']:,.2f}

---

### Possible Reasons

• Festival Season

• Flash Sales

• Holiday Discounts

• Inventory Shortage

• Supply Chain Delays

• Low Customer Demand

""")

else:
    st.warning("No weekly sales data available.")

st.divider()

# -----------------------------------------------------
# ABOUT ANOMALY DETECTION
# -----------------------------------------------------

with st.expander("ℹ About These Models"):

    st.markdown("""
### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm used to identify unusual observations by isolating them from the rest of the data.

---

### Z-Score Method

The Z-Score method detects values that are more than **2 standard deviations** away from the rolling average.

---

### Business Value

These methods help identify:

- Unexpected demand spikes
- Sudden sales drops
- Inventory shortages
- Promotion effectiveness
- Supply chain disruptions
""")

st.divider()

# -----------------------------------------------------
# FINAL RECOMMENDATIONS
# -----------------------------------------------------

st.subheader("📌 Recommendations")

st.info("""
### Suggested Business Actions

📦 Increase inventory before seasonal demand peaks.

📊 Monitor unusual demand spikes weekly.

🚚 Improve supplier coordination for high-demand periods.

🎯 Plan promotions based on historical sales anomalies.

📉 Investigate sudden drops to identify operational issues.
""")

st.divider()

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown("---")

st.caption(
    "Sales Forecasting & Demand Intelligence System | "
    "Anomaly Report | Developed by Sravya Velaga"
)