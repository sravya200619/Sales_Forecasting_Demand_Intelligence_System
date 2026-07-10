import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np

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

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    return df

df = load_data()

# -----------------------------------------------------
# WEEKLY SALES
# -----------------------------------------------------

weekly_sales = (

    df

    .set_index("Order Date")

    .resample("W")["Sales"]

    .sum()

    .reset_index()

)

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

    name="Anomaly"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

# -----------------------------------------------------
# Z-SCORE ANOMALY DETECTION
# -----------------------------------------------------

weekly_sales["RollingMean"] = (

    weekly_sales["Sales"]

    .rolling(8)

    .mean()

)

weekly_sales["RollingStd"] = (

    weekly_sales["Sales"]

    .rolling(8)

    .std()

)

weekly_sales["ZScore"] = (

    weekly_sales["Sales"]

    -

    weekly_sales["RollingMean"]

) / weekly_sales["RollingStd"]

weekly_sales["Z_Anomaly"] = (

    np.abs(

        weekly_sales["ZScore"]

    ) > 2

)

z_anomaly = weekly_sales[

    weekly_sales["Z_Anomaly"]

]

# -----------------------------------------------------
# Z-SCORE GRAPH
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

    st.dataframe(

        anomaly[["Order Date","Sales"]],

        use_container_width=True

    )

with col2:

    st.write("### Z-Score")

    st.dataframe(

        z_anomaly[["Order Date","Sales"]],

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
# DOWNLOAD REPORT
# -----------------------------------------------------

st.subheader("📥 Download Reports")

csv1 = anomaly.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Isolation Forest Report",
    csv1,
    "IsolationForest_Anomalies.csv",
    "text/csv"
)

csv2 = z_anomaly.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Z-Score Report",
    csv2,
    "ZScore_Anomalies.csv",
    "text/csv"
)

st.divider()

# -----------------------------------------------------
# BUSINESS INSIGHTS
# -----------------------------------------------------

st.subheader("💼 Business Interpretation")

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

Possible Reasons

• Festival Season

• Flash Sales

• Black Friday

• Christmas Discounts

• Inventory Shortage

• Supply Chain Delay

• Low Customer Demand

""")

st.divider()

# -----------------------------------------------------
# ABOUT ANOMALY DETECTION
# -----------------------------------------------------

with st.expander("ℹ About These Models"):

    st.markdown("""

### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm that detects unusual observations by isolating them from the rest of the data.

---

### Z-Score Method

The Z-Score method identifies sales values that are more than **2 standard deviations** away from the rolling average.

---

### Business Value

These techniques help businesses identify:

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
    "Sales Forecasting & Demand Intelligence System | Anomaly Report | Developed by Sravya Velaga"
)