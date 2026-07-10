import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Demand Segmentation",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Product Demand Segmentation")

st.markdown("""
Cluster product sub-categories based on historical sales
using **K-Means Clustering**.
""")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Order Date"])

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month

    return df


df = load_data()

# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------

product = (
    df.groupby("Sub-Category")
      .agg(
          TotalSales=("Sales", "sum"),
          AverageSales=("Sales", "mean"),
          MaxSales=("Sales", "max"),
          MinSales=("Sales", "min"),
          OrderCount=("Sales", "count")
      )
      .reset_index()
)

# -------------------------------------------------------
# MONTHLY SALES
# -------------------------------------------------------

monthly = (
    df.groupby(
        [
            "Sub-Category",
            pd.Grouper(key="Order Date", freq="ME")
        ]
    )["Sales"]
    .sum()
    .reset_index()
)

volatility = (
    monthly.groupby("Sub-Category")["Sales"]
    .std()
)

product["SalesVolatility"] = (
    product["Sub-Category"]
    .map(volatility)
)

product["SalesVolatility"] = product["SalesVolatility"].fillna(0)
# -------------------------------------------------------
# YEARLY GROWTH
# -------------------------------------------------------

yearly = (
    df.groupby(["Sub-Category", "Year"])["Sales"]
      .sum()
      .reset_index()
)

yearly["GrowthRate"] = (
    yearly.groupby("Sub-Category")["Sales"]
          .pct_change()
)

growth = (
    yearly.groupby("Sub-Category")["GrowthRate"]
          .mean()
)

product["GrowthRate"] = (
    product["Sub-Category"]
           .map(growth)
)

product["GrowthRate"] = product["GrowthRate"].fillna(0)

# -------------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------------

features = product[[
    "TotalSales",
    "AverageSales",
    "MaxSales",
    "MinSales",
    "OrderCount",
    "SalesVolatility",
    "GrowthRate"
]]

features = features.replace([np.inf, -np.inf], 0)
features = features.fillna(0)

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# -------------------------------------------------------
# ELBOW METHOD
# -------------------------------------------------------

wcss = []

for i in range(2, 9):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=20
    )

    model.fit(scaled_features)

    wcss.append(model.inertia_)

st.subheader("📈 Elbow Method")

fig = px.line(
    x=list(range(2, 9)),
    y=wcss,
    markers=True,
    labels={
        "x": "Number of Clusters",
        "y": "WCSS"
    },
    title="Elbow Method for Optimal Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------------------------------
# KMEANS MODEL
# -------------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

product["Cluster"] = kmeans.fit_predict(
    scaled_features
)
# -------------------------------------------------------
# PCA VISUALIZATION
# -------------------------------------------------------

pca = PCA(n_components=2, random_state=42)

components = pca.fit_transform(scaled_features)

plot_df = pd.DataFrame({
    "PC1": components[:, 0],
    "PC2": components[:, 1],
    "Cluster": product["Cluster"].astype(str),
    "SubCategory": product["Sub-Category"],
    "TotalSales": product["TotalSales"]
})

st.subheader("📦 Product Demand Clusters")

fig_cluster = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name="SubCategory",
    hover_data=["TotalSales"],
    text="SubCategory",
    title="K-Means Product Segmentation"
)

fig_cluster.update_traces(
    textposition="top center"
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

# -------------------------------------------------------
# CLUSTER SUMMARY
# -------------------------------------------------------

summary = (
    product.groupby("Cluster")
           .agg({
               "TotalSales": "mean",
               "AverageSales": "mean",
               "OrderCount": "mean",
               "SalesVolatility": "mean",
               "GrowthRate": "mean"
           })
           .round(2)
)

st.subheader("📊 Cluster Summary")

st.dataframe(
    summary,
    use_container_width=True
)

# -------------------------------------------------------
# SEGMENT LABELS
# -------------------------------------------------------

cluster_names = {
    0: "High Demand",
    1: "Growing Products",
    2: "Seasonal Products",
    3: "Low Demand"
}

product["Demand Segment"] = (
    product["Cluster"]
    .map(cluster_names)
)

# -------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------

st.subheader("📈 Segmentation KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Products",
    len(product)
)

col2.metric(
    "Clusters",
    product["Cluster"].nunique()
)

col3.metric(
    "Highest Sales",
    f"${product['TotalSales'].max():,.0f}"
)

col4.metric(
    "Average Sales",
    f"${product['AverageSales'].mean():,.0f}"
)

st.divider()

# -------------------------------------------------------
# PRODUCT TABLE
# -------------------------------------------------------

st.subheader("📋 Product Segments")

display_cols = [
    "Sub-Category",
    "Demand Segment",
    "TotalSales",
    "AverageSales",
    "OrderCount",
    "SalesVolatility",
    "GrowthRate"
]

st.dataframe(
    product[display_cols].sort_values(
        "TotalSales",
        ascending=False
    ),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------
# DOWNLOAD REPORT
# -------------------------------------------------------

csv = product.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Segmentation Report",
    data=csv,
    file_name="Demand_Segmentation.csv",
    mime="text/csv"
)
# -------------------------------------------------------
# STOCKING STRATEGY
# -------------------------------------------------------

st.subheader("📦 Recommended Stocking Strategy")

st.success("""

### 🟢 High Demand Products
• Maintain high inventory levels.
• Prioritize supplier availability.
• Keep sufficient safety stock.
• Monitor weekly demand trends.

---

### 🔵 Growing Products
• Increase inventory gradually.
• Monitor monthly growth.
• Plan promotional campaigns.
• Improve product visibility.

---

### 🟡 Seasonal Products
• Stock according to seasonal demand.
• Avoid excess inventory.
• Forecast demand before peak seasons.
• Review historical seasonal trends.

---

### 🔴 Low Demand Products
• Reduce inventory levels.
• Bundle with popular products.
• Offer promotional discounts.
• Consider product replacement if demand continues to decline.

""")

st.divider()

# -------------------------------------------------------
# BUSINESS INSIGHTS
# -------------------------------------------------------

st.subheader("💼 Executive Insights")

top_sales = product.loc[
    product["TotalSales"].idxmax()
]

top_avg = product.loc[
    product["AverageSales"].idxmax()
]

top_growth = product.loc[
    product["GrowthRate"].idxmax()
]

st.info(f"""

### Key Business Findings

🏆 Highest Revenue Product Category

**{top_sales['Sub-Category']}**

Total Sales:
**${top_sales['TotalSales']:,.2f}**

---

📈 Highest Average Sales

**{top_avg['Sub-Category']}**

Average Sales:
**${top_avg['AverageSales']:,.2f}**

---

🚀 Fastest Growing Category

**{top_growth['Sub-Category']}**

Growth Rate:
**{top_growth['GrowthRate']:.2%}**

---

Demand segmentation helps identify:

• Products that require continuous stocking.

• Products showing strong growth potential.

• Seasonal products needing demand planning.

• Low-demand products requiring inventory optimization.

""")

st.divider()

# -------------------------------------------------------
# ABOUT DEMAND SEGMENTATION
# -------------------------------------------------------

with st.expander("ℹ About Demand Segmentation"):

    st.markdown("""

### K-Means Clustering

K-Means is an unsupervised Machine Learning algorithm that groups products having similar demand characteristics.

---

### Features Used

- Total Sales
- Average Sales
- Number of Orders
- Sales Volatility
- Growth Rate

---

### Business Benefits

✅ Better inventory planning

✅ Demand-driven purchasing

✅ Warehouse optimization

✅ Reduced stock-outs

✅ Reduced overstocking

✅ Improved forecasting

✅ Better product management

""")

st.divider()

# -------------------------------------------------------
# FINAL RECOMMENDATIONS
# -------------------------------------------------------

st.subheader("📌 Recommendations")

st.warning("""

### Recommended Actions

📦 Maintain adequate inventory for high-demand products.

📈 Closely monitor products with rapid growth.

📅 Prepare inventory before seasonal demand peaks.

💰 Review pricing strategies for low-performing products.

📊 Re-run demand segmentation every month to capture changing market trends.

🚚 Improve supplier coordination for fast-moving products.

""")

st.divider()

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("---")

st.caption(
    "Sales Forecasting & Demand Intelligence System | Product Demand Segmentation Dashboard | Developed by Sravya Velaga"
)