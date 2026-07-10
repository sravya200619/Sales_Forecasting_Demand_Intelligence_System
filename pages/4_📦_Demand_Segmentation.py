import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Demand Segmentation",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Product Demand Segmentation")

st.markdown("""
Cluster product sub-categories based on sales performance, demand growth,
profitability, and volatility using **K-Means Clustering**.
""")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    df["Year"] = df["Order Date"].dt.year

    return df

df = load_data()

# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------

product = df.groupby("Sub-Category").agg({

    "Sales":"sum",
    "Quantity":"sum",
    "Discount":"mean",
    "Profit":"sum"

}).reset_index()

product.columns = [

    "SubCategory",
    "TotalSales",
    "TotalQuantity",
    "AverageDiscount",
    "TotalProfit"

]

avg_order = (

    df

    .groupby("Sub-Category")["Sales"]

    .mean()

)

product["AverageOrderValue"] = avg_order.values

monthly = (

    df

    .groupby([

        "Sub-Category",

        pd.Grouper(

            key="Order Date",

            freq="M"

        )

    ])["Sales"]

    .sum()

    .reset_index()

)

volatility = (

    monthly

    .groupby("Sub-Category")["Sales"]

    .std()

)

product["SalesVolatility"] = volatility.values

yearly = (

    df

    .groupby([

        "Sub-Category",

        "Year"

    ])["Sales"]

    .sum()

    .reset_index()

)

yearly["Growth"] = (

    yearly

    .groupby("Sub-Category")["Sales"]

    .pct_change()

)

growth = (

    yearly

    .groupby("Sub-Category")["Growth"]

    .mean()

)

product["GrowthRate"] = growth.values

product["ProfitMargin"] = (

    product["TotalProfit"]

    /

    product["TotalSales"]

) * 100

product.fillna(0, inplace=True)

st.subheader("Engineered Product Features")

st.dataframe(

    product,

    use_container_width=True

)

# -------------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------------

features = product[[
    "TotalSales",
    "TotalQuantity",
    "AverageOrderValue",
    "SalesVolatility",
    "AverageDiscount",
    "ProfitMargin",
    "GrowthRate"
]]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# -------------------------------------------------------
# ELBOW METHOD
# -------------------------------------------------------

wcss = []

for i in range(2,9):

    model = KMeans(

        n_clusters=i,

        random_state=42,

        n_init=20

    )

    model.fit(scaled_features)

    wcss.append(model.inertia_)

st.subheader("Elbow Method")

fig_elbow = px.line(

    x=list(range(2,9)),

    y=wcss,

    markers=True,

    labels={

        "x":"Number of Clusters",

        "y":"WCSS"

    },

    title="Elbow Method"

)

st.plotly_chart(

    fig_elbow,

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
# PCA
# -------------------------------------------------------

pca = PCA(n_components=2)

components = pca.fit_transform(

    scaled_features

)

plot_df = pd.DataFrame({

    "PC1":components[:,0],

    "PC2":components[:,1],

    "Cluster":product["Cluster"],

    "SubCategory":product["SubCategory"]

})

# -------------------------------------------------------
# CLUSTER VISUALIZATION
# -------------------------------------------------------

st.subheader("Demand Segmentation Clusters")

fig_cluster = px.scatter(

    plot_df,

    x="PC1",

    y="PC2",

    color=plot_df["Cluster"].astype(str),

    text="SubCategory",

    size_max=18,

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

    product

    .groupby("Cluster")

    .mean(numeric_only=True)

)

st.subheader("Cluster Summary")

st.dataframe(

    summary,

    use_container_width=True

)

st.divider()

# -------------------------------------------------------
# DEMAND SEGMENT LABELS
# -------------------------------------------------------

cluster_names = {
    0: "High Volume, Stable Demand",
    1: "Growing Demand",
    2: "Low Volume, High Volatility",
    3: "Declining Demand"
}

product["Demand Segment"] = product["Cluster"].map(cluster_names)

# -------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------

st.subheader("📊 Segmentation Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Sub-Categories",
    product["SubCategory"].nunique()
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
    "Highest Profit",
    f"${product['TotalProfit'].max():,.0f}"
)

st.divider()

# -------------------------------------------------------
# PRODUCT SEGMENT TABLE
# -------------------------------------------------------

st.subheader("📋 Product Demand Segments")

display_cols = [
    "SubCategory",
    "Demand Segment",
    "TotalSales",
    "TotalQuantity",
    "AverageOrderValue",
    "SalesVolatility",
    "GrowthRate",
    "ProfitMargin"
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
# DOWNLOAD CSV
# -------------------------------------------------------

csv = product.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Segmentation Report",
    csv,
    "Demand_Segmentation.csv",
    "text/csv"
)

st.divider()

# -------------------------------------------------------
# STOCKING STRATEGY
# -------------------------------------------------------

st.subheader("📦 Recommended Stocking Strategy")

st.success("""

### 🟢 High Volume, Stable Demand
• Maintain high inventory levels
• Prioritize supplier availability
• Keep safety stock

---

### 🔵 Growing Demand
• Increase stock gradually
• Monitor demand monthly
• Plan marketing campaigns

---

### 🟡 Low Volume, High Volatility
• Keep limited inventory
• Order only when required
• Avoid overstocking

---

### 🔴 Declining Demand
• Reduce inventory
• Bundle products with popular items
• Consider discount strategies

""")

st.divider()

# -------------------------------------------------------
# BUSINESS INSIGHTS
# -------------------------------------------------------

top_product = product.loc[
    product["TotalSales"].idxmax()
]

best_profit = product.loc[
    product["TotalProfit"].idxmax()
]

st.info(f"""

### Executive Insights

🏆 Highest Revenue Sub-Category

**{top_product['SubCategory']}**

Sales: **${top_product['TotalSales']:,.2f}**

---

💰 Most Profitable Sub-Category

**{best_profit['SubCategory']}**

Profit: **${best_profit['TotalProfit']:,.2f}**

---

The segmentation helps identify which product groups require aggressive stocking,
which should be monitored closely, and which may need inventory optimization.

""")

st.divider()

# -------------------------------------------------------
# ABOUT K-MEANS
# -------------------------------------------------------

with st.expander("ℹ About Demand Segmentation"):

    st.markdown("""

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm that groups similar products based on demand characteristics.

### Features Used

- Total Sales
- Quantity Sold
- Average Order Value
- Profit Margin
- Sales Volatility
- Growth Rate
- Average Discount

### Business Benefits

- Better inventory planning
- Smarter warehouse allocation
- Improved purchasing decisions
- Reduced stockouts
- Reduced overstocking

""")

st.divider()

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("---")

st.caption(
    "Sales Forecasting & Demand Intelligence System | Product Demand Segmentation | Developed by Sravya Velaga"
)