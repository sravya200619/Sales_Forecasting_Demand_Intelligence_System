import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Executive Sales Dashboard")

st.markdown("""
This dashboard provides an executive overview of the Superstore sales data.
""")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    # -------------------------------
    # DATE CONVERSION
    # -------------------------------

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=["Order Date", "Ship Date"])

    # -------------------------------
    # CREATE DATE FEATURES
    # -------------------------------

    df["Year"] = df["Order Date"].dt.year

    df["Month"] = df["Order Date"].dt.month_name()

    df["Month Number"] = df["Order Date"].dt.month

    df["Quarter"] = df["Order Date"].dt.quarter

    df["Week"] = df["Order Date"].dt.isocalendar().week.astype(int)

    df["Day"] = df["Order Date"].dt.day_name()

    # -------------------------------
    # ADD OPTIONAL COLUMNS
    # -------------------------------

    if "Profit" not in df.columns:
        df["Profit"] = 0

    if "Quantity" not in df.columns:
        df["Quantity"] = 1

    if "Discount" not in df.columns:
        df["Discount"] = 0

    # -------------------------------
    # FILL MISSING VALUES
    # -------------------------------

    numeric_cols = df.select_dtypes(include=np.number).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)

    object_cols = df.select_dtypes(include="object").columns

    df[object_cols] = df[object_cols].fillna("Unknown")

    return df


df = load_data()

# ----------------------------------------------------
# CHECK REQUIRED COLUMNS
# ----------------------------------------------------

required_columns = [
    "Sales",
    "Region",
    "Category",
    "State",
    "Product Name"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ----------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------

st.sidebar.header("🔍 Filters")

regions = sorted(df["Region"].dropna().unique().tolist())
categories = sorted(df["Category"].dropna().unique().tolist())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    options=regions,
    default=regions
)

selected_categories = st.sidebar.multiselect(
    "Select Category",
    options=categories,
    default=categories
)

filtered_df = df[
    (df["Region"].isin(selected_regions)) &
    (df["Category"].isin(selected_categories))
].copy()

if filtered_df.empty:
    st.warning("No records found for the selected filters.")
    st.stop()

# ----------------------------------------------------
# KPI CARDS
# ----------------------------------------------------

total_sales = filtered_df["Sales"].sum()

total_orders = (
    filtered_df["Order ID"].nunique()
    if "Order ID" in filtered_df.columns else 0
)

total_customers = (
    filtered_df["Customer ID"].nunique()
    if "Customer ID" in filtered_df.columns else 0
)

total_products = (
    filtered_df["Product ID"].nunique()
    if "Product ID" in filtered_df.columns else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Sales",
    f"${total_sales:,.2f}"
)

col2.metric(
    "🛒 Orders",
    f"{total_orders:,}"
)

col3.metric(
    "👥 Customers",
    f"{total_customers:,}"
)

col4.metric(
    "📦 Products",
    f"{total_products:,}"
)

st.divider()

# ----------------------------------------------------
# SALES BY YEAR
# ----------------------------------------------------

st.subheader("📅 Total Sales by Year")

yearly_sales = (
    filtered_df
    .groupby("Year", as_index=False)["Sales"]
    .sum()
)

fig_year = px.bar(
    yearly_sales,
    x="Year",
    y="Sales",
    color="Sales",
    text_auto=".2s",
    template="plotly_white"
)

fig_year.update_layout(
    xaxis_title="Year",
    yaxis_title="Sales"
)

st.plotly_chart(fig_year, use_container_width=True)

st.divider()

# ----------------------------------------------------
# MONTHLY SALES TREND
# ----------------------------------------------------

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby(["Year", "Month Number", "Month"], as_index=False)["Sales"]
    .sum()
    .sort_values(["Year", "Month Number"])
)

fig_month = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    color="Year",
    markers=True,
    template="plotly_white"
)

fig_month.update_layout(
    xaxis_title="Month",
    yaxis_title="Sales"
)

st.plotly_chart(fig_month, use_container_width=True)

st.divider()

# ----------------------------------------------------
# REGION SALES
# ----------------------------------------------------

st.subheader("🌍 Sales by Region")

region_sales = (
    filtered_df
    .groupby("Region", as_index=False)["Sales"]
    .sum()
)

fig_region = px.pie(
    region_sales,
    names="Region",
    values="Sales",
    hole=0.45
)

st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ----------------------------------------------------
# CATEGORY SALES
# ----------------------------------------------------

st.subheader("📦 Sales by Category")

category_sales = (
    filtered_df
    .groupby("Category", as_index=False)["Sales"]
    .sum()
)

fig_category = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    color="Category",
    text_auto=".2s",
    template="plotly_white"
)

st.plotly_chart(fig_category, use_container_width=True)

st.divider()

# ----------------------------------------------------
# TOP 10 PRODUCTS
# ----------------------------------------------------

st.subheader("🏆 Top 10 Best Selling Products")

top_products = (
    filtered_df
    .groupby("Product Name", as_index=False)["Sales"]
    .sum()
    .sort_values("Sales", ascending=False)
    .head(10)
)

fig_products = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    color="Sales",
    text_auto=".2s",
    template="plotly_white"
)

fig_products.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(fig_products, use_container_width=True)

st.divider()

# ----------------------------------------------------
# TOP STATES
# ----------------------------------------------------

if "State" in filtered_df.columns:

    st.subheader("🏙️ Top 10 States by Sales")

    top_states = (
        filtered_df
        .groupby("State", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    fig_states = px.bar(
        top_states,
        x="State",
        y="Sales",
        color="Sales",
        text_auto=".2s",
        template="plotly_white"
    )

    st.plotly_chart(fig_states, use_container_width=True)

st.divider()

# ----------------------------------------------------
# TOP CUSTOMERS
# ----------------------------------------------------

if "Customer Name" in filtered_df.columns:

    st.subheader("👥 Top 10 Customers")

    top_customers = (
        filtered_df
        .groupby("Customer Name", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    st.dataframe(
        top_customers,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ----------------------------------------------------
# SALES SUMMARY
# ----------------------------------------------------

st.subheader("📋 Sales Summary")

summary = (
    filtered_df
    .groupby(["Category", "Region"], as_index=False)["Sales"]
    .sum()
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ----------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------

st.subheader("📥 Download Filtered Dataset")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "filtered_sales.csv",
    "text/csv"
)

st.divider()

# ----------------------------------------------------
# EXECUTIVE INSIGHTS
# ----------------------------------------------------

st.subheader("📌 Executive Insights")

highest_region = region_sales.loc[
    region_sales["Sales"].idxmax()
]

highest_category = category_sales.loc[
    category_sales["Sales"].idxmax()
]

highest_product = top_products.iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Highest Revenue Region",
        highest_region["Region"],
        f"${highest_region['Sales']:,.2f}"
    )

with col2:
    st.metric(
        "Best Category",
        highest_category["Category"],
        f"${highest_category['Sales']:,.2f}"
    )

with col3:
    st.metric(
        "Top Product",
        highest_product["Product Name"],
        f"${highest_product['Sales']:,.2f}"
    )

st.divider()

# ----------------------------------------------------
# BUSINESS RECOMMENDATIONS
# ----------------------------------------------------

st.subheader("📈 Business Recommendations")

st.info(
"""
### Recommendations

• Increase inventory allocation for the highest revenue category.

• Prioritize stock replenishment in the highest-performing region.

• Promote top-selling products through targeted campaigns.

• Monitor low-performing regions for improvement opportunities.

• Use demand forecasts to reduce stock-outs and excess inventory.

• Continue monitoring anomalies for better demand planning.
"""
)

st.divider()

# ----------------------------------------------------
# DATASET PREVIEW
# ----------------------------------------------------

with st.expander("📄 View Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

st.divider()

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

### Sales Forecasting & Demand Intelligence System

Developed by <b>Sravya Velaga</b>

Machine Learning | Time Series Forecasting | Streamlit Dashboard

</div>
""",
unsafe_allow_html=True
)