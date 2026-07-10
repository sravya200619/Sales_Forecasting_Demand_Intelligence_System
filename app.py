import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df

df = load_data()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart--v1.png",
    width=80
)

st.sidebar.title("Sales Forecasting")

st.sidebar.markdown("---")

st.sidebar.success("Navigation")

st.sidebar.info(
"""
Use the pages on the left to explore:

• Executive Dashboard

• Forecast Explorer

• Anomaly Report

• Product Segmentation

• Model Comparison
"""
)

# -------------------------
# Main Page
# -------------------------

st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")

st.markdown("""
### Retail Business Intelligence Dashboard

This project provides an end-to-end analytics solution for retail sales forecasting.

The application includes:

- Sales Analytics
- Demand Forecasting
- Time Series Analysis
- Product Demand Segmentation
- Sales Anomaly Detection
- Business Insights
""")

st.divider()

# -------------------------
# KPI Cards
# -------------------------

total_sales = df["Sales"].sum()

total_orders = df["Order ID"].nunique()

customers = df["Customer ID"].nunique()

products = df["Product ID"].nunique()

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "Orders",
    total_orders
)

col3.metric(
    "Customers",
    customers
)

col4.metric(
    "Products",
    products
)

st.divider()

# -------------------------
# Dataset Preview
# -------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

# -------------------------
# Dataset Summary
# -------------------------

col1,col2 = st.columns(2)

with col1:

    st.subheader("Dataset Information")

    st.write(f"Rows : {df.shape[0]}")

    st.write(f"Columns : {df.shape[1]}")

    st.write(
        f"Date Range : {df['Order Date'].min().date()} "
        f"to {df['Order Date'].max().date()}"
    )

with col2:

    st.subheader("Available Categories")

    st.write(df["Category"].unique())

st.divider()

st.success("Select a page from the left sidebar to continue.")

st.caption("Developed using Streamlit • XGBoost • Prophet • SARIMA • Scikit-Learn")