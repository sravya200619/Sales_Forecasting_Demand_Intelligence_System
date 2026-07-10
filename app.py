import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    # Convert dates (DD/MM/YYYY)
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

    # Remove rows with invalid dates
    df = df.dropna(subset=["Order Date", "Ship Date"])

    return df


df = load_data()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("📈 Sales Forecasting")

st.sidebar.markdown("---")

st.sidebar.success("Navigation")

st.sidebar.info("""
Use the pages on the left:

• Executive Dashboard

• Forecast Explorer

• Anomaly Report

• Product Segmentation

• Model Comparison
""")

# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")

st.markdown("""
### Retail Business Intelligence Dashboard

This dashboard provides:

- Sales Analytics
- Demand Forecasting
- Time Series Analysis
- Product Segmentation
- Sales Anomaly Detection
- Business Insights
""")

st.divider()

# -------------------------------------------------
# KPI Cards
# -------------------------------------------------

total_sales = df["Sales"].sum()

total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else 0

customers = df["Customer ID"].nunique() if "Customer ID" in df.columns else 0

products = df["Product ID"].nunique() if "Product ID" in df.columns else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sales", f"${total_sales:,.2f}")

with col2:
    st.metric("Orders", total_orders)

with col3:
    st.metric("Customers", customers)

with col4:
    st.metric("Products", products)

st.divider()

# -------------------------------------------------
# Dataset Preview
# -------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

# -------------------------------------------------
# Dataset Summary
# -------------------------------------------------

col1, col2 = st.columns(2)

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

    if "Category" in df.columns:
        st.write(df["Category"].unique())
    else:
        st.write("Category column not found.")

st.divider()

st.success("✅ Select a page from the left sidebar to continue.")

st.caption(
    "Developed using Streamlit | XGBoost | Prophet | SARIMA | Scikit-Learn"
)