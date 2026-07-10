import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Executive Sales Dashboard")

st.markdown(
"""
This dashboard provides an executive overview of the Superstore sales data.
"""
)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    df["Year"] = df["Order Date"].dt.year

    df["Month"] = df["Order Date"].dt.month_name()

    df["Month Number"] = df["Order Date"].dt.month

    df["Quarter"] = df["Order Date"].dt.quarter

    return df


df = load_data()

# ----------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------

st.sidebar.header("Filters")

regions = st.sidebar.multiselect(

    "Select Region",

    options=sorted(df["Region"].unique()),

    default=sorted(df["Region"].unique())

)

categories = st.sidebar.multiselect(

    "Select Category",

    options=sorted(df["Category"].unique()),

    default=sorted(df["Category"].unique())

)

filtered_df = df[

    (df["Region"].isin(regions)) &

    (df["Category"].isin(categories))

]

# ----------------------------------------------------
# KPI CARDS
# ----------------------------------------------------

total_sales = filtered_df["Sales"].sum()

total_orders = filtered_df["Order ID"].nunique()

total_customers = filtered_df["Customer ID"].nunique()

total_products = filtered_df["Product ID"].nunique()

col1,col2,col3,col4 = st.columns(4)

col1.metric(

    "💰 Total Sales",

    f"${total_sales:,.0f}"

)

col2.metric(

    "🛒 Orders",

    total_orders

)

col3.metric(

    "👥 Customers",

    total_customers

)

col4.metric(

    "📦 Products",

    total_products

)

st.divider()

# ----------------------------------------------------
# SALES BY YEAR
# ----------------------------------------------------

st.subheader("📅 Sales by Year")

yearly_sales = (

    filtered_df

    .groupby("Year")["Sales"]

    .sum()

    .reset_index()

)

fig_year = px.bar(

    yearly_sales,

    x="Year",

    y="Sales",

    text_auto=".2s",

    color="Sales",

    title="Total Sales by Year"

)

fig_year.update_layout(

    xaxis_title="Year",

    yaxis_title="Sales",

    template="plotly_white"

)

st.plotly_chart(fig_year, use_container_width=True)

st.divider()

# ----------------------------------------------------
# MONTHLY SALES TREND
# ----------------------------------------------------

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (

    filtered_df

    .groupby(["Year","Month Number","Month"])["Sales"]

    .sum()

    .reset_index()

    .sort_values(["Year","Month Number"])

)

fig_month = px.line(

    monthly_sales,

    x="Month",

    y="Sales",

    color="Year",

    markers=True,

    title="Monthly Sales Trend"

)

fig_month.update_layout(

    xaxis_title="Month",

    yaxis_title="Sales",

    template="plotly_white"

)

st.plotly_chart(fig_month, use_container_width=True)

st.divider()

# ----------------------------------------------------
# REGION SALES
# ----------------------------------------------------

st.subheader("🌍 Sales by Region")

region_sales = (

    filtered_df

    .groupby("Region")["Sales"]

    .sum()

    .reset_index()

)

fig_region = px.pie(

    region_sales,

    names="Region",

    values="Sales",

    hole=0.45,

    title="Regional Sales Distribution"

)

st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ----------------------------------------------------
# CATEGORY SALES
# ----------------------------------------------------

st.subheader("📦 Category Sales")

category_sales = (

    filtered_df

    .groupby("Category")["Sales"]

    .sum()

    .reset_index()

)

fig_category = px.bar(

    category_sales,

    x="Category",

    y="Sales",

    color="Category",

    text_auto=".2s",

    title="Sales by Category"

)

fig_category.update_layout(

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

    .groupby("Product Name")["Sales"]

    .sum()

    .sort_values(ascending=False)

    .head(10)

    .reset_index()

)

fig_products = px.bar(

    top_products,

    x="Sales",

    y="Product Name",

    orientation="h",

    color="Sales",

    text_auto=".2s",

    title="Top 10 Products by Sales"

)

fig_products.update_layout(

    template="plotly_white",

    yaxis={"categoryorder":"total ascending"}

)

st.plotly_chart(fig_products, use_container_width=True)

st.divider()

# ----------------------------------------------------
# TOP STATES
# ----------------------------------------------------

st.subheader("🏙️ Top 10 States by Sales")

top_states = (

    filtered_df

    .groupby("State")["Sales"]

    .sum()

    .sort_values(ascending=False)

    .head(10)

    .reset_index()

)

fig_states = px.bar(

    top_states,

    x="State",

    y="Sales",

    color="Sales",

    text_auto=".2s",

    title="Top States"

)

fig_states.update_layout(template="plotly_white")

st.plotly_chart(fig_states, use_container_width=True)

st.divider()

# ----------------------------------------------------
# TOP CUSTOMERS
# ----------------------------------------------------

st.subheader("👥 Top 10 Customers")

top_customers = (

    filtered_df

    .groupby("Customer Name")["Sales"]

    .sum()

    .sort_values(ascending=False)

    .head(10)

    .reset_index()

)

st.dataframe(

    top_customers,

    use_container_width=True

)

st.divider()

# ----------------------------------------------------
# SALES SUMMARY TABLE
# ----------------------------------------------------

st.subheader("📋 Sales Summary")

summary = filtered_df.groupby(

    ["Category","Region"]

)["Sales"].sum().reset_index()

st.dataframe(

    summary,

    use_container_width=True

)

st.divider()

# ----------------------------------------------------
# DOWNLOAD FILTERED DATA
# ----------------------------------------------------

st.subheader("📥 Download Filtered Dataset")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(

    label="Download CSV",

    data=csv,

    file_name="filtered_sales.csv",

    mime="text/csv"

)

st.divider()

# ----------------------------------------------------
# BUSINESS INSIGHTS
# ----------------------------------------------------

st.subheader("📌 Executive Insights")

highest_region = region_sales.sort_values(
    "Sales",
    ascending=False
).iloc[0]

highest_category = category_sales.sort_values(
    "Sales",
    ascending=False
).iloc[0]

highest_product = top_products.iloc[0]

st.success(

f"""
✅ Highest Revenue Region: **{highest_region['Region']}**

💰 Sales: **${highest_region['Sales']:,.2f}**

📦 Best Performing Category: **{highest_category['Category']}**

💰 Sales: **${highest_category['Sales']:,.2f}**

🏆 Best Selling Product:

**{highest_product['Product Name']}**

💰 Sales: **${highest_product['Sales']:,.2f}**
"""

)

st.info("""

### Business Recommendations

• Increase inventory for high-performing categories.

• Focus marketing efforts on the best-selling products.

• Expand operations in high-revenue regions.

• Monitor low-performing categories for optimization.

• Use forecasting results to improve stock planning.

""")

st.divider()

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.markdown("---")

st.caption(

"Developed by Sravya Velaga | Sales Forecasting & Demand Intelligence System"

)