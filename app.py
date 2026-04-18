import streamlit as st
import pandas as pd
import plotly.express as px

#Config
st.set_page_config(page_title="Retail Analytics", layout="wide")
st.title("Supermarket Sales Analytics Dashboard")
st.markdown("_Dilan's Prototype built for retail demand analysis._")

#Load Data
@st.cache_data
def load_data():
    return pd.read_csv("supermarket_sales.csv")

df = load_data()

#Sidebar filtering
st.sidebar.header("Filter Data")
branch = st.sidebar.multiselect(
    "Select Branch:",
    options=df["Branch"].unique(),
    default=df["Branch"].unique()
)

df_selection = df.query("Branch == @branch")

#Top KPIs
total_sales = int(df_selection["Total"].sum())
average_rating = round(df_selection["Rating"].mean(), 1)

col1, col2 = st.columns(2)
with col1:
    st.info(f"💰 **Total Revenue:** LKR {total_sales:,}")
with col2:
    st.info(f"⭐ **Average Customer Rating:** {average_rating}")

st.markdown("---")

#Plotly visuals
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Bar Chart
    sales_by_product = df_selection.groupby("Product line")["Total"].sum().reset_index()
    fig_product = px.bar(
        sales_by_product,
        x="Total",
        y="Product line",
        orientation="h",
        title="Revenue by Product Category"
    )
    st.plotly_chart(fig_product, use_container_width=True)

with chart_col2:
    # Pie Chart
    sales_by_payment = df_selection.groupby("Payment")["Total"].sum().reset_index()
    fig_payment = px.pie(
        sales_by_payment,
        values="Total",
        names="Payment",
        title="Revenue by Payment Method",
        hole=0.4
    )
    st.plotly_chart(fig_payment, use_container_width=True)

#Raw Data Table
st.markdown("### Raw Data View")
st.dataframe(df_selection)