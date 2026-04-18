import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression # NEW
import numpy as np # NEW

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

st.markdown("---")
st.markdown("### 🤖 AI Demand Forecasting (Machine Learning)")

# 1. Feature Engineering: Prepare Data for the ML Model
df["Date"] = pd.to_datetime(df["Date"])
daily_sales = df.groupby("Date")["Total"].sum().reset_index()

# Create a 'Day Index' (e.g., 1, 2, 3...) to represent time passing
daily_sales["Day_Index"] = np.arange(len(daily_sales))

X = daily_sales[["Day_Index"]] # Features
y = daily_sales["Total"]       # Target Variable

# 2. Train the Machine Learning Model
model = LinearRegression()
model.fit(X, y)

# 3. Predict the Next 7 Days
last_day_index = daily_sales["Day_Index"].max()
last_date = daily_sales["Date"].max()

# Create future data points
future_days = pd.DataFrame({"Day_Index": np.arange(last_day_index + 1, last_day_index + 8)})
predictions = model.predict(future_days)

# Create a DataFrame for the predictions
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Total": predictions,
    "Type": "Forecast (AI)"
})

# Add a label to historical data
daily_sales["Type"] = "Historical"

# Combine past data and future predictions
combined_df = pd.concat([daily_sales[["Date", "Total", "Type"]], forecast_df])

# 4. Visualize the AI Forecast
fig_forecast = px.line(
    combined_df,
    x="Date",
    y="Total",
    color="Type",
    title="7-Day Revenue Forecast vs Historical Trend",
    template="plotly_dark",
    color_discrete_map={"Historical": "#3498db", "Forecast (AI)": "#e74c3c"}
)

fig_forecast.update_traces(line=dict(width=3))
st.plotly_chart(fig_forecast, use_container_width=True)

# 5. Business Insight
st.success(f" **Model Insight:** The Linear Regression model projects total revenue for the next 7 days to average around LKR {int(predictions.mean()):,} per day. Use this to optimize supply chain logistics.")