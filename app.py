import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# MONKEY PATCH FOR PYTHON 3.14 + ASTOR
if not hasattr(ast, 'Str'):
    ast.Str = ast.Constant
    ast.Num = ast.Constant
    ast.NameConstant = ast.Constant

from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
import google.generativeai as genai

#Config
st.set_page_config(page_title="Retail Analytics", layout="wide")
st.title("Supermarket Sales Analytics Dashboard")
st.markdown("_Dilan's Prototype built for retail demand analysis._")

#Load Data
# @st.cache_data
# def load_data():
#     return pd.read_csv("supermarket_sales.csv")
#
# df = load_data()

# Load Data - Neon Cloud DB
@st.cache_data(ttl=600)  # Caches data for 10 minutes to optimize database hits
def load_data():
    # DATABASE_URL from secrets
    conn = st.connection("postgresql", type="sql", url=st.secrets["DATABASE_URL"])

    df = conn.query("SELECT * FROM sales;")

    # Map sql columns
    df.columns = ['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender',
                  'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Total',
                  'Date', 'Time', 'Payment', 'cogs', 'gross margin percentage',
                  'gross income', 'Rating']
    return df


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
    st.plotly_chart(fig_product, width="stretch")

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
    st.plotly_chart(fig_payment, width="stretch")

#Raw Data Table
st.markdown("### Raw Data View")
st.dataframe(df_selection)

st.markdown("---")
st.markdown("### AI Demand Forecasting (Machine Learning)")

# Prepare Data for ML Model
df["Date"] = pd.to_datetime(df["Date"])
daily_sales = df.groupby("Date")["Total"].sum().reset_index()

#  Day Index - represent time passing
daily_sales["Day_Index"] = np.arange(len(daily_sales))

X = daily_sales[["Day_Index"]] # Features
y = daily_sales["Total"]       # Target Var

# Train ML Model
model = LinearRegression()
model.fit(X, y)

# Predict Next 7Days
last_day_index = daily_sales["Day_Index"].max()
last_date = daily_sales["Date"].max()

#future data points
future_days = pd.DataFrame({"Day_Index": np.arange(last_day_index + 1, last_day_index + 8)})
predictions = model.predict(future_days)

#DataFrame for predictions
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Total": predictions,
    "Type": "Forecast (AI)"
})

#historical data
daily_sales["Type"] = "Historical"

#past and future predictions
combined_df = pd.concat([daily_sales[["Date", "Total", "Type"]], forecast_df])

#AI Forecast
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
st.plotly_chart(fig_forecast, width="stretch")

#business insight
st.success(f" **Model Insight:** The Linear Regression model projects total revenue for the next 7 days to average around LKR {int(predictions.mean()):,} per day. This can be used to optimize supply chain logistics.")

st.markdown("---")
st.markdown("### AI Customer Segmentation (K-Means Clustering)")
st.markdown("_Using Decoupled Unsupervised ML to identify distinct purchasing personas._")

# Pre-Trained Model and Scaler
@st.cache_resource
def load_models():
    loaded_kmeans = joblib.load("kmeans_model.pkl")
    loaded_scaler = joblib.load("scaler.pkl")
    return loaded_kmeans, loaded_scaler

kmeans, scaler = load_models()

# Prepare data
features = ['Unit price', 'Quantity', 'Total', 'Rating']
X_cluster = df[features]

# Predict
X_scaled = scaler.transform(X_cluster)
df['Cluster'] = kmeans.predict(X_scaled)

# mathematical clusters to  labels
segment_map = {
    0: "Budget/Routine Shoppers",
    1: "High-Spend Premium",
    2: "High-Volume Bulk Buyers"
}
df['Customer Segment'] = df['Cluster'].map(segment_map)

# Visualize the Clusters
fig_cluster = px.scatter(
    df,
    x="Total",
    y="Rating",
    color="Customer Segment",
    title="Transaction Segmentation: Spend vs. Customer Satisfaction",
    template="plotly_dark",
    hover_data=["Product line", "Gender", "City"]
)

fig_cluster.update_traces(marker=dict(size=8, opacity=0.8))
st.plotly_chart(fig_cluster, width="stretch")

st.success("**Strategic Action:** The model identified three distinct shopper segments. 'High-Spend Premium' transactions correlate with specific product lines. This cluster can be targeted with loyalty program (Nexus) upgrades to increase retention.")

st.markdown("---")
st.markdown("### GenAI Data Assistant (PandasAI & Gemini)")
st.markdown("_Ask questions about the retail data in plain English._")

# Initialize Gemini LLM
api_key = st.secrets["GEMINI_API_KEY"]
llm = GoogleGemini(api_key=api_key)

# Google core library overwriting PandasAI engine
import google.generativeai as genai
llm.google_gemini = genai.GenerativeModel('gemini-2.5-flash')

# Convert to AI DataFrame
sdf = SmartDataframe(df, config={"llm": llm})

# ui
st.markdown("> **Try asking:** _'Which branch has the highest average rating?'_ or _'What is the most popular payment method?'_")
user_question = st.text_input("What would you like to know about the sales data?")

if user_question:
    with st.spinner("Gemini is analyzing the database..."):
        try:
            answer = sdf.chat(user_question)
            st.success(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")