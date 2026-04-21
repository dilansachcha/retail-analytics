# 🛒 Supermarket Sales Analytics & ML Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dilan-retail-analytics.streamlit.app/)

An end-to-end, cloud-native interactive data dashboard engineered to bridge the gap between raw retail data and actionable business intelligence. This application goes beyond standard Exploratory Data Analysis (EDA) by integrating decoupled Machine Learning models and a Generative AI natural language assistant.

## Key Features

* **Cloud Data Engineering:** Real-time data ingestion from a serverless PostgreSQL database (Neon DB) replacing static CSV files.
* **Interactive Dashboards:** Dynamic filtering and visualizations utilizing Plotly and Pandas to uncover revenue trends by product line and payment methods.
* **Predictive Machine Learning:** * **Demand Forecasting:** A Scikit-Learn Linear Regression model projecting 7-day future sales to optimize inventory and supply chain logistics.
  * **Customer Segmentation:** An unsupervised K-Means Clustering model that categorizes transactions into distinct shopper personas (e.g., Budget, Premium, Bulk) to drive targeted loyalty programs.
* **GenAI Data Assistant:** Integrated **PandasAI** with the **Google Gemini 2.5 Flash API**, allowing non-technical users to query the database and generate insights using plain English.

## Tech Stack

* **Frontend:** Streamlit, Plotly
* **Data Engineering:** PostgreSQL (Neon DB), SQLAlchemy, Pandas
* **Machine Learning:** Scikit-Learn, NumPy, Joblib
* **Generative AI:** Google Gemini API, PandasAI
* **DevOps / Deployment:** Git, Streamlit Cloud (Environment & Dependency Management)

## Local Execution

If you wish to run this project locally, you will need to set up your own database and API keys.

1. Clone the repository:
   ```bash
   git clone [https://github.com/dilansachcha/retail-analytics.git](https://github.com/dilansachcha/retail-analytics.git)
   ```

2. Install the strict dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.streamlit/secrets.toml` file in the root directory and add your credentials:
   ```toml
   DATABASE_URL = "postgresql://<user>:<password>@<host>/<dbname>"
   GEMINI_API_KEY = "your_google_gemini_api_key"
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Production Configuration

* **Dependency Management:** The production environment utilizes strict version pinning (`pandasai==2.2.1`, `numpy<2.0.0`) to ensure binary compatibility between legacy ML libraries and modern cloud Linux environments.
* **Python Compatibility:** Implemented custom monkey patches (`ast.Str = ast.Constant`) to bridge PandasAI execution with cutting-edge Python 3.14 AST deprecations.
