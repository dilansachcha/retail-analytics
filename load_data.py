import pandas as pd
from sqlalchemy import create_engine
import toml
import os

print("Starting ETL Pipeline...")

# secret URL - secrets.toml
secrets_path = os.path.join(".streamlit", "secrets.toml")
secrets = toml.load(secrets_path)
db_url = secrets["DATABASE_URL"]

# Load CSV
print("Extracting data from CSV...")
df = pd.read_csv("supermarket_sales.csv")

# Clean up clmn names
df.columns = [c.lower().replace(' ', '_') for c in df.columns]

# push data to neon
print("Pushing data to Neon Cloud Database...")
engine = create_engine(db_url)

# 1000 row table
df.to_sql('sales', engine, if_exists='replace', index=False)

print("ETL Complete! Your data is now in the cloud.")