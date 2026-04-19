import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

print("Starting model training pipeline...")

# Load Data
df = pd.read_csv("supermarket_sales.csv")

# Features & Scale
features = ['Unit price', 'Quantity', 'Total', 'Rating']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Train Model
print("Training K-Means algorithm...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 4. Serialize and Save
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Success! Model and Scaler saved as .pkl files.")