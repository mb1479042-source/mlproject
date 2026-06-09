# FILE NAME: model_training.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("🔄 Step 1: Loading Dataset...")
try:
    df = pd.read_csv("ecommerce_reviews.csv", encoding='latin1') 
except FileNotFoundError:
    print("❌ Error: 'ecommerce_reviews.csv' file-ah kandupika mudiyala!")
    exit()

print("🧹 Step 2: Data Preprocessing for Retail Analysis...")
# Null values-ah remove panrom
df = df.dropna(subset=['CustomerID'])
# Quantity, Price columns-la positive values mattum edukrom
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Total Amount calculation
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

print("🎯 Step 3: Feature Engineering (RFM Analytics)...")
# InvoiceDate calculation (Reference date vachi days find panrom)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Recency, Frequency, Monetary metrics create panrom
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'count',                                   # Frequency
    'TotalAmount': 'sum'                                    # Monetary
})

# Columns rename panrom
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalAmount': 'Monetary'
}, inplace=True)

print("🔢 Step 4: Outlier Treatment & Scaling...")
# Values extreme variations handle panna log transform & scaling panrom
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

print("🚀 Step 5: Machine Learning Model Training (K-Means Clustering)...")
# Target: Customers-ah 3 tier groups-ah split panrom (High value, Medium, Low value)
kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_model.fit(rfm_scaled)

# Cluster tags-ah actual data-la add panrom
rfm['Cluster'] = kmeans_model.labels_

print("\n🎉 SUCCESS! Customer Segments Created:")
print(rfm.groupby('Cluster').mean())

print("\n📦 Step 6: Exporting Segmentation Model Files...")
joblib.dump(kmeans_model, "sentiment_model.pkl") # Standard app mapping-kaga intha name
joblib.dump(scaler, "tfidf_vectorizer.pkl")       # Standard app scaler configuration
print("✅ Everything Done! Model outputs successfully saved.")
