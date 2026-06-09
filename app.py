# FILE NAME: app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# CSS Styles for Premium Look
st.set_page_config(page_title="AI Customer Segmentation Engine", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #1E3A8A; color: white; border-radius: 6px; width: 100%; height: 3rem; font-size: 1.1rem; }
    .card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# 📦 Loading ML Saved Models
try:
    kmeans = joblib.load("sentiment_model.pkl")
    scaler = joblib.load("tfidf_vectorizer.pkl")
except:
    st.error("❌ Model files (`.pkl`) backend folder-la illa! Please check.")
    st.stop()

# Header Layout
st.title("📊 AI-Driven E-Commerce Customer Segmentation Engine")
st.write("Predict value groups and customer tiers instantly using advanced RFM Machine Learning analytics.")
st.markdown("---")

# Layout Columns Splitting
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='card'><h3>📥 Input Metrics</h3>", unsafe_allow_html=True)
    
    recency = st.number_input("Recency (Days since last purchase):", min_value=1, max_value=365, value=30)
    frequency = st.number_input("Frequency (Total transactions done):", min_value=1, max_value=1000, value=15)
    monetary = st.number_input("Monetary Value (Total amount spent in ₹/$):", min_value=1.0, max_value=100000.0, value=500.0)
    
    submit = st.button("Analyze Segment Cluster")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'><h3>🎯 Diagnostic Analytics & Group Predictions</h3>", unsafe_allow_html=True)
    
    if submit:
        # Preprocessing user live inputs
        input_data = np.array([[recency, frequency, monetary]])
        input_log = np.log1p(input_data)
        input_scaled = scaler.transform(input_log)
        
        # Predicting cluster tag value
        cluster_pred = kmeans.predict(input_scaled)[0]
        
        # Mapping metrics to cluster groups
        if cluster_pred == 2:
            st.success("💎 ** Tier 1 Customer: Core High-Value User**")
            st.info("💡 **Strategy:** VIP Offers, Priority shipping, Loyalty clubs, Premium preview accessibility updates.")
        elif cluster_pred == 1:
            st.warning("⚡ **Tier 2 Customer: Medium-Value Casual Shopper**")
            st.info("💡 **Strategy:** Bulk purchase discounts, Personalized dynamic coupon codes, Engagement notifications triggers.")
        else:
            st.error("📉 **Tier 3 Customer: At-Risk / Churned / Low-Value User**")
            st.info("💡 **Strategy:** Win-back reactivation email triggers, Heavily slashed clearance discounts campaigns.")
            
        # Metric Grid metrics preview
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Recency Days", f"{recency} Days")
        m_col2.metric("Order Frequency", f"{frequency} Orders")
        m_col3.metric("Total Spends", f"${monetary:.2f}")
    else:
        st.write("👈 Form metrics-ah click panni 'Analyze Segment Cluster' button click panna parameters graph values display aagum.")
    
    st.markdown("</div>", unsafe_allow_html=True)
