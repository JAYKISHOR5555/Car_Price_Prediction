import streamlit as st
import numpy as np
from car_price_model import load_model

st.title("🚗 Car Price Predictor App")

# Load trained model
model = load_model()

# Input fields
age = st.slider("Car Age (Years)", 1, 10, 3)
mileage = st.slider("Mileage (in km)", 10000, 100000, 30000, step=5000)
engine = st.selectbox("Engine Size (L)", [1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2])
doors = st.selectbox("Number of Doors", [2, 4])

# Predict button
if st.button("Predict Price"):
    features = np.array([[age, mileage, engine, doors]])
    price = model.predict(features)[0]
    st.success(f"💰 Estimated Car Price: ₹{int(price):,}")
