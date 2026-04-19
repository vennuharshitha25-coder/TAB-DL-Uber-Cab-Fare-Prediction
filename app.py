import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved files
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Uber/Ola Fare Prediction", layout="centered")

st.title("🚖 Uber / Ola Fare Prediction")
st.write("Enter trip details to predict estimated fare.")

# Input fields
pickup_latitude = st.number_input("Pickup Latitude", value=17.3850, format="%.6f")
pickup_longitude = st.number_input("Pickup Longitude", value=78.4867, format="%.6f")

dropoff_latitude = st.number_input("Dropoff Latitude", value=17.4500, format="%.6f")
dropoff_longitude = st.number_input("Dropoff Longitude", value=78.3900, format="%.6f")

passenger_count = st.slider("Passenger Count", 1, 6, 1)
distance_km = st.number_input("Distance (km)", min_value=1.0, value=5.0)

hour = st.slider("Hour of Day", 0, 23, 10)
weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, 2)
is_weekend = st.selectbox("Weekend?", [0, 1])

surge_multiplier = st.slider("Surge Multiplier", 1.0, 3.0, 1.0)

traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
demand_level = st.selectbox("Demand Level", ["Low", "Medium", "High"])

ride_category = st.selectbox("Ride Category", ["Mini", "Prime", "Sedan", "SUV"])
weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"])

# Predict button
if st.button("Predict Fare"):

    # Convert text inputs using encoders
    input_dict = {
        "pickup_latitude": pickup_latitude,
        "pickup_longitude": pickup_longitude,
        "dropoff_latitude": dropoff_latitude,
        "dropoff_longitude": dropoff_longitude,
        "passenger_count": passenger_count,
        "distance_km": distance_km,
        "hour": hour,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "surge_multiplier": surge_multiplier,
        "traffic_level": traffic_level,
        "demand_level": demand_level,
        "ride_category": ride_category,
        "weather": weather
    }

    row = []

    for col in features:
        value = input_dict[col]

        if col in encoders:
            le = encoders[col]

            # handle unseen labels safely
            if value not in le.classes_:
                value = le.classes_[0]

            value = le.transform([value])[0]

        row.append(value)

    data = pd.DataFrame([row], columns=features)

    prediction = model.predict(data)[0]

    st.success(f"💰 Estimated Fare: ₹ {round(prediction,2)}")