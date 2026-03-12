import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---- Load pre-trained objects ----
with open("top_features.pkl", "rb") as f:
    top_features = joblib.load(f)

with open("imputer_deploy.pkl", "rb") as f:
    imputer = joblib.load(f)

with open("scaler_deploy.pkl", "rb") as f:
    scaler = joblib.load(f)

with open("ensemble_model_deploy.pkl", "rb") as f:
    model = joblib.load(f)

# ---- Streamlit UI ----
st.title("Ensemble Model Prediction App")

st.write("Enter the input values for prediction:")

# Dynamically create input fields for each feature
user_input = {}
for feature in top_features:
    user_input[feature] = st.text_input(f"{feature}:", "")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input], columns=top_features)

# Convert numeric features to float
for col in input_df.columns:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

# Predict button
if st.button("Predict"):
    # Handle missing values
    input_imputed = imputer.transform(input_df)
    # Scale the input
    input_scaled = scaler.transform(input_imputed)
    # Convert the transformed NumPy array back to a DataFrame with the correct column names
    processed_input = pd.DataFrame(input_scaled, columns=top_features)
    # Make prediction
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)

    st.success(f"Predicted value: {prediction[0]}")
    st.success(f"Probality of risk: {round(probability[0][1], 3)}")
