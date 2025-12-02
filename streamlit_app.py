import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models (each may be pure model or dict)
lassa = joblib.load("lassa.joblib")
cholera = joblib.load("cholera.joblib")
measles = joblib.load("measles.joblib")
yellow_fever = joblib.load("yellow-fever.joblib")

models = {
    "Lassa Fever": lassa,
    "Cholera": cholera,
    "Measles": measles,
    "Yellow Fever": yellow_fever
}

# Function: safely extract top 10 features
def get_top_features(model):
    # Case 1: Model saved as dictionary
    if isinstance(model, dict) and "features" in model:
        return model["features"]

    # Case 2: LightGBM / XGBoost / CatBoost model with feature names
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        return model.feature_name_[:10]

    # Case 3: No feature information
    return []

# Function: safely extract model instance
def extract_model(model):
    if isinstance(model, dict) and "model" in model:
        return model["model"]
    return model

# Function: prediction
def make_prediction(model, features, input_data):
    model_instance = extract_model(model)

    # Build dataframe
    df = pd.DataFrame([input_data], columns=features)

    # Run prediction
    pred = model_instance.predict(df)
    return pred[0]

# UI
st.title("Disease Case Classification Prediction App")

disease = st.selectbox("Select Disease", list(models.keys()))

selected_model = models[disease]

# Get features
top_features = get_top_features(selected_model)

# Warn user if no features found
if not top_features:
    st.error(f"❌ No feature information found inside the {disease} model.")
    st.write("This means the model was saved without a feature list.")
else:
    st.write(f"### Top 10 Features for {disease}")
    st.write(top_features)

# Input section
st.sidebar.header(f"Enter Input Values for {disease}")

input_data = {}
for f in top_features:
    input_data[f] = st.sidebar.number_input(f"Enter {f}", value=0.0)

# Predict button
if st.sidebar.button("Predict"):
    if not top_features:
        st.error("Cannot predict because no feature list exists for this model.")
    else:
        result = make_prediction(selected_model, top_features, input_data)
        st.success(f"Prediction for {disease}: **{result}**")
