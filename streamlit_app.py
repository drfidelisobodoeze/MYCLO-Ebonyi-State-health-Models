import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the models and metadata
lassa_model = joblib.load("lassa.joblib")
measles_model = joblib.load("measles.joblib")
cholera_model = joblib.load("cholera.joblib")
yellow_fever_model = joblib.load("yellow-fever.joblib")

# Mapping disease models to display
models = {
    "Lassa": lassa_model,
    "Measles": measles_model,
    "Cholera": cholera_model,
    "Yellow Fever": yellow_fever_model
}

# Helper function to get top 10 features
def get_top_features(model):
    return model['features']

# Helper function to make predictions
def make_prediction(model, features, input_data):
    model_instance = model['model']
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model_instance.predict(input_df)
    return prediction[0]

# Title and description of the app
st.title("Disease Classification Prediction")
st.write("Select the disease below, enter the data, and predict if the case is classified!")

# Sidebar for selecting disease
disease = st.sidebar.selectbox("Choose the Disease", ("Lassa", "Measles", "Cholera", "Yellow Fever"))

# Get the selected model
selected_model = models[disease]

# Display top 10 features
top_features = get_top_features(selected_model)
st.write(f"**Top 10 features for {disease} classification:**")
st.write(top_features)

# Form for entering patient data (only the top 10 features)
st.sidebar.header("Enter the Patient Data")
input_data = {}
for feature in top_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0)

# Predict button
if st.sidebar.button("Predict"):
    prediction = make_prediction(selected_model, top_features, input_data)
    st.write(f"The prediction result for the selected disease is: {prediction}")

# Displaying additional details
st.write("### Disease Data Information")
st.write(f"Model used: {disease}")
