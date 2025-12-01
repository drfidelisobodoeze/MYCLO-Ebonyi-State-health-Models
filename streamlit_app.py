import streamlit as st
import joblib
import pandas as pd

# Load Models
lassa_model = joblib.load('xgboost_model.joblib')
yellow_fever_model = joblib.load('yellow_fever_lightgbm_model.joblib')
measles_model = joblib.load('measles_lightgbm_model.joblib')
cholera_model = joblib.load('cholera_catboost_model.joblib')

st.title("Disease Outcome Classification App")

# Disease selection
disease = st.selectbox(
    "Select Disease Model",
    ["Lassa Fever", "Yellow Fever", "Measles", "Cholera"]
)

# Collect inputs based on disease
if disease in ["Lassa Fever", "Yellow Fever"]:
    st.header(f"Input Data for {disease}")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    outcome_of_case = st.selectbox("Outcome of Case", ["Recovered", "Deceased"])

    input_df = pd.DataFrame([{
        "Sex": sex,
        "Age": age,
        "Outcome_of_case": outcome_of_case
    }])

    model = lassa_model if disease == "Lassa Fever" else yellow_fever_model

elif disease == "Measles":
    st.header("Input Data for Measles")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    temp_cat = st.selectbox("Temperature Category", ["Normal", "High"])

    input_df = pd.DataFrame([{
        "Sex": sex,
        "Age": age,
        "Temperature Category": temp_cat
    }])

    model = measles_model

else:  # Cholera
    st.header("Input Data for Cholera")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    temp_cat = st.selectbox("Temperature Category", ["Normal", "High"])

    input_df = pd.DataFrame([{
        "Sex": sex,
        "Age": age,
        "Temperature Category": temp_cat
    }])

    model = cholera_model

# Convert categories
for col in ["Sex", "Outcome_of_case", "Temperature Category"]:
    if col in input_df.columns:
        input_df[col] = input_df[col].astype("category")

# Prediction button
if st.button("Predict"):
    pred = model.predict(input_df)
    st.success(f"Predicted Outcome: **{pred[0]}**")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.write("This Streamlit app loads four disease models:")
st.sidebar.write("- Lassa Fever (XGBoost)")
st.sidebar.write("- Yellow Fever (LightGBM)")
st.sidebar.write("- Measles (LightGBM)")
st.sidebar.write("- Cholera (CatBoost)")
