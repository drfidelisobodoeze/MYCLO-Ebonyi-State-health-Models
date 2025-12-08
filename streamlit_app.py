import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD MODELS
# ============================================================

lassa_xgb = joblib.load("lassa_xgb.joblib")
measles_model = joblib.load("measles.joblib")
cholera_model = joblib.load("cholera_lgb.joblib")
yellow_fever_model = joblib.load("yellow-fever.joblib")

models = {
    "Lassa Fever": lassa_xgb,
    "Measles": measles_model,
    "Cholera": cholera_model,
    "Yellow Fever": yellow_fever_model
}

# ============================================================
# FEATURE SCHEMA DEFINITIONS
# (NUMERIC OR CATEGORICAL → LIST)
# ============================================================

feature_schema = {
    "Lassa Fever": {
        "age": "numeric",
        "temperature": "numeric",
        "headache": ["Yes", "No"],
        "bleeding": ["Yes", "No"],
        "vomiting": ["Yes", "No"],
        "abdominal_pain": ["Yes", "No"],
        "diarrhea": ["Yes", "No"],
        "weakness": ["Yes", "No"],
        "protein_level": "numeric",
        "platelet_count": "numeric"
    },

    "Measles": {
        "age": "numeric",
        "fever": ["None", "Mild", "High"],
        "rash": ["Present", "Absent"],
        "cough": ["Yes", "No"],
        "runny_nose": ["Yes", "No"],
        "conjunctivitis": ["Yes", "No"],
        "koplik_spots": ["Yes", "No"],
        "travel_history": ["Yes", "No"],
        "exposure": ["Yes", "No"],
        "vaccination_status": ["Vaccinated", "Unvaccinated"]
    },

    "Cholera": {
        "age": "numeric",
        "watery_diarrhea": ["Yes", "No"],
        "vomiting": ["Yes", "No"],
        "dehydration": ["None", "Mild", "Severe"],
        "heart_rate": "numeric",
        "temperature": "numeric",
        "bp_systolic": "numeric",
        "bp_diastolic": "numeric",
        "sodium": "numeric",
        "chloride": "numeric"
    },

    "Yellow Fever": {
        "age": "numeric",
        "fever": ["None", "Mild", "High"],
        "headache": ["Yes", "No"],
        "jaundice": ["Yes", "No"],
        "muscle_pain": ["Yes", "No"],
        "vomiting": ["Yes", "No"],
        "bleeding": ["Yes", "No"],
        "liver_function": ["Normal", "Elevated", "Critical"],
        "platelet_count": "numeric",
        "exposure": ["Yes", "No"]
    }
}

# ============================================================
# ENCODER — Converts categorical → integer encoding
# ============================================================

def encode_features(input_dict, schema):
    encoded = {}

    for feature, value in input_dict.items():
        ftype = schema[feature]

        if ftype == "numeric":
            encoded[feature] = float(value)

        elif isinstance(ftype, list):  # categorical
            mapping = {cat: i for i, cat in enumerate(ftype)}
            encoded[feature] = mapping[value]

        else:
            raise ValueError(f"Unknown feature type: {ftype}")

    return encoded


# ============================================================
# UNIVERSAL PREDICTOR — WORKS WITH XGB/LGBM/PURE SKLEARN
# ============================================================

def make_prediction(model, schema, encoded_data):
    
    # Unwrap dictionary model
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    # ===============================
    # GET MODEL EXPECTED FEATURE NAMES
    # ===============================

    expected_cols = None

    # XGBoost
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        expected_cols = booster.feature_names

    # LightGBM
    elif hasattr(model, "feature_name_"):
        expected_cols = model.feature_name_

    # Sklearn fallback (not guaranteed)
    elif hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)

    # If still none → use input keys (dangerous but ok for raw models)
    if expected_cols is None:
        expected_cols = list(encoded_data.keys())

    # ===============================
    # BUILD INPUT DATAFRAME
    # ===============================

    df = pd.DataFrame([encoded_data])

    # Add missing columns → Fill with 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Drop extra unexpected columns
    df = df[expected_cols]

    # ===============================
    # MAKE PREDICTION
    # ===============================
    pred = model.predict(df)

    # Some XGBoost/LGBM return array in array
    if hasattr(pred, "__len__"):
        return pred[0]

    return pred


# ============================================================
# STREAMLIT APP UI
# ============================================================

st.title("🧠 Multi-Disease Case Classification System")
st.write("Enter patient features below for automated disease classification.")

# Select model
disease = st.selectbox("Select Disease", list(models.keys()))
selected_model = models[disease]
schema = feature_schema[disease]

st.sidebar.header(f"Enter patient data for {disease}")

# Input fields
input_data = {}

for feature, ftype in schema.items():

    if ftype == "numeric":
        input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    elif isinstance(ftype, list):
        input_data[feature] = st.sidebar.selectbox(f"{feature}", ftype)

# Predict button
if st.sidebar.button("Predict"):

    encoded_data = encode_features(input_data, schema)

    prediction = make_prediction(selected_model, schema, encoded_data)

    st.success(f"### Prediction for **{disease}**: {prediction}")
