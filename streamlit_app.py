import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD MODELS (.joblib can be sklearn, xgb, lgbm, or dict)
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
# MANUAL FEATURE DEFINITION (MIX OF CATEGORICAL & NUMERICAL)
# Update these to match your training dataset
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
# ENCODING FUNCTION (Handles Categorical → Numeric Mapping)
# ============================================================

def encode_input(input_dict, schema):
    encoded = {}

    for feature, value in input_dict.items():
        feature_type = schema[feature]

        # Numeric
        if feature_type == "numeric":
            encoded[feature] = float(value)

        # Categorical → Convert to integer encoding
        elif isinstance(feature_type, list):
            mapping = {cat: i for i, cat in enumerate(feature_type)}
            encoded[feature] = mapping[value]

        else:
            raise ValueError(f"Unknown feature type: {feature}")

    return encoded


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(model, features, encoded_data):

    # If model is dict style
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    df = pd.DataFrame([encoded_data], columns=features)
    prediction = model.predict(df)
    return prediction[0]


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🧠 Multi-Disease Case Classification System")
st.subheader("Supports Numeric & Categorical Inputs with Auto-Encoding")

disease = st.selectbox("Select Disease Model", list(models.keys()))
selected_model = models[disease]

schema = feature_schema[disease]
features = list(schema.keys())

st.write(f"### Features for {disease}")
st.write(schema)

# Collect input values
st.sidebar.header("Enter Patient Data")
input_data = {}

for feature, ftype in schema.items():

    if ftype == "numeric":
        input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    elif isinstance(ftype, list):
        input_data[feature] = st.sidebar.selectbox(f"{feature}", ftype)

# Predict Button
if st.sidebar.button("Predict Case"):

    # Encode categorical inputs
    encoded_data = encode_input(input_data, schema)

    # Predict
    prediction = make_prediction(selected_model, features, encoded_data)

    st.success(f"### Prediction for **{disease}**: {prediction}")
