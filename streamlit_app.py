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
# FEATURE SCHEMA
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
# CASE LABELS
# ============================================================

CASE_LABELS = {
    "Lassa Fever": {0:"Not a Case", 1:"Suspected Case", 2:"Probable Case", 3:"Confirmed Case"},
    "Measles":     {0:"Not a Case", 1:"Suspected Case", 2:"Probable Case", 3:"Confirmed Case"},
    "Cholera":     {0:"Not a Case", 1:"Suspected Case", 2:"Probable Case", 3:"Confirmed Case"},
    "Yellow Fever":{0:"Not a Case", 1:"Suspected Case", 2:"Probable Case", 3:"Confirmed Case"},
}

# ============================================================
# ENCODING FUNCTION
# ============================================================

def encode_input(input_dict, schema):
    encoded = {}
    for feature, value in input_dict.items():
        ftype = schema[feature]
        if ftype == "numeric":
            encoded[feature] = float(value)
        elif isinstance(ftype, list):
            mapping = {cat: i for i, cat in enumerate(ftype)}
            encoded[feature] = mapping[value]
        else:
            raise ValueError(f"Unknown feature type: {feature}")
    return encoded

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(model, features, encoded_data):
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
    df = pd.DataFrame([encoded_data], columns=features)
    raw = model.predict(df)[0]
    return raw

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

# -------------------
# SINGLE PREDICTION
# -------------------
st.sidebar.header("Enter Patient Data")
input_data = {}
for feature, ftype in schema.items():
    if ftype == "numeric":
        input_data[feature] = st.sidebar.number_input(feature, value=0.0)
    elif isinstance(ftype, list):
        input_data[feature] = st.sidebar.selectbox(feature, ftype)

if st.sidebar.button("Predict Case"):
    encoded_data = encode_input(input_data, schema)
    raw_pred = make_prediction(selected_model, features, encoded_data)
    label = CASE_LABELS[disease][raw_pred]

    color_class = {
        "Confirmed Case": "#27ae60",
        "Probable Case": "#f39c12",
        "Suspected Case": "#e67e22",
        "Not a Case": "#c0392b"
    }.get(label, "#7f8c8d")

    st.markdown(f'<div style="padding:20px; border-radius:12px; color:white; background-color:{color_class}; text-align:center; font-size:22px;">Prediction: <b>{label}</b></div>', unsafe_allow_html=True)
    st.caption(f"Raw Model Output: {raw_pred}")
