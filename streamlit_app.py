import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# LOAD MODELS
# ==========================================================
@st.cache_resource
def load_models():
    return {
        "Lassa Fever": joblib.load("lassa_xgb.joblib"),
        "Measles": joblib.load("measles.joblib"),
        "Cholera": joblib.load("cholera_lgb.joblib"),
        "Typhoid Fever": joblib.load("yellow-fever.joblib"),
    }

models = load_models()

# ==========================================================
# CASE CLASSIFICATION LABELS (DISEASE-SPECIFIC)
# ==========================================================
CASE_LABELS = {
    "Lassa Fever": {
        0: "Not a Case",
        1: "Suspected Case",
        2: "Probable Case",
        3: "Confirmed Case"
    },
    "Measles": {
        0: "Not a Case",
        1: "Suspected Case",
        2: "Probable Case",
        3: "Confirmed Case"
    },
    "Cholera": {
        0: "Not a Case",
        1: "Suspected Case",
        2: "Probable Case",
        3: "Confirmed Case"
    },
    "Yello fever": {
        0: "Not a Case",
        1: "Suspected Case",
        2: "Probable Case",
        3: "Confirmed Case"
    },
}

# ==========================================================
# CATEGORY ENCODERS (Modify with your real encodings)
# ==========================================================
categorical_encoders = {
    "sex": {"Male": 0, "Female": 1},
    "location": {"Urban": 0, "Rural": 1},
    "contact_case": {"No": 0, "Yes": 1},
}

# ==========================================================
# INPUT FORM BUILDER
# ==========================================================
def build_input_form(feature_names):
    st.subheader("Enter Patient Data")

    data = {}

    for feature in feature_names:
        if feature in categorical_encoders:
            data[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                list(categorical_encoders[feature].keys())
            )
        else:
            data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=0.0
            )
    return data

# ==========================================================
# ENCODE INPUTS
# ==========================================================
def encode_inputs(input_dict):
    encoded = {}

    for key, value in input_dict.items():
        if key in categorical_encoders:
            encoded[key] = categorical_encoders[key][value]
        else:
            encoded[key] = value

    return encoded

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def make_prediction(model, df, disease):
    raw_pred = model.predict(df)[0]
    label = CASE_LABELS[disease].get(raw_pred, f"Unknown Class ({raw_pred})")
    return raw_pred, label

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("🩺 Disease Case Classification Prediction System")
st.write("Predict classification for **Lassa Fever, Measles, Cholera & Yellow Fever** using ML models.")

# Model selection
selected_disease = st.selectbox("Select Disease Model:", list(models.keys()))
selected_model = models[selected_disease]

# Show required features
st.info(f"Model expects the following features: {list(selected_model.feature_names_in_)}")

# Build input UI
user_inputs = build_input_form(selected_model.feature_names_in_)
encoded_data = encode_inputs(user_inputs)

# Convert to DataFrame
df = pd.DataFrame([encoded_data], columns=selected_model.feature_names_in_)

# Predict button
if st.button("🔍 Predict Case Classification"):
    try:
        numeric_pred, label_pred = make_prediction(selected_model, df, selected_disease)

        st.success(f"🎯 **Predicted Case Classification:** {label_pred}")
        st.caption(f"Model raw output: {numeric_pred}")

    except Exception as e:
        st.error("❌ Error occurred during prediction")
        st.code(str(e))
