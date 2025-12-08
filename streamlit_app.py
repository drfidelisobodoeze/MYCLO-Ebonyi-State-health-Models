import streamlit as st
import pandas as pd
import joblib

# ============================================================
# MOBILE-RESPONSIVE STYLING
# ============================================================
st.markdown("""
<style>
/* Make main container wider on mobile */
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

/* Full-width input widgets */
.stSelectbox, .stNumberInput, .stTextInput {
    width: 100% !important;
}

/* Remove side padding */
[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
lassa_model = joblib.load("lassa_xgb.joblib")
measles_model = joblib.load("measles.joblib")
cholera_model = joblib.load("cholera_lgb.joblib")
yellow_fever_model = joblib.load("yellow-fever.joblib")

models = {
    "Lassa Fever": lassa_model,
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
# ENCODER
# ============================================================
def encode_input_onehot(input_dict, schema, model):
    model_obj = model["model"] if isinstance(model, dict) and "model" in model else model

    if hasattr(model_obj, "get_booster"):
        model_features = model_obj.get_booster().feature_names
    elif hasattr(model_obj, "feature_name_"):
        model_features = model_obj.feature_name_
    else:
        model_features = list(input_dict.keys())

    encoded = dict.fromkeys(model_features, 0)

    for feature, value in input_dict.items():
        ftype = schema[feature]

        if ftype == "numeric":
            if feature in encoded:
                encoded[feature] = float(value)

        elif isinstance(ftype, list):
            col_name = f"{feature}_{value}"
            if col_name in encoded:
                encoded[col_name] = 1

    return pd.DataFrame([encoded])

# ============================================================
# PREDICT
# ============================================================
def make_prediction(model, input_df):
    model_obj = model["model"] if isinstance(model, dict) and "model" in model else model
    return model_obj.predict(input_df)[0]

# ============================================================
# UI TITLE
# ============================================================
st.title("🧠 MYCLO - EBONYI STATE Multi-Disease Case Classification System")
st.subheader("Enter Patient Data Below")

# ============================================================
# MAIN INPUT FORM (MOBILE FRIENDLY)
# ============================================================
disease = st.selectbox("Select Disease Model", list(models.keys()))
model = models[disease]
schema = feature_schema[disease]

with st.form("input_form"):
    st.header("Patient Data Input")

    input_data = {}

    for feature, ftype in schema.items():
        if ftype == "numeric":
            input_data[feature] = st.number_input(feature, value=0.0)
        else:
            input_data[feature] = st.selectbox(feature, ftype)

    submit = st.form_submit_button("Predict Case")

# ============================================================
# RESULTS
# ============================================================
if submit:
    try:
        input_df = encode_input_onehot(input_data, schema, model)
        raw_pred = make_prediction(model, input_df)
        label = CASE_LABELS[disease][raw_pred]

        color_class = {
            "Confirmed Case": "#27ae60",
            "Probable Case": "#f39c12",
            "Suspected Case": "#e67e22",
            "Not a Case": "#c0392b"
        }.get(label, "#7f8c8d")

        st.markdown(
            f'<div style="padding:20px; border-radius:12px; color:white; '
            f'background-color:{color_class}; text-align:center; font-size:22px;">'
            f'Prediction: <b>{label}</b></div>',
            unsafe_allow_html=True
        )

        st.caption(f"Raw Model Output: {raw_pred}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
