import streamlit as st
import pandas as pd
import joblib

# ============================================================
# RESPONSIVE LAYOUT
# ============================================================
st.markdown("""
<style>
.block-container {
    max-width: 60%;
    margin-left: auto;
    margin-right: auto;
}
@media only screen and (max-width: 768px) {
    .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
.stSelectbox, .stNumberInput, .stTextInput {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

lassa_model = load_model("lassa_xgb_9features.joblib")
measles_model = load_model("measles.joblib")
cholera_model = load_model("cholera_lgb.joblib")
yellow_fever_model = load_model("yellow-fever.joblib")

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
        "Fever": ["Yes", "No"],
        "Current_body_temperature_in___C": "numeric",
        "Abdominal_pain": ["Yes", "No"],
        "Bleeding_or_bruising": ["Yes", "No"],
        "Vomiting": ["Yes", "No"],
        "Sore_throat": ["Yes", "No"],
        "Diarrhea": ["Yes", "No"],
        "General_weakness": ["Yes", "No"],
        "Chest_pain": ["Yes", "No"],
        "Latest_sample_final_laboratory_result": ["Positive", "Negative","Indeterminate"]
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
    "Lassa Fever": {0:"Not a Case", 1:"Suspected Case", 2:"Confirmed Case"},
    "Measles":     {0:"Not a Case", 1:"Suspected Case", 2:"Probable Case", 3:"Confirmed Case"},
    "Cholera":     {0:"Not a Case", 1:"Suspected Case", 2:"Probable Case", 3:"Confirmed Case"},
    "Yellow Fever":{0:"Not a Case", 1:"Suspected Case", 2:"Probable Case", 3:"Confirmed Case"},
}

# ============================================================
# ENCODER
# ============================================================
def encode_input_onehot(input_dict, schema, model):
    model_obj = model
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
# PREDICTION
# ============================================================
def make_prediction(model, input_df):
    return model.predict(input_df)[0]

# ============================================================
# LASSA FEVER CLINICAL RULES
# ============================================================
def lassa_clinical_rules(input_data):
    """
    Overrides ML prediction:
    - Lab Positive -> Confirmed Case
    - Fever > 38C -> Suspected Case
    - Lab Negative + all other symptoms 'No' -> Not a Case
    """
    temp = input_data.get("Current_body_temperature_in___C", 37)
    lab_result = input_data.get("Latest_sample_final_laboratory_result", "Negative").upper()

    categorical_features = [
        "Fever",
        "Abdominal_pain",
        "Bleeding_or_bruising",
        "Vomiting",
        "Sore_throat",
        "Diarrhea",
        "General_weakness",
        "Chest_pain"
    ]
    all_no = all(input_data.get(f, "No") == "No" for f in categorical_features)

    if lab_result == "POSITIVE":
        return "Confirmed Case"
    elif lab_result == "NEGATIVE" and all_no:
        return "Not a Case"
    elif temp > 38:
        return "Suspected Case"
    else:
        return None

# ============================================================
# UI
# ============================================================
st.title("🧠 MYCLO - EBONYI STATE Multi-Disease Case Classification System")
st.subheader("Enter Patient Data Below")

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
# DISPLAY RESULTS
# ============================================================
if submit:
    try:
        input_df = encode_input_onehot(input_data, schema, model)
        raw_pred = make_prediction(model, input_df)
        label = CASE_LABELS[disease][raw_pred]

        # Apply clinical rules for Lassa Fever
        if disease == "Lassa Fever":
            rule_override = lassa_clinical_rules(input_data)
            final_label = rule_override if rule_override else label
        else:
            final_label = label
            rule_override = None

        # Color mapping
        bg_color = {
            "Confirmed Case": "#27ae60",
            "Probable Case": "#f39c12",
            "Suspected Case": "#e67e22",
            "Not a Case": "#95a5a6"
        }.get(final_label, "#7f8c8d")

        border_color = "#ff69b4"
        shadow_style = "box-shadow: 3px 3px 12px rgba(0,0,0,0.2);"

        st.markdown(
            f'''
            <div style="
                padding:20px;
                border-radius:12px;
                color:white;
                background-color:{bg_color};
                border: 3px solid {border_color};
                {shadow_style}
                text-align:center;
                font-size:22px;">
                Final Prediction: <b>{final_label}</b>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.caption(f"ML Model Prediction: {label}")
        if rule_override:
            st.caption(f"Clinical Rule Override: {rule_override}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
