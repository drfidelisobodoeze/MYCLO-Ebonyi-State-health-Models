import streamlit as st
import pandas as pd
import joblib
import numpy as np
from collections import OrderedDict

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

# Global variables for Cholera model metadata
CHOLERA_FEATURES = []
CHOLERA_TARGET_MAP = {}

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache(allow_output_mutation=True)
def load_generic_model_safe(path):
    """Load a joblib model and automatically extract from tuple if necessary."""
    loaded = joblib.load(path)
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded

@st.cache(allow_output_mutation=True)
def load_cholera_model(path):
    """Load the Cholera model dictionary from joblib."""
    return joblib.load(path)

try:
    # Load non-Cholera models safely
    lassa_model = load_generic_model_safe("lassa_xgb_9features.joblib")
    measles_model = load_generic_model_safe("measles.joblib")
    yellow_fever_model = load_generic_model_safe("yellow-fever.joblib")
    
    # Load Cholera model
    cholera_data = load_cholera_model("cholera.joblib")
    cholera_model = cholera_data['model']
    CHOLERA_FEATURES = cholera_data['features']
    CHOLERA_TARGET_MAP = cholera_data['target_map']
    
    models = {
        "Lassa Fever": lassa_model,
        "Measles": measles_model,
        "Cholera": cholera_model,
        "Yellow Fever": yellow_fever_model
    }
    
except FileNotFoundError as e:
    st.error(f"Model file not found. Please check your paths: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading models. Check your joblib files and paths.")
    st.stop()

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
        "vaccination_status": ["Vaccinated", "Unvaccinated"]  # <-- fixed
    },
    "Cholera": {
        "Age": "numeric",
        "Current_body_temperature_C": "numeric",
        "Diarrhea": ["No", "Yes", "Unknown"],
        "Vomiting": ["No", "Yes", "Unknown"],
        "Dehydration": ["No", "Yes", "Unknown"],
        "Fast heart rate (Tachycardia)": ["No", "Yes", "Unknown"],
        "Blood in urine (hematuria)": ["No", "Yes", "Unknown"],
        "Bloody or black stools (melena)": ["No", "Yes", "Unknown"],
        "Vaccination status": ["Unvaccinated", "Vaccinated", "Unknown"],
        "Outcome of case": ["Alive", "Dead", "Unknown"],
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
def encode_input_onehot(input_dict, schema, model, disease):
    if disease == "Cholera":
        model_features = CHOLERA_FEATURES
    else:
        model_obj = model
        if hasattr(model_obj, "get_booster"):
            model_features = model_obj.get_booster().feature_names
        elif hasattr(model_obj, "feature_name_"):
            model_features = model_obj.feature_name_
        else:
            model_features = list(input_dict.keys())
    encoded = dict.fromkeys(model_features, 0)
    for feature, value in input_dict.items():
        ftype = schema.get(feature)
        if ftype == "numeric":
            if feature in encoded:
                try:
                    encoded[feature] = float(value)
                except ValueError:
                    pass
        elif isinstance(ftype, list):
            col_name = f"{feature}_{value}"
            if col_name in encoded:
                encoded[col_name] = 1
    return pd.DataFrame([encoded]).reindex(columns=model_features, fill_value=0)

# ============================================================
# PREDICTION
# ============================================================
def make_prediction(model, input_df):
    return model.predict(input_df)[0]

# ============================================================
# CLINICAL RULES
# ============================================================
def lassa_clinical_rules(input_data):
    temp = input_data.get("Current_body_temperature_in___C", 37)
    lab_result = input_data.get("Latest_sample_final_laboratory_result", "Negative").upper()
    categorical_features = ["Fever","Abdominal_pain","Bleeding_or_bruising","Vomiting",
                            "Sore_throat","Diarrhea","General_weakness","Chest_pain"]
    all_no = all(input_data.get(f, "No") == "No" for f in categorical_features)
    if lab_result == "POSITIVE":
        return "Confirmed Case"
    elif lab_result == "NEGATIVE" and all_no and temp <= 38.0:
        return "Not a Case"
    elif temp > 38:
        return "Suspected Case"
    else:
        return None

def measles_clinical_rules(input_data):
    koplik = input_data.get("koplik_spots", "No")
    conjunctivitis = input_data.get("conjunctivitis", "No")
    vaccination = input_data.get("vaccination_status", "Unvaccinated")
    categorical_features = ["fever","rash","cough","runny_nose","conjunctivitis",
                            "koplik_spots","travel_history","exposure"]
    all_negative = all(input_data.get(f) in ["No","Absent","None"] for f in categorical_features)
    if vaccination == "Vaccinated" or all_negative:
        return "Not a Case"
    elif koplik == "Yes" and conjunctivitis == "Yes":
        return "Confirmed Case"
    else:
        return None

def cholera_clinical_rules(input_data):
    diarrhea = input_data.get("Diarrhea", "No")
    vomiting = input_data.get("Vomiting", "No")
    dehydration = input_data.get("Dehydration", "No") 
    tachycardia = input_data.get("Fast heart rate (Tachycardia)", "No")
    vaccination = input_data.get("Vaccination status", "Unvaccinated")
    all_positive_symptoms = (diarrhea=="Yes" and vomiting=="Yes" and dehydration=="Yes" and tachycardia=="Yes")
    is_unvaccinated = (vaccination=="Unvaccinated")
    if all_positive_symptoms and is_unvaccinated:
        return "Confirmed Case"
    elif (diarrhea=="No" and vomiting=="No" and dehydration=="No") and vaccination=="Vaccinated":
        return "Not a Case"
    elif diarrhea=="Yes" and vomiting=="Yes":
        return "Suspected Case"
    else:
        return None

# ============================================================
# UI
# ============================================================
st.title("🩺 MYCLO - EBONYI STATE Multiple-Disease Classification/Prediction System")
st.subheader("PhD Research Work by Calister Nnenna Ogbonna-Mbah")
st.subheader("Enter Patient Data Below")

if models:
    disease = st.selectbox("Select Disease Model", list(models.keys()))
    model = models[disease]
    schema = feature_schema[disease]

    with st.form("input_form"):
        st.header(f"Input for {disease}")
        input_data = {}
        cols = st.columns(2)
        feature_list = list(schema.items())
        for i, (feature, ftype) in enumerate(feature_list):
            col = cols[i % 2]
            with col:
                ui_feature_name = feature.replace('___C','°C').replace('_',' ')
                if ftype == "numeric":
                    input_data[feature] = st.number_input(ui_feature_name, value=0.0, format="%.2f", key=f"{disease}_{feature}")
                else:
                    input_data[feature] = st.selectbox(ui_feature_name, ftype, key=f"{disease}_{feature}")
        st.markdown("---")
        submit = st.form_submit_button("Predict Case Classification")

# ============================================================
# DISPLAY RESULTS
# ============================================================
if submit:
    try:
        input_df = encode_input_onehot(input_data, schema, model, disease)
        if disease == "Cholera":
            raw_pred = make_prediction(model, input_df)
            label = CHOLERA_TARGET_MAP[raw_pred]
        else:
            raw_pred = make_prediction(model, input_df)
            label = CASE_LABELS[disease][raw_pred]

        if disease == "Lassa Fever":
            rule_override = lassa_clinical_rules(input_data)
        elif disease == "Measles":
            rule_override = measles_clinical_rules(input_data)
        elif disease == "Cholera":
            rule_override = cholera_clinical_rules(input_data)
        else:
            rule_override = None

        final_label = rule_override if rule_override else label

        bg_color = {
            "Confirmed Case": "#27ae60",
            "Probable Case": "#f39c12",
            "Suspected Case": "#e67e22",
            "Not a Case": "#95a5a6",
            "Confirmed case": "#27ae60",
            "Probable case": "#f39c12",
            "Suspect case": "#e67e22",
            "Not a case": "#95a5a6"
        }.get(final_label, "#7f8c8d")

        border_color = "#ff69b4"
        shadow_style = "box-shadow: 3px 3px 12px rgba(0,0,0,0.2);"

        st.markdown("## 📊 Classification Result")
        st.markdown(f"""
        <div style='
            background-color: {bg_color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
            border: 3px solid {border_color};
            {shadow_style}
        '>
            CLASSIFICATION: {final_label}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Input Data Summary")
        st.json(input_data)

    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your model files and input values.")
