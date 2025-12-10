import streamlit as st
import pandas as pd
import joblib
import numpy as np
from collections import OrderedDict

# Global variables for model metadata (will be populated on load)
CHOLERA_FEATURES = []
CHOLERA_TARGET_MAP = {}
YF_FEATURES = []
YF_TARGET_MAP = {}

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
# LOAD MODELS (FIXED FOR TUPLE ISSUE & METADATA)
# ============================================================
@st.cache_resource
def load_generic_model(path):
    # Loads Lassa/Measles model, handling case where it's saved as a tuple or list.
    loaded_data = joblib.load(path)
    
    # FIX: Check if the loaded data is a tuple/list and extract the model (element 0)
    if isinstance(loaded_data, (tuple, list)):
        return loaded_data[0]
    
    # Check if the loaded data is a dictionary (in case it was saved like Cholera/YF)
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        return loaded_data['model']

    # Otherwise, assume the loaded object is the model itself
    return loaded_data

@st.cache_resource
def load_metadata_model(path, feature_list_global, target_map_global):
    # Loads dictionary containing model and metadata (for Cholera, YF)
    data = joblib.load(path)
    # Populate global metadata variables
    globals()[feature_list_global] = data['features']
    globals()[target_map_global] = data['target_map']
    return data['model']

try:
    # Load non-metadata models (Lassa, Measles) - uses the FIXED function
    lassa_model = load_generic_model("lassa_xgb_9features.joblib")
    measles_model = load_generic_model("measles.joblib")
    
    # Load Cholera and Yellow Fever models (which contain metadata dictionaries)
    cholera_model = load_metadata_model("cholera.joblib", 'CHOLERA_FEATURES', 'CHOLERA_TARGET_MAP')
    yellow_fever_model = load_metadata_model("yellow-fever.joblib", 'YF_FEATURES', 'YF_TARGET_MAP')
    
    models = {
        "Lassa Fever": lassa_model,
        "Measles": measles_model,
        "Cholera": cholera_model,
        "Yellow Fever": yellow_fever_model
    }
    
except FileNotFoundError as e:
    st.error(f"Model file not found. Please ensure all four model files are in the correct path: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")
    st.stop()


# ============================================================
# FEATURE SCHEMA (UPDATED FOR CHOLERA & YELLOW FEVER)
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
    # CHOLERA SCHEMA
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
    # YELLOW FEVER SCHEMA
    "Yellow Fever": {
        "age": "numeric",
        "fever": ["None", "Mild", "High", "Unknown"],
        "headache": ["No", "Yes", "Unknown"],
        "jaundice": ["No", "Yes", "Unknown"],
        "muscle_pain": ["No", "Yes", "Unknown"],
        "vomiting": ["No", "Yes", "Unknown"],
        "bleeding_or_bruising": ["No", "Yes", "Unknown"],
        "hemorrhagic_syndrome": ["No", "Yes", "Unknown"],
        "dark_urine": ["No", "Yes", "Unknown"],
        "fatigue_general_weakness": ["No", "Yes", "Unknown"],
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
# ENCODER (MODIFIED for Cholera & Yellow Fever)
# ============================================================
def encode_input_onehot(input_dict, schema, model, disease):
    
    # --- MODEL-SPECIFIC FEATURE LIST ---
    if disease == "Cholera":
        model_features = CHOLERA_FEATURES
    elif disease == "Yellow Fever":
        model_features = YF_FEATURES
    else:
        # --- GENERIC ENCODER LOGIC (for Lassa, Measles) ---
        model_obj = model
        if hasattr(model_obj, "get_booster"):
            model_features = model_obj.get_booster().feature_names
        elif hasattr(model_obj, "feature_name_"):
            model_features = model_obj.feature_name_
        else:
            model_features = list(input_dict.keys())

    # Create a template DataFrame initialized with zeros, using the correct feature list
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
            # One-hot encoding construction: 'FeatureName_Value'
            col_name = f"{feature}_{value}"
            if col_name in encoded:
                encoded[col_name] = 1
                
    # Create DataFrame and ensure the feature order matches the model's expectations
    return pd.DataFrame([encoded]).reindex(columns=model_features, fill_value=0)

# ============================================================
# PREDICTION
# ============================================================
def make_prediction(model, input_df):
    return model.predict(input_df)[0]

# ============================================================
# CLINICAL RULES (Unchanged)
# ============================================================

def lassa_clinical_rules(input_data):
    temp = input_data.get("Current_body_temperature_in___C", 37)
    lab_result = input_data.get("Latest_sample_final_laboratory_result", "Negative").upper()
    categorical_features = [
        "Fever","Abdominal_pain","Bleeding_or_bruising","Vomiting",
        "Sore_throat","Diarrhea","General_weakness","Chest_pain"
    ]
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

    categorical_features = [
        "fever","rash","cough","runny_nose","conjunctivitis",
        "koplik_spots","travel_history","exposure"
    ]
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
    
    all_positive_symptoms = (
        diarrhea == "Yes" and
        vomiting == "Yes" and
        dehydration == "Yes" and
        tachycardia == "Yes"
    )
    is_unvaccinated = (vaccination == "Unvaccinated")

    if all_positive_symptoms and is_unvaccinated:
        return "Confirmed Case"
    elif (diarrhea == "No" and vomiting == "No" and dehydration == "No") and (vaccination == "Vaccinated"):
        return "Not a Case"
    elif diarrhea == "Yes" and vomiting == "Yes":
        return "Suspected Case"
    else:
        return None 
        
def yellow_fever_clinical_rules(input_data):
    # No custom clinical rules defined for Yellow Fever in this system.
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
                # Clean up feature name for UI display
                ui_feature_name = feature.replace('___C', '°C').replace('_', ' ')
                
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
        raw_pred = make_prediction(model, input_df)

        # Determine the label using the correct target map
        if disease == "Cholera":
            label = CHOLERA_TARGET_MAP[raw_pred]
        elif disease == "Yellow Fever":
            label = YF_TARGET_MAP[raw_pred] # Use the custom YF map
        else:
            label = CASE_LABELS[disease][raw_pred]

        # Apply clinical rules
        rule_override = None
        if disease == "Lassa Fever":
            rule_override = lassa_clinical_rules(input_data)
        elif disease == "Measles":
            rule_override = measles_clinical_rules(input_data)
        elif disease == "Cholera":
            rule_override = cholera_clinical_rules(input_data)
        elif disease == "Yellow Fever":
            rule_override = yellow_fever_clinical_rules(input_data) 

        final_label = rule_override if rule_override else label

        # Color mapping (Includes both Title Case and lower case keys for consistency)
        bg_color = {
            "Confirmed Case": "#27ae60", "Confirmed case": "#27ae60",
            "Probable Case": "#f39c12", "Probable case": "#f39c12",
            "Suspected Case": "#e67e22", "Suspect case": "#e67e22",
            "Not a Case": "#95a5a6", "Not a case": "#95a5a6"
        }.get(final_label, "#7f8c8d")

        border_color = "#ff69b4"
        shadow_style = "box-shadow: 3px 3px 12px rgba(0,0,0,0.2);"

        # Display the result 
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
        <div style='margin-top: 15px; font-size: 14px; color: #555; text-align: center;'>
            ML Model Prediction: **{label}** | Overridden by Clinical Rule: **{'Yes' if rule_override else 'No'}**
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Input Data Summary")
        st.json(input_data)

    except Exception as e:
        st.error(f"An error occurred during prediction for {disease}. Please check your model files and input values. Error: {e}")
