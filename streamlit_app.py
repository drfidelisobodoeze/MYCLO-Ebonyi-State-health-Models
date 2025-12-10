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
        "vaccination_status": ["Vaccinated", "Unvacc]()_
