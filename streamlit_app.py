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
# ENCODI
