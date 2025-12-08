import streamlit as st
import pandas as pd
import joblib

# =============================
# Load ML Models
# =============================

# IMPORTANT:
# If each .joblib contains ONLY the model, then add feature lists manually below.
# If they contain {"model": model, "features": [...]}, code will detect automatically.

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

# =============================
# → MANUAL FEATURE LIST (Edit These)
# If your .joblib files DO NOT contain metadata
# Update these lists to match your training features
# =============================

manual_features = {
    "Lassa Fever": ["age", "temperature", "headache", "bleeding", "vomiting", "abdominal_pain",
                    "diarrhea", "weakness", "protein_level", "platelet_count"],
    
    "Measles": ["age", "fever", "rash", "cough", "runny_nose", "conjunctivitis",
                "koplik_spots", "travel_history", "exposure", "vaccination_status"],
    
    "Cholera": ["age", "watery_diarrhea", "vomiting", "dehydration", "heart_rate",
                "temperature", "bp_systolic", "bp_diastolic", "sodium", "chloride"],
    
    "Yellow Fever": ["age", "fever", "headache", "jaundice", "muscle_pain",
                     "vomiting", "bleeding", "liver_function", "platelet_count", "exposure"]
}

# =============================
#AUTO-DETECT FEATURES FROM MODEL
# =============================
def get_top_features(model, disease):
    # If model is dict with features
    if isinstance(model, dict) and "features" in model:
        return model["features"]

    # Fallback to manual list
    return manual_features[disease]


# =============================
# Prediction Function
# =============================
def make_prediction(model, features, input_data):

    # If joblib contains {"model": model, "features": [...]}
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    # Convert input to DataFrame
    df = pd.DataFrame([input_data], columns=features)

    # Predict
    prediction = model.predict(df)
    return prediction[0]

# =============================
# UI STYLING
# =============================
st.markdown("""
<style>
body {
    background-color: #f4f4f9;
    font-family: 'Roboto', sans-serif;
}
@media screen and (max-width: 600px) {
    .main {
        max-width: 100%;
        padding: 10px;
    }
    h1 { font-size: 1.6em; text-align: center; }
    .stNumberInput input, .stSelectbox select, .stButton button {
        width: 100%;
        padding: 10px;
    }
}
</style>
""", unsafe_allow_html=True)

# =============================
# Streamlit APP
# =============================

st.title("🧪 MYCLO Ebonyi State Health ML Models")
st.subheader("Multiple Infectious Diseases Case Classification Prediction App")

# Select disease
disease = st.selectbox("Select Disease Model", list(models.keys()))

selected_model = models[disease]

# Get features
top_features = get_top_features(selected_model, disease)

st.write(f"### 🔍 Required Features for {disease}")
st.write(top_features)

# Sidebar inputs
input_data = {}
with st.sidebar:
    st.header("Enter Input Values")
    for feature in top_features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    predict_btn = st.button("Predict Case Classification")

# Perform prediction
if predict_btn:
    prediction = make_prediction(selected_model, top_features, input_data)
    st.success(f"### ✅ Prediction for **{disease}**: {prediction}")
