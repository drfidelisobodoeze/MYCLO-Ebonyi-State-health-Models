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
# Note: Ensure these files exist in the same directory or use the correct path.
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
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
except FileNotFoundError as e:
    st.error(f"Model file not found. Please ensure all model files are in the correct path: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")
    st.stop()


# ============================================================
# FEATURE SCHEMA
# ============================================================
feature_schema = {
    "Lassa Fever": {
        "Fever": ["Yes", "No"],
        "Current_body_temperature_in___C": "numeric", # Check this feature name against your model (See note in previous response)
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
    # Get feature names from the trained model (for XGBoost and LightGBM)
    if hasattr(model_obj, "get_booster"):
        model_features = model_obj.get_booster().feature_names
    elif hasattr(model_obj, "feature_name_"):
        model_features = model_obj.feature_name_
    else:
        # Fallback: This is UNSAFE for one-hot encoded models (see previous note)
        model_features = list(input_dict.keys())

    encoded = dict.fromkeys(model_features, 0)
    for feature, value in input_dict.items():
        ftype = schema[feature]
        if ftype == "numeric":
            if feature in encoded:
                encoded[feature] = float(value)
        elif isinstance(ftype, list):
            # One-hot encoding logic: feature_value
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
    # NOTE: Using 'Current_body_temperature_in___C' as per your schema
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

# ============================================================
# MEASLES CLINICAL RULES
# ============================================================
def measles_clinical_rules(input_data):
    koplik = input_data.get("koplik_spots", "No")
    conjunctivitis = input_data.get("conjunctivitis", "No")
    vaccination = input_data.get("vaccination_status", "Unvaccinated")

    # Check if all categorical features are "No"/"Absent"/"None"
    categorical_features = [
        "fever","rash","cough","runny_nose","conjunctivitis",
        "koplik_spots","travel_history","exposure"
    ]
    # Check if all features are in a negative state (excluding age)
    all_negative = all(input_data.get(f) in ["No","Absent","None"] for f in categorical_features)

    if vaccination == "Vaccinated" or all_negative:
        return "Not a Case"
    elif koplik == "Yes" and conjunctivitis == "Yes":
        return "Confirmed Case"
    else:
        return None

# ============================================================
# UI
# ============================================================
st.title("🩺 MYCLO - EBONYI STATE Multi-Disease Case Classification System")
st.subheader("Enter Patient Data Below")

# Ensure models list is not empty before proceeding
if models:
    disease = st.selectbox("Select Disease Model", list(models.keys()))
    model = models[disease]
    schema = feature_schema[disease]

    with st.form("input_form"):
        st.header(f"Input for {disease}")
        input_data = {}
        # Organize inputs into two columns for better layout
        cols = st.columns(2)
        
        feature_list = list(schema.items())
        # Split features roughly in half for the two columns
        half_point = len(feature_list) // 2

        for i, (feature, ftype) in enumerate(feature_list):
            col = cols[i % 2] # Cycle between the two columns
            with col:
                if ftype == "numeric":
                    # Use float or int depending on expected input
                    input_data[feature] = st.number_input(feature, value=0.0, format="%.2f", key=f"{disease}_{feature}")
                else:
                    input_data[feature] = st.selectbox(feature, ftype, key=f"{disease}_{feature}")
        
        # Ensure the submit button is outside the columns for full width
        st.markdown("---")
        submit = st.form_submit_button("Predict Case Classification")

# ============================================================
# DISPLAY RESULTS
# ============================================================
if submit:
    try:
        input_df = encode_input_onehot(input_data, schema, model)
        raw_pred = make_prediction(model, input_df)
        label = CASE_LABELS[disease][raw_pred]

        # Apply clinical rules
        if disease == "Lassa Fever":
            rule_override = lassa_clinical_rules(input_data)
        elif disease == "Measles":
            rule_override = measles_clinical_rules(input_data)
        else:
            # Cholera and Yellow Fever prediction relies solely on ML model
            rule_override = None

        final_label = rule_override if rule_override else label

        # Color mapping
        bg_color = {
            "Confirmed Case": "#27ae60",
            "Probable Case": "#f39c12",
            "Suspected Case": "#e67e22",
            "Not a Case": "#95a5a6"
        }.get(final_label, "#7f8c8d")

        border_color = "#ff69b4"
        shadow_style = "box-shadow: 3px 3px 12px rgba(0,0,0,0.2);"

        # Display the result (The completed block)
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
         #<div style='margin-top: 15px; font-size: 14px; color: #555; text-align: center;'>
          #   ML Model Prediction: **{label}** | Overridden by Clinical Rule: **{'Yes' if rule_override else 'No'}**
        # </div>
        """, unsafe_allow_html=True)
        
        # Optional: Show input data for verification
        st.markdown("---")
        st.subheader("Input Data Summary")
        st.json(input_data)

    except Exception as e:
        st.error(f"An error occurred during prediction for {disease}. Please check your model files and input values. Error: {e}")
