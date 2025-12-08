import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Disease Case Classification App",
    layout="wide",
    page_icon="🩺"
)

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
# CASE CLASS LABELS
# ==========================================================
CASE_LABELS = {
    "Lassa Fever":      {0:"Not a Case",1:"Suspected Case",2:"Probable Case",3:"Confirmed Case"},
    "Measles":          {0:"Not a Case",1:"Suspected Case",2:"Probable Case",3:"Confirmed Case"},
    "Cholera":          {0:"Not a Case",1:"Suspected Case",2:"Probable Case",3:"Confirmed Case"},
    "Typhoid Fever":    {0:"Not a Case",1:"Suspected Case",2:"Probable Case",3:"Confirmed Case"},
}

# ==========================================================
# CATEGORICAL ENCODERS
# ==========================================================
categorical_encoders = {
    "sex": {"Male": 0, "Female": 1},
    "location": {"Urban": 0, "Rural": 1},
    "contact_case": {"No": 0, "Yes": 1},
}

# ==========================================================
# ENCODING FUNCTIONS
# ==========================================================
def encode_single_row(input_dict):
    encoded = {}
    for col, value in input_dict.items():
        if col in categorical_encoders:
            encoded[col] = categorical_encoders[col][value]
        else:
            encoded[col] = value
    return encoded

def encode_csv(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        if col in categorical_encoders:
            df[col] = df[col].map(categorical_encoders[col])
    return df

# ==========================================================
# PREDICTION FUNCTIONS
# ==========================================================
def predict_single(model, df, disease):
    raw = model.predict(df)[0]
    label = CASE_LABELS[disease].get(raw, f"Unknown ({raw})")
    return raw, label

def predict_batch(model, df, disease):
    raw_predictions = model.predict(df)
    labels = [CASE_LABELS[disease].get(p, f"Unknown ({p})") for p in raw_predictions]
    return raw_predictions, labels

# ==========================================================
# UI STYLING
# ==========================================================
st.markdown("""
<style>
.big-title {
    font-size: 36px;
    font-weight: bold;
    color: #2C3E50;
    text-align: center;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 22px;
    margin-top: 20px;
}
.confirmed { background-color: #27ae60 !important; }
.probable  { background-color: #f39c12 !important; }
.suspected { background-color: #e67e22 !important; }
.not-case  { background-color: #c0392b !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# MAIN PAGE
# ==========================================================
st.markdown('<p class="big-title">🩺 Disease Case Classification Prediction System</p>', unsafe_allow_html=True)
st.write("Predict case classifications for **Lassa Fever, Measles, Cholera, and Yellow Fever**.")

selected_disease = st.selectbox("Select Disease Model", list(models.keys()))
selected_model = models[selected_disease]

# ==========================================================
# SAFE FEATURE NAME EXTRACTION
# ==========================================================
if hasattr(selected_model, "get_booster"):  # XGBoost
    feature_names = selected_model.get_booster().feature_names
elif hasattr(selected_model, "feature_name_"):  # LightGBM
    feature_names = selected_model.feature_name_
elif hasattr(selected_model, "feature_names_in_"):  # Sklearn
    feature_names = list(selected_model.feature_names_in_)
else:
    feature_names = list(categorical_encoders.keys())  # fallback

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("📌 Model Information")
st.sidebar.write("### Expected Model Features:")
st.sidebar.info(", ".join(feature_names))
st.sidebar.write("### Categorical Encoders:")
st.sidebar.json(categorical_encoders)

# ==========================================================
# TABS
# ==========================================================
tab1, tab2 = st.tabs(["🧍 Single Prediction", "📤 CSV Upload"])

# ==========================================================
# TAB 1 - SINGLE PREDICTION
# ==========================================================
with tab1:
    st.subheader("Enter Patient Data")

    user_data = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):
        with (col1 if i % 2 == 0 else col2):
            if feature in categorical_encoders:
                user_data[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    list(categorical_encoders[feature].keys())
                )
            else:
                user_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=0.0
                )

    encoded = encode_single_row(user_data)
    df_input = pd.DataFrame([encoded], columns=feature_names)

    if st.button("🔍 Predict Case Classification"):
        raw_pred, label = predict_single(selected_model, df_input, selected_disease)

        color_class = {
            "Confirmed Case": "confirmed",
            "Probable Case": "probable",
            "Suspected Case": "suspected",
            "Not a Case": "not-case",
        }.get(label, "not-case")

        st.markdown(
            f'<div class="result-card {color_class}">Prediction: <b>{label}</b></div>',
            unsafe_allow_html=True
        )
        st.caption(f"Raw Model Output: {raw_pred}")

# ==========================================================
# TAB 2 - CSV UPLOAD
# ==========================================================
with tab2:
    st.subheader("Upload CSV File for Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV with the required feature columns", type=["csv"])

    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file)
        st.write("### Uploaded File Preview")
        st.dataframe(df_csv.head())

        missing_cols = [col for col in feature_names if col not in df_csv.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            df_encoded = encode_csv(df_csv)

            if st.button("📊 Run Batch Predictions"):
                raw_preds, labels = predict_batch(selected_model, df_encoded, selected_disease)

                df_results = df_csv.copy()
                df_results["Raw Prediction"] = raw_preds
                df_results["Case Classification"] = labels

                st.success("Batch prediction completed successfully!")
                st.write("### Prediction Results")
                st.dataframe(df_results)

                csv_file = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_file,
                    file_name="predictions_output.csv",
                    mime="text/csv"
                )
