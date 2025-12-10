# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_generic_model_safe(path):
    """
    Load a joblib model and automatically extract from tuple if necessary
    """
    loaded = joblib.load(path)
    if isinstance(loaded, tuple):
        return loaded[0]  # Extract model from tuple
    return loaded

@st.cache_resource
def load_cholera_model(path):
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
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading models. Check your joblib files and paths.")
    st.stop()
