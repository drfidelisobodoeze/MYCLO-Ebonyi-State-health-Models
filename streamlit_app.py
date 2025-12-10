try:
    # Load non-Cholera models (assumed to be raw model objects)
    lassa_model = load_generic_model("lassa_xgb_9features.joblib")
    
    # Safe Measles model loading
    loaded = load_generic_model("measles.joblib")
    if isinstance(loaded, tuple):
        measles_model = loaded[0]  # extract the actual model
    else:
        measles_model = loaded

    yellow_fever_model = load_generic_model("yellow-fever.joblib")
    
    # Load the specific Cholera joblib file (which contains a dict)
    cholera_data = load_cholera_model("cholera.joblib")
    cholera_model = cholera_data['model']
    
    # Populate global Cholera metadata
    CHOLERA_FEATURES = cholera_data['features']
    CHOLERA_TARGET_MAP = cholera_data['target_map']
    
    models = {
        "Lassa Fever": lassa_model,
        "Measles": measles_model,
        "Cholera": cholera_model,
        "Yellow Fever": yellow_fever_model
    }
    
except FileNotFoundError as e:
    st.error(f"Model file not found. Please ensure all model files (including 'cholera.joblib') are in the correct path: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")
    st.stop()
