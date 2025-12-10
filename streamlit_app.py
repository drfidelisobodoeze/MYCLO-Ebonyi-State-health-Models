# Global variables for model metadata (must be at the top of the file)
CHOLERA_FEATURES = []
CHOLERA_TARGET_MAP = {}
YF_FEATURES = [] # ADDED for Yellow Fever metadata
YF_TARGET_MAP = {} # ADDED for Yellow Fever metadata

# ============================================================
# LOAD MODELS (FIXED FOR TUPLE ISSUE & METADATA)
# ============================================================
@st.cache_resource
def load_generic_model(path):
    """Loads Lassa/Measles model, handling case where it's saved as a tuple/list."""
    loaded_data = joblib.load(path)
    
    # FIX: Check if the loaded data is a tuple/list and extract the model (element 0)
    if isinstance(loaded_data, (tuple, list)):
        return loaded_data[0]
    
    # Check if the loaded data is a dictionary and contains the model key
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        return loaded_data['model']

    # Otherwise, assume the loaded object is the model itself
    return loaded_data

@st.cache_resource
def load_metadata_model(path, feature_list_global, target_map_global):
    """Loads dictionary containing model and metadata (for Cholera, YF)."""
    data = joblib.load(path)
    
    # Populate global metadata variables (uses globals() to modify the lists/dicts defined above)
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
    st.error(f"An error occurred while loading models: {e}. Check if model files are correctly structured.")
    st.stop()
