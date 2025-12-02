import streamlit as st
import joblib

# Load the models
lassa_xgb = joblib.load("lassa_xgb.joblib")  # Ensure correct path here
measles_model = joblib.load("measles.joblib")
cholera_model = joblib.load("cholera_lgb.joblib")
yellow_fever_model = joblib.load("yellow-fever.joblib")

models = {
    "Lassa Fever": lassa_xgb,
    "Measles": measles_model,
    "Cholera": cholera_model,
    "Yellow Fever": yellow_fever_model
}

# Function to safely extract top 10 features
def get_top_features(model):
    if isinstance(model, dict) and "features" in model:
        return model["features"]
    return []

# Function to make predictions
def make_prediction(model, features, input_data):
    model_instance = model["model"] if isinstance(model, dict) else model
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model_instance.predict(input_df)
    return prediction[0]

# Streamlit app interface
st.title("Disease Case Classification Prediction App")

disease = st.selectbox("Select Disease", list(models.keys()))

selected_model = models[disease]

# Display top 10 features for the selected disease
top_features = get_top_features(selected_model)

if top_features:
    st.write(f"### Top 10 Features for {disease}")
    st.write(top_features)

# Prediction form
input_data = {}
for feature in top_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

if st.sidebar.button("Predict"):
    if not top_features:
        st.error("No feature list found for the selected model.")
    else:
        prediction = make_prediction(selected_model, top_features, input_data)
        st.success(f"Prediction for {disease}: {prediction}")
