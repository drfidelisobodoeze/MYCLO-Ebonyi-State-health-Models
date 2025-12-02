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
# Helper function to make predictions
def make_prediction(model, features, input_data):
    model_instance = model['model']
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model_instance.predict(input_df)
    return prediction[0]

import streamlit as st

# Custom CSS styles for a mobile-friendly design
st.markdown("""
    <style>
    /* Base styling for the body */
    body {
        background-color: #f4f4f9;
        font-family: 'Roboto', sans-serif;
    }

    /* Adjust the layout for smaller screens (mobile devices) */
    @media screen and (max-width: 600px) {
        .main {
            max-width: 100%;
            margin: 0;
            padding: 10px;
        }

        h1 {
            font-size: 2em;
            text-align: center;
        }

        /* Mobile-friendly input field sizes */
        .stNumberInput input, .stSelectbox select, .stButton button {
            width: 100%;
            font-size: 14px;
            padding: 10px;
        }

        /* Make the sidebar more compact on mobile */
        .sidebar .sidebar-content {
            padding: 10px;
        }
    }

    /* Desktop view styling (large screens) */
    @media screen and (min-width: 601px) {
        .main {
            max-width: 800px;
            margin: auto;
        }

        h1 {
            font-size: 2.5em;
            font-weight: 600;
            text-align: center;
            color: #333;
        }

        .stSelectbox select {
            padding: 10px;
            font-size: 16px;
        }

        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
    }

    /* Styling for prediction results */
    .stSuccess {
        background-color: #e8f5e9;
        color: #388e3c;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)

# The rest of your Streamlit app goes here...

# Streamlit app interface
    
st.title("MYCLO-Ebonyi-State-health-Models: Multiple Infectious Diseases Prediction using ensemble models")
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
