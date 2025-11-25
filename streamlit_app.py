import streamlit as st
import joblib
import pandas as pd

# Load all four models
lassa_model = joblib.load('lassa_fever_lightgbm_model.joblib')
yellow_fever_model = joblib.load('yellow_fever_lightgbm_model.joblib')
measles_model = joblib.load('measles_lightgbm_model.joblib')
cholera_model = joblib.load('cholera_lightgbm_model.joblib')

# Streamlit app title
st.title('Disease Outcome Classification')

# Disease Selection
st.header('Select Disease Model')
disease = st.selectbox('Choose Disease', ['Lassa Fever', 'Yellow Fever', 'Measles', 'Cholera'])

# Input form for each disease
if disease == 'Lassa Fever':
    st.header('Input Data for Lassa Fever')
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    outcome_of_case = st.selectbox('Outcome of Case', ['Recovered', 'Deceased'])
    input_data = {
        'Sex': sex,
        'Age': age,
        'Outcome_of_case': outcome_of_case
    }
    input_df = pd.DataFrame([input_data])
    model = lassa_model

elif disease == 'Yellow Fever':
    st.header('Input Data for Yellow Fever')
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    outcome_of_case = st.selectbox('Outcome of Case', ['Recovered', 'Deceased'])
    input_data = {
        'Sex': sex,
        'Age': age,
        'Outcome_of_case': outcome_of_case
    }
    input_df = pd.DataFrame([input_data])
    model = yellow_fever_model

elif disease == 'Measles':
    st.header('Input Data for Measles')
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    temperature_category = st.selectbox('Temperature Category', ['Normal', 'High'])
    input_data = {
        'Sex': sex,
        'Age': age,
        'Temperature Category': temperature_category
    }
    input_df = pd.DataFrame([input_data])
    model = measles_model

else:
    st.header('Input Data for Cholera')
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    temperature_category = st.selectbox('Temperature Category', ['Normal', 'High'])
    input_data = {
        'Sex': sex,
        'Age': age,
        'Temperature Category': temperature_category
    }
    input_df = pd.DataFrame([input_data])
    model = cholera_model

# Convert categorical columns to category type (needed for LightGBM to recognize them)
categorical_columns = ['Sex', 'Outcome_of_case', 'Temperature Category']
for col in categorical_columns:
    input_df[col] = input_df[col].astype('category')

# Prediction button
if st.button('Predict'):
    # Make the prediction
    prediction = model.predict(input_df)
    st.write(f'Predicted Disease Outcome: {prediction[0]}')

# Sidebar: Show model info
st.sidebar.header('Model Information')
st.sidebar.write('This app uses four disease classification models trained with LightGBM.')
st.sidebar.write('You can select a disease, enter the data, and predict the disease outcome.')

