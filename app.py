import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
import pandas as pd

# Custom deserialization function
def custom_binary_crossentropy(config):
    config.pop('fn', None)
    return BinaryCrossentropy(**config)

# Custom objects dictionary
custom_objects = {'BinaryCrossentropy': custom_binary_crossentropy}

# Load the trained model
try:
    model = load_model('model.h5', custom_objects=custom_objects)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Load the encoders and scaler
try:
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('label_encoder_geography.pkl', 'rb') as file:
        label_encoder_geography = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    st.write("Encoders and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading encoders or scaler: {e}")
    label_encoder_gender, label_encoder_geography, scaler = None, None, None

# Example prediction (replace with your actual prediction code)
if model and label_encoder_gender and label_encoder_geography and scaler:
    # Sample input data (replace with user input)
    sample_data = {
        'credit_score': 600,
        'geography': 'France',
        'gender': 'Male',
        'age': 40,
        'tenure': 3,
        'balance': 100000,
        'num_of_products': 2,
        'has_cr_card': 1,
        'is_active_member': 1,
        'estimated_salary': 50000
    }

    # Preprocess the input data (example)
    input_data = pd.DataFrame([sample_data])
    input_data['gender'] = label_encoder_gender.transform(input_data['gender'])
    input_data['geography'] = label_encoder_geography.transform(input_data['geography'])
    numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'num_of_products', 'estimated_salary']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction}")
