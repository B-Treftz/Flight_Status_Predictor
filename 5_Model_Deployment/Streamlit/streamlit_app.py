import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('best_rf_model.pkl', 'rb') as file:
    best_rf = pickle.load(file)

# Define the user interface
st.title('Flight Delay Prediction')

# Input fields
year = st.selectbox('Year', [2023, 2024])
month = st.selectbox('Month', list(range(1, 13)))
day = st.selectbox('Day', list(range(1, 32)))
dep_time_block = st.selectbox('Departure Time Block', [
    'Night', 'Early Morning', 'Evening', 'Morning', 'Afternoon', 'Early Afternoon'])
carrier = st.selectbox('Carrier', [
    'Southwest Airlines Co.', 'United Air Lines Inc.', 'American Airlines Inc.',
    'Spirit Air Lines', 'SkyWest Airlines Inc.', 'Delta Air Lines Inc.',
    'Endeavor Air Inc.', 'PSA Airlines Inc.', 'Envoy Air',
    'Hawaiian Airlines Inc.', 'Republic Airline', 'JetBlue Airways',
    'Allegiant Air', 'Frontier Airlines Inc.', 'Alaska Airlines Inc.'
])

# Predict button
if st.button('Predict'):
    # Prepare the features as a DataFrame
    features = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Dep_Time_Block_Group': [dep_time_block],
        'Carrier': [carrier]
    })
    
    # Preprocess the features
    preprocessed_features = preprocessor.transform(features)
    
    # Make prediction
    prediction = best_rf.predict(preprocessed_features)
    
    # Display the result
    if prediction[0] == 1:
        st.write('The flight will likely be delayed by 15 minutes or more.')
    else:
        st.write('The flight will likely not be delayed by 15 minutes or more.')

