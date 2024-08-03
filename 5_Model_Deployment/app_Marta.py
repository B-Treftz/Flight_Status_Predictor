# app2.py
import streamlit as st
#from utils import PrepProcessor, columns

import pickle
import numpy as np
import pandas as pd
#import joblib



# Load the trained model
with open('model_Marta.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Will your flight be late?')


# Input features
feature1 = st.selectbox('Carrier name', ['Southwest Airlines Co.','Delta Air Lines Inc.','American Airlines Inc.','United Air Lines Inc.','SkyWest Airlines Inc.','Republic Airline','Spirit Air Lines','JetBlue Airways','Alaska Airlines Inc.','Envoy Air','Endeavor Air Inc.','PSA Airlines Inc.','Frontier Airlines Inc.','Allegiant Air','Hawaiian Airlines Inc.' ])
feature2 = st.selectbox('Week day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
feature3 = st.selectbox('Departure Time Block Group',['Early Morning','Morning', 'Early Afternoon', 'Afternoon','Night'])
feature4 = st.number_input('Scheduled Departure Time',1200)
feature5 = st.number_input('Distance Group',min_value=1, max_value=11, value=1)

# Predict button
if st.button('Predict'):
    # Make prediction
    features = np.array([[feature1, feature2, feature3, feature4, feature5]])
    prediction = model.predict(features)
    
    st.write(f'Prediction: {prediction[0]}')