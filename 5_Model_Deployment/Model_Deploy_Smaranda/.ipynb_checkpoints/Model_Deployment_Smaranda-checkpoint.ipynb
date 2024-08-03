{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "18c1563f-c449-48e3-ac05-363eaa114619",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-07-30 11:53:43.630 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the best model\n",
    "model = joblib.load(r'C:\\Users\\Windows 11\\Desktop\\FSPapp folder\\GradientBoosting_best_model.pkl')\n",
    "\n",
    "# Title and instructions\n",
    "st.title('Flight Delay Prediction')\n",
    "st.write('Enter the flight details to predict if the arrival will be delayed.')\n",
    "\n",
    "# Define input fields\n",
    "arr_hour = st.number_input('Arrival Hour', min_value=0, max_value=23, value=12)\n",
    "dep_hour = st.number_input('Departure Hour', min_value=0, max_value=23, value=12)\n",
    "dep_time_day_interaction = st.number_input('Departure Time Day Interaction', min_value=0, max_value=100, value=1)\n",
    "distance_miles = st.number_input('Distance (Miles)', min_value=0, max_value=10000, value=500)\n",
    "is_weekend = st.selectbox('Is Weekend?', [0, 1])\n",
    "month = st.number_input('Month', min_value=1, max_value=12, value=1)\n",
    "number_of_flights = st.number_input('Number of Flights', min_value=0, max_value=1000, value=1)\n",
    "scheduled_arrival_time = st.number_input('Scheduled Arrival Time', min_value=0, max_value=2359, value=1200)\n",
    "scheduled_departure_time = st.number_input('Scheduled Departure Time', min_value=0, max_value=2359, value=1200)\n",
    "scheduled_gate_to_gate_time = st.number_input('Scheduled Gate to Gate Time', min_value=0, max_value=1000, value=120)\n",
    "week_day = st.number_input('Week Day', min_value=0, max_value=6, value=0)\n",
    "carrier_name = st.text_input('Carrier Name')\n",
    "destination_city_state = st.text_input('Destination City State')\n",
    "origin_city_state = st.text_input('Origin City State')\n",
    "\n",
    "# Button to make prediction\n",
    "if st.button('Predict'):\n",
    "    # Prepare input data\n",
    "    input_data = np.array([[\n",
    "        arr_hour, dep_hour, dep_time_day_interaction, distance_miles, is_weekend,\n",
    "        month, number_of_flights, scheduled_arrival_time, scheduled_departure_time,\n",
    "        scheduled_gate_to_gate_time, week_day, carrier_name, destination_city_state,\n",
    "        origin_city_state\n",
    "    ]])\n",
    "\n",
    "    # Apply preprocessing (scaling and encoding) to input data\n",
    "    input_data_preprocessed = model.named_steps['preprocessor'].transform(input_data)\n",
    "\n",
    "    # Make prediction\n",
    "    prediction = model.named_steps['classifier'].predict(input_data_preprocessed)\n",
    "    prediction_proba = model.named_steps['classifier'].predict_proba(input_data_preprocessed)\n",
    "\n",
    "    # Display results\n",
    "    st.write('Prediction:', 'Delayed' if prediction[0] else 'On Time')\n",
    "    st.write('Probability of Delay:', prediction_proba[0][1])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "738c0dae-fc04-4d5f-a000-554da9f12ae7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
