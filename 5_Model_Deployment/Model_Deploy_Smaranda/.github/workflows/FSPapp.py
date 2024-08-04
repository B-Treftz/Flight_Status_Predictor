import tracemalloc
import streamlit as st 
import numpy as np 
import pandas as pd
from datetime import datetime
import calendar
import joblib 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Enable tracemalloc
tracemalloc.start()

# Set page configuration first
st.set_page_config(page_title='FSP@streamlit', layout='centered')

# Custom CSS to change the selector label color to purple
st.markdown(
    """
    <style>
    .stSelectbox label {
        color: purple;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS to change button color to purple
st.markdown(
    """
    <style>
    .stButton button {
        background-color: purple;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model_and_preprocessor():
    # Assuming the files are in the same directory as FSPapp.py
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'GradientBoosting_best_model_full.pkl')
    preprocessor_path = os.path.join(base_path, 'preprocessor_full.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor file not found: {preprocessor_path}")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# Function to preprocess user input 
def flightPredict(input_data): 
    # List of all required features
    all_features = [
        'Carrier_Name', 'Month', 'Year', 'Day', 
        'Scheduled_Arrival_Time', 'Scheduled_Departure_Time', 
        'Destination_City_State', 'Origin_City_State',
        'Arr_Hour', 'Is_Weekend', 'Dep_Hour', 
        'Dep_Time_Day_Interaction', 'Distance_Miles', 
        'Week_Day', 'Scheduled_Gate_to_Gate_Time', 'Number_of_Flights'
    ]

    # Add missing features with default values
    for feature in all_features:
        if feature not in input_data:
            input_data[feature] = 0  # Default value, you can customize this
    
    X = pd.DataFrame([input_data])

    # Preprocess the input data
    X_processed = preprocessor.transform(X)

# Function to format time for easier processing
def format_time(time_str):
    time_str = time_str.zfill(4)
    if time_str == '2400': 
        return '00:00'
    try: 
        time_obj = datetime.strptime(time_str, "%H%M")            
        return time_obj.strftime("%H:%M")
    except ValueError: 
        return time_str

# Function to return the sorted list of formatted times ### app functionality to be extended with detailed time blocks in a future version
def update_times(selectedTimeOfDay, times_of_flights): 
    flightTimes = { 
        'Early Morning': times_of_flights,
        'Morning': times_of_flights,
        'Early Afternoon': times_of_flights,
        'Afternoon': times_of_flights,
        'Evening': times_of_flights,
        'Night': times_of_flights
    }
    times = flightTimes.get(selectedTimeOfDay, [])
    formatted_times = sorted([format_time(time) for time in times])
    return formatted_times

def get_days_in_month(year, month):
    _, days = calendar.monthrange(year, month)
    return days

######### About the app ###############
markdown_about_msg = """
        Welcome to the Flight Status Predictor App!    
        Enter your flight details below to predict if the arrival will be delayed.
    """
st.markdown(markdown_about_msg)

######### Initial setup #############
carrierNames = ['United Air Lines Inc.', 'Delta Air Lines Inc.', 
                'American Airlines Inc.', 'Southwest Airlines Co.',
                'Allegiant Air', 'JetBlue Airways', 'PSA Airlines Inc.',
                'Endeavor Air Inc.', 'Alaska Airlines Inc.',
                'Frontier Airlines Inc.', 'Envoy Air', 'Hawaiian Airlines Inc.',
                'SkyWest Airlines Inc.', 'Republic Airline', 'Spirit Air Lines']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
         'Oct', 'Nov', 'Dec']
days = [day for day in range(1,32)]
currentYear = datetime.now().year
years = [year for year in range(currentYear, 2029)]
depTimeBlock = ['Early Morning', 'Morning', 'Early Afternoon', 'Afternoon', 
                'Evening','Night']

times_of_flights = ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2400']

######## Define selectors ############
selectCarrier = st.selectbox('Select Carrier', options=carrierNames)
selectYear = st.selectbox('Select Year', options=years)
selectMonth = st.selectbox('Select Month', options=months)
selectDay = st.selectbox('Select Day', options=days)
selectScheduledArrTime = st.selectbox('Select Scheduled Arrival Time', options=times_of_flights)
selectScheduledDepTime = st.selectbox('Select Scheduled Departure Time', options=times_of_flights)
destination_city_state = st.text_input('Destination City State')
origin_city_state = st.text_input('Origin City State')

# Map months to numerical values
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
selectMonth_num = month_mapping[selectMonth]

# Button click event
if st.button('Generate Flight Prediction'): 
    input_data = {
        'Carrier_Name': selectCarrier, 
        'Month': selectMonth_num, 
        'Year': selectYear, 
        'Day': selectDay, 
        'Scheduled_Arrival_Time': int(selectScheduledArrTime),
        'Scheduled_Departure_Time': int(selectScheduledDepTime),
        'Destination_City_State': destination_city_state,
        'Origin_City_State': origin_city_state
    }

    
    # Call the flightPredict function to get the prediction
    result = flightPredict(input_data)
    
    # Display the prediction result to the user
    if result == 1: 
        st.error('Sorry, your flight is delayed.')  
    else: 
        st.success('Happy travel, your flight is on time!')  
