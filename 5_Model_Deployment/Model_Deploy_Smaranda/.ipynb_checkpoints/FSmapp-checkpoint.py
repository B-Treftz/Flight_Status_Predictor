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

def format_time(time_str):
    time_str = time_str.zfill(4)
    if time_str == '2400': 
        return '00:00'
    try: 
        time_obj = datetime.strptime(time_str, "%H%M")            
        return time_obj.strftime("%H:%M")
    except ValueError: 
        return time_str

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

######### initial setup 
currentYear = datetime.now().year
carrierNames = ['United Air Lines Inc.', 'Delta Air Lines Inc.', 
                'American Airlines Inc.', 'Southwest Airlines Co.',
                'Allegiant Air', 'JetBlue Airways', 'PSA Airlines Inc.',
                'Endeavor Air Inc.', 'Alaska Airlines Inc.',
                'Frontier Airlines Inc.', 'Envoy Air', 'Hawaiian Airlines Inc.',
                'SkyWest Airlines Inc.', 'Republic Airline', 'Spirit Air Lines']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
         'Oct', 'Nov', 'Dec']
days = [day for day in range(1,32)]
years = [year for year in range(currentYear, 2029)]
depTimeBlock = ['Early Morning', 'Morning', 'Early Afternoon', 'Afternoon', 
                'Evening','Night']

times_of_flights = ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2400']

@st.cache_resource
def load_model_and_create_preprocessor(): 
    model_path = r'C:\Users\Windows 11\Desktop\FSPapp folder\GradientBoosting_best_model.pkl'
    model = joblib.load(model_path)
    
    # Recreate the preprocessor
    numerical_features = ['Arr_Hour', 'Dep_Hour', 'Dep_Time_Day_Interaction', 'Distance_Miles', 'Is_Weekend', 'Month', 'Number_of_Flights', 'Scheduled_Arrival_Time', 'Scheduled_Departure_Time', 'Scheduled_Gate_to_Gate_Time', 'Week_Day']
    categorical_features = ['Carrier_Name', 'Destination_City_State', 'Origin_City_State']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return model, preprocessor

model, preprocessor = load_model_and_create_preprocessor()

def flightPredict(input_data): 
    # Ensure input_data is a DataFrame
    X = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(X)
    prediction = model.predict(X_processed)[0]
    return prediction

######### About the app ###############
markdown_about_msg = """
        Welcome to the Flight Status Predictor App!
    
        This app aims to predict whether the arrival time of the flight 
        will be delayed for more than 15 minutes or not.
    """
st.markdown(markdown_about_msg)

############ Set the parameters ##########################

# Dropdown menu to select carrier 
selectCarrier = st.selectbox('Select :orange[Carrier]', options=carrierNames)
    
# Dropdown menu to select year
selectYear = st.selectbox('Select :orange[Year]', options=years)

# Dropdown menu to select month 
selectMonth = st.selectbox('Select :orange[Month]', options=months)

emp = st.empty()
if selectYear and selectMonth: 
    month_index = months.index(selectMonth) + 1
    days_in_month = get_days_in_month(selectYear, month_index)
    days = list(range(1, days_in_month + 1))
    selectDay = st.selectbox('Select :orange[Day]', options=days)
else: 
    selectDay = emp.selectbox('Select :orange[Day]', options=days, disabled=True)
        
selectTimeOfDay = st.selectbox('Select :orange[Departure Time of Day]', options=depTimeBlock)
    
depTimes = update_times(selectTimeOfDay, times_of_flights)
selectScheduledDepTime = st.selectbox('Select :orange[Scheduled Departure Time]', options=depTimes)
if selectScheduledDepTime:
    hours, minutes = selectScheduledDepTime.split(':')
    selectScheduledDepTime = int(hours) * 100 + int(minutes)
        
arrTimes = update_times(selectTimeOfDay, times_of_flights)
selectScheduledArrTime = st.selectbox('Select :orange[Scheduled Arrival Time]', options=arrTimes)
if selectScheduledArrTime:
    hours, minutes = selectScheduledArrTime.split(':')
    selectScheduledArrTime = int(hours) * 100 + int(minutes)
        
# Compute Dep_Time_Day_Interaction
def compute_dep_time_day_interaction(dep_time, day):
    # Example interaction: just a simple product for demonstration
    return dep_time * day

dep_time_day_interaction = compute_dep_time_day_interaction(selectScheduledDepTime, selectDay)

# Map months to numerical values
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
selectMonth_num = month_mapping[selectMonth]

if selectCarrier and selectMonth_num and selectDay and selectYear and selectTimeOfDay and selectScheduledDepTime and selectScheduledArrTime:
    if st.button('Generate Flight Prediction'): 
        input_data = {
            'Carrier_Name': selectCarrier, 
            'Dep_Time_Day_Interaction': dep_time_day_interaction,
            'Month': selectMonth_num, 
            'Year': selectYear, 
            'Day': selectDay, 
            'Scheduled_Arrival_Time': selectScheduledArrTime, 
            'Scheduled_Departure_Time': selectScheduledDepTime,
            'Arr_Hour': int(selectScheduledArrTime // 100),
            'Dep_Hour': int(selectScheduledDepTime // 100),
            'Distance_Miles': st.number_input('Distance (Miles)', min_value=0, max_value=10000, value=500),
            'Is_Weekend': st.selectbox('Is Weekend?', [0, 1]),
            'Number_of_Flights': st.number_input('Number of Flights', min_value=0, max_value=1000, value=1),
            'Scheduled_Gate_to_Gate_Time': st.number_input('Scheduled Gate to Gate Time', min_value=0, max_value=1000, value=120),
            'Week_Day': st.number_input('Week Day', min_value=0, max_value=6, value=0),
            'Destination_City_State': st.text_input('Destination City State'),
            'Origin_City_State': st.text_input('Origin City State')
        }
            
        result = flightPredict(input_data)
            
        if result == 1: 
            st.error('Flight Delayed :thumbsdown:')
        else: 
            st.success('Flight Not Delayed :thumbsup:')
else: 
    flightPredictButton = st.button('Generate Flight Prediction', disabled=True)
