from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import pandas as pd
import joblib

app = Flask(__name__)
Bootstrap(app)

model_path = 'models/flight_model.pkl'
preprocessor_path = 'preprocessor/flight_preprocessor.pkl'

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

######### App routing and prediction ###############

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gather data from the form
    carrierName = request.form.get('carrier')
    year = request.form.get('year')
    month = request.form.get('month')
    day = request.form.get('day')
    timeOfDay = request.form.get('depTimeBlock')
    scheduledDepTime = request.form.get('scheduledDepTime')
    scheduledArrTime = request.form.get('scheduledArrTime') 
    
    # Create a DataFrame
    data = {
        'Carrier_Name': [carrierName],
        'Dep_Time_Block_Group': [timeOfDay],
        'Month': [month],
        'Year': [year],
        'Day': [day],
        'Scheduled_Arrival_Time': [scheduledArrTime],
        'Scheduled_Departure_Time': [scheduledDepTime]
    }
    df = pd.DataFrame(data)
    
    x_processed = preprocessor.transform(df)
        
    prediction = model.predict(x_processed)[0]
    prediction = int(prediction)

    if (prediction == 1): 
        message = 'Flight Delayed'
    else: 
        message = 'Flight Not Delayed'
    
    # Return the result as JSON
    return jsonify(message=message)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
    




















