from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd

app = Flask(__name__)

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('best_rf_model.pkl', 'rb') as file:
    best_rf = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from request
        data = request.get_json()
        features = {
            'Year': data['Year'],
            'Month': data['Month'],
            'Day': data['Day'],
            'Dep_Time_Block_Group': data['Dep_Time_Block_Group'],
            'Carrier': data['Carrier']
        }

        # Convert features to DataFrame
        features_df = pd.DataFrame([features])

        # Preprocess the features
        preprocessed_features = preprocessor.transform(features_df)

        # Make prediction
        prediction = best_rf.predict(preprocessed_features)

        if prediction[0] == 1:
            message = "The flight will arrive delayed by 15 minutes or more."
        else:
            message = "The flight will not be delayed by 15 minutes or more."

        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)

