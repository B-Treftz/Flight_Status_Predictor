from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class PrepProcesor(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        # Define categorical and numerical features
        self.numerical_features = ['Scheduled_Arrival_Time', 'Scheduled_Departure_Time', 'Year', 'Day']
        self.categorical_features = ['Carrier_Name', 'Dep_Time_Block_Group', 'Month']
    
    def fit(self, X, y=None):
        # Fit the scaler on numerical features
        self.scaler.fit(X[self.numerical_features])
        
        # Fit the encoder on categorical features
        self.encoder.fit(X[self.categorical_features])
        
        return self
    
    def transform(self, X, y=None):
        # Create a copy of the input data
        X_transformed = X.copy()
        
        # Scale numerical features
        X_transformed[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        
        # Encode categorical features
        encoded_features = self.encoder.transform(X[self.categorical_features]).toarray()
        encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features)
        
        # Add encoded features to the dataframe
        for i, col in enumerate(encoded_feature_names):
            X_transformed[col] = encoded_features[:, i]
        
        # Drop original categorical columns
        X_transformed = X_transformed.drop(columns=self.categorical_features)
        
        # Convert final result to a numpy array
        return X_transformed.values