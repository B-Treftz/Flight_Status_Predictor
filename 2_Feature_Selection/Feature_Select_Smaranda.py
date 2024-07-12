#!/usr/bin/env python
# coding: utf-8

# **FLIGHT STATUS PREDICTOR PROJECT

# *The goal of this project is to develop and deploy an ML model in which an end user can specify a set of features describing a commercial flight of interest and receive a categorical (yes or no) output indicating if the arrival time of that flight will be delayed more than fifteen minutes or not. 
# 

# **WEEK 1: CLEANING, FORMATING & EDA

# In[1]:


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Import dataset
df = pd.read_csv('downsampled_data_updated.csv')


# In[3]:


# Check dataframe
df.head()


# In[4]:


# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)


# In[5]:


# Check columns to identify target variables
print (df.columns)


# In[6]:


# Define the target variable
target = 'Arr_Delay_At_Least_15_Minutes'


# In[7]:


# Define feature variables excluding the target
features = [col for col in df.columns if col != target]


# *To avoid information leakage, it's essential to ensure that none of the features provide direct or indirect information about the arrival delay that wouldn't be available at the time of prediction.
# 
# Will drop 'Dep_Delay_At_Least_15_Minutes' as is highly likely to leak information about the target variable because departure delays often correlate strongly with arrival delays. Including this column would give the model access to information that is too directly related to the target.
# 
# Other Potential Columns to consider dropping are any columns that are calculated using data that would not be known until the flight has been completed or that provide direct information about the arrival delay should be excluded. These might include actual arrival times and any metrics or status indicators updated during or after the flight.

# In[8]:


# Identify and drop columns that might cause leakage
leakage_columns = ['Dep_Delay_At_Least_15_Minutes', 'Arr_Time_Block_Group']
features = [col for col in df.columns if col not in leakage_columns + [target]]


# In[9]:


# Split the data into features and target variables
X = df[features]
y = df[target]


# In[10]:


# Handle missing values
X = X.dropna()

# Ensure y matches the index after dropping missing values
y = y.loc[X.index]  

# Check changes
print("Features:", X.columns)
print("Target:", y.name)


# In[11]:


# Identify num of delayed flights
num_delayed_flights = df[df[target] == True].shape[0]
print(f"Number of delayed flights: {num_delayed_flights}")


# In[12]:


# Distribution of Airlines
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Carrier_Name')
plt.title('Distribution of Airlines')
plt.xticks(rotation=75)
plt.show()


# In[13]:


# Airlines with most delays 
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Carrier_Name', hue=target)
plt.title('Carrier with Most Delays')
plt.xlabel('Airline')
plt.ylabel('Number of Flights')
plt.xticks(rotation=75)
plt.legend(title='Delayed 15+ Minutes')
plt.show()


# In[14]:


# Calculate the percentage of delays for each carrier
carrier_delay_percentages = df.groupby('Carrier_Name')[target].mean() * 100

# Reset index for plotting
carrier_delay_percentages = carrier_delay_percentages.reset_index()

# Plot the percentage of delays
plt.figure(figsize=(12, 6))
barplot = sns.barplot(data=carrier_delay_percentages, x='Carrier_Name', y=target)
plt.title('Carrier with Most Delays')
plt.xlabel('Airline')
plt.ylabel('Percentage of Flights Delayed by 15+ Minutes')
plt.xticks(rotation=75)

# Add percentage labels on the bars
for p in barplot.patches:
    barplot.annotate(f'{p.get_height():.1f}%', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     xytext=(0, 9), 
                     textcoords='offset points')

plt.show()


# ##### FEATURE ENGINEERING & FEATURE SELECTION

# In[15]:


# Import new libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder


# **Identify numerical and categorical variables

# In[16]:


# Get data types of each column
data_types = X.dtypes

# Initialize lists to hold the names of categorical and numerical variables
categorical_vars = []
numerical_vars = []

# Identify numerical and categorical variables based on their data types
for column, dtype in data_types.items():
    if pd.api.types.is_numeric_dtype(dtype):
        numerical_vars.append(column)
    else:
        categorical_vars.append(column)

# Print the identified variables
print("Numerical Variables:")
print(numerical_vars)

print("Categorical Variables:")
print(categorical_vars)


# **Handling Outliers

# In[17]:


# Visualize the distribution of numerical features
for feature in numerical_vars:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=False, bins=25)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# In[18]:


# Handle outliers by capping at the 99th percentile
for feature in numerical_vars:
    cap_value = df[feature].quantile(0.99)
    df.loc[:, feature] = df[feature].apply(lambda x: min(x, cap_value))


# #Re-visualize the capped data
# for feature in numerical_vars:
#     plt.figure(figsize=(8, 4))
#     sns.histplot(df[feature], kde=False, bins=25)
#     plt.title(f'Histogram of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.show()

# In[19]:


# Create a correlation matrix
correlation_features = numerical_vars + [target]
correlation_matrix = df[correlation_features].corr()


# In[20]:


# Plot the correlation matrix using a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Selected Features')
plt.show()


# In[39]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import psutil
import os

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

# Monitor memory usage before encoding
print(f"Memory usage before encoding: {memory_usage():.2f} MB")

# One Hot Encoding with sparse output
encoder = OneHotEncoder(sparse_output=True, drop='first')
encoded_features = encoder.fit_transform(df[categorical_vars])

# Combine sparse matrix with numerical features
numerical_features = df[numerical_vars].values
X_combined = sp.hstack((encoded_features, numerical_features))

# Monitor memory usage after encoding
print(f"Memory usage after encoding: {memory_usage():.2f} MB")


# In[22]:


# Assuming target variable
y = np.random.randint(0, 2, df.shape[0])

# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

print("Data split successfully")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


# In[45]:


from sklearn.ensemble import RandomForestClassifier
# Try a sample data for initial testing
X_train_sample = X_train[:10000]
y_train_sample = y_train[:10000]

print(f"Memory usage before training: {memory_usage():.2f} MB")

# Train model with optimized parameters
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_sample, y_train_sample)

print(f"Memory usage after training: {memory_usage():.2f} MB")

# Get feature importances
importances = model.feature_importances_

# Get the feature names for encoded categorical columns
encoded_feature_names = encoder.get_feature_names_out(categorical_vars)
all_feature_names = np.concatenate([encoded_feature_names, numerical_vars])

# Create feature importance DataFrame
feature_importance_df = pd.DataFrame({'Feature': all_feature_names , 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance',  ascending=False)
print(feature_importance_df)


# In[ ]:




