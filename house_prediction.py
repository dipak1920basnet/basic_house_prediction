"""
Basic Project: House Price Prediction (Linear Regression)

Description: Build a model to predict house prices based on features such as size, number of rooms, and location.
Tasks:
Implement data preprocessing and feature scaling.
Train a multiple linear regression model with gradient descent.
Evaluate the model using mean squared error.

"""

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd 
data = fetch_california_housing()
column = data.feature_names
x = data.data
y = data.target
new_data = pd.DataFrame(x, columns=column)
new_data['Price'] = y

## Data Preprocessing and Features Scaling
print(new_data.head())