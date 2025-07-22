import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Data Preparation
def load_data():
    data = {
        'Age': [3, 5, 2, 7, 1, 4, 6, 3, 2, 5],
        'Mileage': [35000, 60000, 20000, 90000, 15000, 40000, 75000, 36000, 25000, 62000],
        'EngineSize': [1.5, 1.2, 2.0, 1.3, 2.2, 1.6, 1.4, 1.8, 2.0, 1.3],
        'NumDoors': [4, 4, 2, 4, 2, 4, 4, 2, 2, 4],
        'Price': [500000, 320000, 600000, 250000, 650000, 450000, 300000, 470000, 620000, 310000]
    }
    df = pd.DataFrame(data)
    return df

# Train-Test Split
def split_data(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Train Random Forest
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "predictions": y_pred
    }

# Save model
def save_model(model, filename='models/car_price_model.pkl'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, filename)

# Load model
def load_model(filename='models/car_price_model.pkl'):
    return joblib.load(filename)
