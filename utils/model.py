import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


def train_regression_model(data):
    # Prepare features (X) and target variable (y)
    X = data[['Assists', 'Rebounds']]
    y = data['Points']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'models/regression_model.pkl')

    return model, X_test, y_test
