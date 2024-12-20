import matplotlib.pyplot as plt
import streamlit as st  # Import Streamlit here
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot actual vs predicted points
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Ideal line')
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points')
    plt.title('Actual vs Predicted Points')
    plt.legend()

    # Use Streamlit's method to display the plot
    st.pyplot(plt.gcf())  # Pass the current figure object

    return mse, r2
