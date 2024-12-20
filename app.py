import os
import pandas as pd
import streamlit as st
from utils.data_generation import generate_synthetic_data
from utils.model import train_regression_model
from utils.evaluation import evaluate_model
import joblib

# Load data
st.title("Sports Analytics Project - Player Performance Simulation")


@st.cache_data
def load_data():
    # Check if the synthetic data file exists
    if not os.path.exists('data/synthetic_data.csv'):
        st.write("Generating synthetic data...")
        generate_synthetic_data()  # Generate data if file doesn't exist
    df = pd.read_csv('data/synthetic_data.csv')
    return df


df = load_data()

# Display the synthetic data (full dataset, scrollable)
st.subheader("Synthetic Data")
st.write("Displaying the entire dataset. You can scroll to explore all the rows.")
st.dataframe(df, height=500)  # Scrollable dataframe with height of 600px

# Check if the model is already trained
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.X_test = None
    st.session_state.y_test = None

# Train Model
if st.button("Train Model"):
    model, X_test, y_test = train_regression_model(df)

    # Save model and test data to session state to avoid retraining every time
    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.write("Model trained successfully!")

    # Evaluate Model
    mse, r2 = evaluate_model(model, X_test, y_test)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

# Ensure the model is loaded
if st.session_state.model:
    # Simulate future performance
    st.subheader("Simulate Player Performance")
    player_id = st.selectbox("Select Player ID", df['PlayerID'].unique())
    assists = st.slider("Assists", 0, 15)
    rebounds = st.slider("Rebounds", 0, 15)

    if st.button("Simulate"):
        # Create a DataFrame for the input with the same columns as the training data
        input_data = pd.DataFrame([[assists, rebounds]], columns=['Assists', 'Rebounds'])

        # Make the prediction using the trained model
        simulated_points = st.session_state.model.predict(input_data)[0]

        st.write(f"Simulated Points for Player {player_id}: {simulated_points:.2f}")

