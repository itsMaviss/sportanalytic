import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_generation import generate_player_data, generate_match_data
from utils.model import train_model


def main():
    st.title("Badminton Analytics Application")

    # Initialize session state variables
    if "player_data" not in st.session_state:
        st.session_state.player_data = None
    if "match_data" not in st.session_state:
        st.session_state.match_data = None
    if "eda_fig" not in st.session_state:
        st.session_state.eda_fig = None
    if "training_fig" not in st.session_state:
        st.session_state.training_fig = None
    if "eval_matrix" not in st.session_state:
        st.session_state.eval_matrix = None
    if "conf_matrix" not in st.session_state:
        st.session_state.conf_matrix = None

    # Step 1: Data Generation
    st.header("1. Synthetic Data Generation")

    # Display the sliders before the button to allow adjusting values
    num_players = st.slider("Number of Players", 50, 200, 50)
    num_matches = st.slider("Number of Matches", 100, 500, 500)

    if st.button("Generate Data"):
        # Clear the previous data and graphs when generating new data
        st.session_state.eda_fig = None
        st.session_state.training_fig = None
        st.session_state.player_data = None
        st.session_state.match_data = None
        st.session_state.eval_matrix = None
        st.session_state.conf_matrix = None

        # Generate new data and store in session state
        st.session_state.player_data = generate_player_data(num_players)
        st.write("### Player Data", st.session_state.player_data)

        st.session_state.match_data = generate_match_data(st.session_state.player_data, num_matches)
        st.write("### Match Data", st.session_state.match_data)

    # Step 2: Exploratory Data Analysis (EDA)
    st.header("2. Exploratory Data Analysis")
    if st.session_state.player_data is not None:
        if st.button("Perform EDA"):
            st.subheader("Player Stats Distribution")
            fig, ax = plt.subplots()
            sns.boxplot(data=st.session_state.player_data.drop("PlayerID", axis=1), ax=ax)
            ax.set_title("EDA")
            st.session_state.eda_fig = fig
        if st.session_state.eda_fig is not None:
            st.pyplot(st.session_state.eda_fig)
    else:
        st.warning("Please generate data first.")

    st.header("3. Train Model")
    if st.session_state.player_data is not None:
        # Step 3: Modeling
        if st.button("Train Model"):
            model, accuracy, precision, recall, conf_matrix = train_model(
                st.session_state.player_data, st.session_state.match_data
            )
            # Confusion Matrix Heatmap
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Predicted Negative", "Predicted Positive"],
                        yticklabels=["True Negative", "True Positive"])
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.session_state.conf_matrix = fig

            metrics = ["Accuracy", "Precision", "Recall"]
            values = [accuracy, precision, recall]
            fig, ax = plt.subplots()
            ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
            ax.set_title("Evaluation Matrix")
            ax.set_ylim(0, 1)  # Metrics are between 0 and 1
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')  # Annotate bars
            st.session_state.eval_matrix = fig

            fig, ax = plt.subplots()
            feature_importances = model.feature_importances_
            features = [
                "P1_Speed", "P1_Stamina", "P1_Agility",
                "P2_Speed", "P2_Stamina", "P2_Agility"
            ]
            ax.bar(features, feature_importances, color='skyblue')
            ax.set_title("Feature Importance")
            ax.set_ylabel("Importance")
            st.session_state.training_fig = fig

        # Only render the feature importance graph if it exists
        if st.session_state.conf_matrix is not None:
            st.pyplot(st.session_state.conf_matrix)
        if st.session_state.eval_matrix is not None:
            st.pyplot(st.session_state.eval_matrix)
        if st.session_state.training_fig is not None:
            st.pyplot(st.session_state.training_fig)

    else:
        st.warning("Please generate data first.")

    # Step 4: Simulation
    st.header("4. Simulation")
    if st.session_state.player_data is not None:
        # Track the player's selection in session state to persist across rerenders
        if "selected_player" not in st.session_state:
            st.session_state.selected_player = None
        if "show_player_selectbox" not in st.session_state:
            st.session_state.show_player_selectbox = False  # Keep track of whether the dropdown is visible

        if st.button("Simulate Match"):
            # Show player selection dropdown after clicking the button
            st.session_state.show_player_selectbox = True

        if st.session_state.show_player_selectbox:
            # Show the selectbox once "Simulate Match" is clicked
            selected_player = st.selectbox(
                "Select a Player",
                st.session_state.player_data["PlayerID"],
                key="player_selectbox"
            )
            st.session_state.selected_player = selected_player  # Save the selection in session state

        if st.session_state.selected_player is not None:
            # Display selected player data once a player has been chosen
            selected_data = st.session_state.player_data[
                st.session_state.player_data["PlayerID"] == st.session_state.selected_player
                ]
            st.write("Selected Player Data:", selected_data)
    else:
        st.warning("Please generate data first.")


if __name__ == "__main__":
    main()
