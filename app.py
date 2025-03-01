import streamlit as st
import numpy as np
import pickle

# Load the trained model
model_path = "game_difficulty_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# UI Title
st.title("Game Difficulty Prediction")

# User Input Fields
score = st.number_input("Enter Score Achieved:", min_value=0.0, step=0.1)
time_taken = st.number_input("Enter Time Taken (seconds):", min_value=0.0, step=0.1)
attempts = st.number_input("Enter Number of Attempts:", min_value=1, step=1)
hints = st.number_input("Enter Number of Hints Used:", min_value=0, step=1)
sessions = st.number_input("Enter Sessions Per Week:", min_value=1, step=1)
previous_difficulty = st.number_input("Enter Previous Difficulty Level:", min_value=0.0, step=0.1)

# Prediction Button
if st.button("Predict Difficulty"):
    # Prepare input data for prediction
    input_data = np.array([[previous_difficulty, score, time_taken, attempts, hints, sessions]])
    predicted_difficulty = model.predict(input_data)[0]
    
    # Display the predicted difficulty level
    st.success(f"Predicted Game Difficulty Level: {predicted_difficulty:.2f}")
