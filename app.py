import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model_path = "game_difficulty_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load dataset
df = pd.read_csv("student_game_data.csv")

# Streamlit UI
st.title("Game Difficulty Prediction & Data Analysis")

# Sidebar for Navigation
menu = st.sidebar.selectbox("Choose a Feature", ["Predict Difficulty", "Data Visualization"])

# ------------------------ SECTION 1: PREDICTION UI ------------------------
if menu == "Predict Difficulty":
    st.header("Enter Game Details for Difficulty Prediction")

    # User Input Fields
    score = st.number_input("Enter Score Achieved:", min_value=0.0, step=0.1)
    time_taken = st.number_input("Enter Time Taken (seconds):", min_value=0.0, step=0.1)
    attempts = st.number_input("Enter Number of Attempts:", min_value=1, step=1)
    hints = st.number_input("Enter Number of Hints Used:", min_value=0, step=1)
    sessions = st.number_input("Enter Sessions Per Week:", min_value=1, step=1)
    previous_difficulty = st.number_input("Enter Previous Difficulty Level:", min_value=0.0, step=0.1)

    # Prediction Button
    if st.button("Predict Difficulty"):
        input_data = np.array([[previous_difficulty, score, time_taken, attempts, hints, sessions]])
        predicted_difficulty = model.predict(input_data)[0]
        st.success(f"Predicted Game Difficulty Level: {predicted_difficulty:.2f}")

# ------------------------ SECTION 2: DATA VISUALIZATION ------------------------
elif menu == "Data Visualization":
    st.header("Visualizing Game Data Trends")

    # Heatmap Button
    if st.button("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Scatter Plot: Score vs. Predicted Difficulty
    if st.button("Show Score vs. Difficulty Scatterplot"):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=df["Score_Achieved"], y=df["Predicted_Difficulty"], ax=ax)
        ax.set_xlabel("Score Achieved")
        ax.set_ylabel("Predicted Difficulty")
        ax.set_title("Score vs. Predicted Difficulty")
        st.pyplot(fig)

    # Distribution Plot
    if st.button("Show Difficulty Level Distribution"):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df["Predicted_Difficulty"], bins=20, kde=True, ax=ax)
        ax.set_xlabel("Predicted Difficulty Level")
        ax.set_title("Distribution of Predicted Difficulty")
        st.pyplot(fig)

    # Line Plot
    if st.button("Show Time Taken vs. Difficulty Trend"):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x=df["Time_Taken"], y=df["Predicted_Difficulty"], ax=ax)
        ax.set_xlabel("Time Taken (seconds)")
        ax.set_ylabel("Predicted Difficulty")
        ax.set_title("Time Taken vs. Predicted Difficulty Trend")
        st.pyplot(fig)
