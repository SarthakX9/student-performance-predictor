import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Student Performance Predictor")

study_hours = st.slider("Study Hours", 0, 10)
attendance = st.slider("Attendance", 0, 100)
previous_marks = st.slider("Previous Marks", 0, 100)
assignments = st.slider("Assignments Completed", 0, 10)
sleep_hours = st.slider("Sleep Hours", 0, 10)

if st.button("Predict"):

    features = np.array([[
        study_hours,
        attendance,
        previous_marks,
        assignments,
        sleep_hours
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.success(f"Predicted Marks: {prediction[0]:.2f}")