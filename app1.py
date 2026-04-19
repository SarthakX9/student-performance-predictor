import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.title("Student Performance Predictor")

study_hours = st.slider("Study Hours", 0, 10)
attendance = st.slider("Attendance", 0, 100)
previous_marks = st.slider("Previous Marks", 0, 100)
assignments = st.slider("Assignments Completed", 0, 10)
sleep_hours = st.slider("Sleep Hours", 0, 10)

if st.button("Predict"):

    prediction = model.predict([[
        study_hours,
        attendance,
        previous_marks,
        assignments,
        sleep_hours
    ]])

    st.success(f"Predicted Marks: {prediction[0]:.2f}")