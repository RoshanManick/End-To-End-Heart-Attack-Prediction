import streamlit as st
import joblib
import numpy as np

# Custom CSS for styling and background image
st.markdown("""
    <style>
    /* Background image */
    body {
        background-image: url("https://media.istockphoto.com/id/1473675453/photo/well-balanced-diet-and-blood-pressure-control-for-heart-care.webp?s=2048x2048&w=is&k=20&c=PFrXUJF26KCQYvbzVTSYbaI8izhoxjH32RB8nCsfuw4=");
        background-size: 1500px;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Form and container styles */
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 600px;
        margin: auto;
    }

    /* Title styling */
    h1 {
        color: #444444;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }

    /* Input and button styles */
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }

    .stButton button {
        background-color: #5c6bc0;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }

    .stButton button:hover {
        background-color: #3949ab;
    }

    /* Prediction result styling */
    .stSubheader {
        color: #ff4c4c;
        text-align: center;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
model = joblib.load('heart_attack_model.pkl')

# Title of the app
st.title("Heart Attack Prediction ")

# Input fields for the user to enter data
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (1=normal; 2=fixed defect; 3=reversible defect)", min_value=1, max_value=3, value=2)

# Convert inputs to numpy array
inputs = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(inputs)[0]
    probability = model.predict_proba(inputs)[0][1]

    # Display the prediction
    st.subheader("Prediction Results")
    if prediction == 1:
        st.write(f"**Warning!** The model predicts a high risk of heart attack with a probability of {probability:.2f}.")
    else:
        st.write(f"**Good news!** The model predicts a low risk of heart attack with a probability of {probability:.2f}.")
