import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('heart_attack_model.pkl')

# Title of the app
st.title("Heart Attack Prediction")

# Set background image and custom CSS
st.markdown("""
    <style>
    body {
        background-image: url("https://static.vecteezy.com/system/resources/previews/006/662/453/non_2x/medical-stethoscope-and-heart-pulse-red-on-the-blue-plank-floor-small-and-large-red-heart-shaped-models-3d-rendering-free-photo.jpg"); /* Replace with your image URL */
        background-size: cover;
        background-size: 1000x 1000px;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #333;
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        border-radius: 10px;
        padding: 30px;
        max-width: 600px;
        margin: auto;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    h1 {
        color: #1e88e5; /* Blue color */
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 32px;
        margin-bottom: 20px;
    }

    .stButton button {
        background-color: #1e88e5; /* Blue */
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-weight: bold;
        width: 100%;
    }

    .stButton button:hover {
        background-color: #1565c0; /* Darker blue on hover */
    }

    /* Input fields */
    input, select {
        margin-bottom: 15px;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        width: 100%;
        box-sizing: border-box;
    }

    /* Prediction result styling */
    .stSubheader {
        text-align: center;
        font-size: 20px;
        margin-top: 20px;
        font-weight: bold;
        color: #d32f2f; /* Red for warnings */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Parameters")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.sidebar.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.sidebar.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2, value=1)
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
slope = st.sidebar.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.sidebar.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thal = st.sidebar.number_input("Thalassemia (1=normal; 2=fixed defect; 3=reversible defect)", min_value=1, max_value=3, value=2)

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
