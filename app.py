import streamlit as st
# Customizing app appearance with Streamlit themes
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded"
)
# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.write("### Hello, Rakesh! Let's predict if a Person is Diabetic or Not")
st.title("Welcome to the Diabetes Prediction App!")
st.write("This app helps predict diabetes risk based on input parameters.")

import streamlit as st
import numpy as np
import pickle

# Load your trained model (ensure you have the .pkl model in the same folder)
# Replace 'model.pkl' with your actual trained model file
model_path = 'model.pkl'
try:
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    model = None
    st.error("Trained model file not found. Please ensure 'model.pkl' is in the app directory.")

# Set up the app layout
st.title("Diabetes Prediction App")
st.subheader("Enter all details")

# Create columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Age")
    pregnancies = st.text_input("Pregnancies")
    glucose = st.text_input("Glucose")
    blood_pressure = st.text_input("Blood Pressure")

with col2:
    insulin = st.text_input("Insulin")
    bmi = st.text_input("BMI")
    skin_thickness = st.text_input("Skin Thickness")
    dpf = st.text_input("DPF (Diabetes Pedigree Function)")

# Predict button
if st.button("Predict"):
    # Validate input fields
    if not all([age, pregnancies, glucose, blood_pressure, insulin, bmi, skin_thickness, dpf]):
        st.error("Please fill in all fields.")
    else:
        try:
            # Convert inputs to floats
            input_data = np.array([[
                float(pregnancies), float(glucose), float(blood_pressure),
                float(skin_thickness), float(insulin), float(bmi),
                float(dpf), float(age)
            ]])

            # Prediction
            if model:
                prediction = model.predict(input_data)
                result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
                st.success(f"Prediction: {result}")
            else:
                st.error("Prediction model is not loaded.")
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
