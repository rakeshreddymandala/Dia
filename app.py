import streamlit as st
import numpy as np
import pickle
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Set Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("🚨 API key for Google Generative AI is missing. Disease analysis will not work.")



# Header Section
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction App</h1>
    <p style='text-align: center;'>🎯 Predict your diabetes risk and receive AI-powered health advice!</p>
    """,
    unsafe_allow_html=True
)

# Initialize session state for result
if "result" not in st.session_state:
    st.session_state.result = None

# Load the trained model
model_path = "model.pkl"
try:
    model = pickle.load(open(model_path, "rb"))
except FileNotFoundError:
    model = None
    st.error("❌ Trained model file not found. Please ensure 'model.pkl' is in the app directory.")

# Input Form Section
st.subheader("Enter Your Details:")
with st.form("user_inputs"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age", value="")
        pregnancies = st.text_input("Pregnancies", value="")
        glucose = st.text_input("Glucose Level", value="")
        blood_pressure = st.text_input("Blood Pressure Level", value="")

    with col2:
        insulin = st.text_input("Insulin Level", value="")
        bmi = st.text_input("BMI", value="")
        skin_thickness = st.text_input("Skin Thickness", value="")
        dpf = st.text_input("Diabetes Pedigree Function (DPF)", value="")

    submitted = st.form_submit_button("Predict")

# Helper function to validate numeric inputs
def validate_inputs(inputs):
    try:
        return np.array([[float(i) for i in inputs]])
    except ValueError:
        st.error("⚠️ All fields must contain valid numeric values.")
        return None

# Handle Prediction
if submitted:
    input_data = validate_inputs([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    if input_data is not None:
        if model:
            with st.spinner("Analyzing your data..."):
                prediction = model.predict(input_data)
                st.session_state.result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
                st.success(f"🩺 Prediction: **{st.session_state.result}**")
        else:
            st.error("❌ Prediction model is not loaded.")

# Helper function to get AI response
def get_gemini_response(prompt, user_input):
    try:
        if not api_key:
            return "Error: API key is not set. Please configure it in your environment variables."

        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([prompt, user_input])

        if response and hasattr(response, "text"):
            return response.text
        else:
            return "Error: No response text received from the AI."
    except Exception as e:
        return f"Error: {str(e)}"

# AI Analysis Section
if st.button("Analyze Disease"):
    if not st.session_state.result:
        st.error("⚠️ Please predict the disease first.")
    else:
        with st.spinner("Analyzing your condition..."):
            prompt = """
            You are an expert doctor. Analyze the condition based on the following prediction and provide:
            - Likely causes of the condition.
            - Preventive measures and care tips.
            - Nutritional and lifestyle suggestions.
            Avoid referencing any copyrighted material, and write the analysis in your own words.
            """
            response = get_gemini_response(prompt, st.session_state.result)

        if response.startswith("Error:"):
            st.error(response)
        elif not response.strip():
            st.error("❌ Received an empty response from the AI.")
        else:
            st.success("✔️ Analysis Complete!")
            st.subheader("Detailed Response:")
            st.write(response)

# Disclaimer Section
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: small;'>
    ⚠️ This application is for informational purposes only and not a substitute for professional medical advice.
    Always consult a healthcare provider for medical concerns.
    </p>
    """,
    unsafe_allow_html=True
)
