import streamlit as st
import numpy as np
import pickle
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("API key for Google Generative AI is missing. Disease analysis will not work.")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# App title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.write("### Predict diabetes risk and get AI-powered health advice!")

# Initialize session state for result
if "result" not in st.session_state:
    st.session_state.result = None

# Load the trained model
model_path = 'model.pkl'
try:
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    model = None
    st.error("Trained model file not found. Please ensure 'model.pkl' is in the app directory.")

# Input fields
st.subheader("Enter the details:")
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

# Helper function to validate numeric inputs
def validate_inputs(inputs):
    try:
        return np.array([[float(i) for i in inputs]])
    except ValueError:
        st.error("All fields must contain valid numeric values.")
        return None

# Predict button
if st.button("Predict"):
    input_data = validate_inputs([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    if input_data is not None:
        if model:
            prediction = model.predict(input_data)
            st.session_state.result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
            st.success(f"Prediction: {st.session_state.result}")
        else:
            st.error("Prediction model is not loaded.")

# Helper function to get AI response
def get_gemini_response(prompt, user_input):
    try:
        # Ensure API key is set and generative model is initialized
        if not api_key:
            return "Error: API key is not set. Please configure it in your environment variables."

        # Call the Generative AI model
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([prompt, user_input])

        # Process response
        if response and hasattr(response, "text"):
            return response.text
        else:
            return "Error: No response text received from the model."
    except Exception as e:
        return f"Error: {str(e)}"

# Analyze Disease button logic
if st.button("Analyze Disease"):
    if not st.session_state.result:
        st.error("Please predict the disease first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Prompt for AI assistance
                prompt = """
                You are an expert doctor. Analyze the condition based on the following prediction and provide:
                - Likely causes of the condition.
                - Preventive measures and care tips.
                - Nutritional and lifestyle suggestions.
                Avoid referencing any copyrighted material, and write the analysis in your own words.
                write Disclaimer at the end
                """
                # Get AI response
                response = get_gemini_response(prompt, st.session_state.result)
                
                # Display the response
                if response.startswith("Error:"):
                    st.error(response)
                elif not response.strip():
                    st.error("Received an empty response from the AI.")
                else:
                    st.success("Analysis Complete!")
                    st.subheader("Detailed Response:")
                    st.write(response)
                    # Add a disclaimer at the end of the app
                    

            except Exception as e:
                st.error(f"Unexpected error: {e}")


