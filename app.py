# import numpy as np
# import pandas as pd 
# import streamlit as st 
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import joblib
# import warnings 
# warnings.filterwarnings('ignore')
# data = pd.read_csv('Diabetes_Prediction (Project).csv')
# df = data.copy()



# st.markdown("<h1 style = 'color:#0802A3; text-align: center; font-family: Arial Black; font-size:42px'>DIABETES PREDICTION</h1>", unsafe_allow_html=True)

# st.markdown("<h4 style = 'margin: -30px; color: #000000; text-align: center; font-family: cursive;font-size:32px'>Built By Ismail Ibitoye</h4>", unsafe_allow_html = True)

# st.markdown("<br>", unsafe_allow_html = True)
# st.image('pngwing.com (1).png', width = 350, use_column_width = True)
# st.markdown("<br>", unsafe_allow_html = True)
# st.markdown("<p style=font-family:Comic Sans>Diabetes prediction involves analyzing various factors such as BMI, age, and other relevant health indicators to assess the likelihood of developing diabetes. By examining these columns in a dataset, predictive models can identify patterns and correlations indicative of diabetes risk. Features like BMI, age, blood glucose levels, and family medical history are crucial predictors utilized in model development. Leveraging machine learning algorithms, such as logistic regression or decision trees, enables the creation of predictive models capable of accurately forecasting diabetes risk. The incorporation of additional factors, such as lifestyle choices and dietary habits, enhances the predictive power of the model, aiding in early detection and intervention strategies.Ultimately, the goal of diabetes prediction is to notify individual of his or her health status.</p>", 
#              unsafe_allow_html = True)
# st.markdown('<br>', unsafe_allow_html = True)
# st.dataframe(data, use_container_width = True)  

# st.sidebar.image('pngwing.com (2).png', caption = 'welcome user')



# blood_pressure = st.sidebar.number_input('Blood_pressure_level', data['BloodPressure'].min(), data['BloodPressure'].max())
# skin_thickness = st.sidebar.number_input('Skin_thickness', data['SkinThickness'].min(), data['SkinThickness'].max())
# Insulin = st.sidebar.number_input('Insulin_level', data['Insulin'].min(), data['Insulin'].max())
# bmi = st.sidebar.number_input('BMI', data['BMI'].min(), data['BMI'].max())
# age = st.sidebar.number_input('Age', data['Age'].min(), data['Age'].max())
# Diabetic_functionality = st.sidebar.number_input('Diabetic_Functionality', data['DiabetesPedigreeFunction'].min(), data['DiabetesPedigreeFunction'].max())
# glucose = st.sidebar.number_input('Glucose_level', data['Glucose'].min(), data['Glucose'].max())


# input_var = pd.DataFrame({ 'BloodPressure':[blood_pressure], 
#                           'SkinThickness':[skin_thickness], 'Insulin':[Insulin], 'BMI':[bmi], 
#                            'Age': [age], 'DiabetesPedigreeFunction' :[Diabetic_functionality], 'Glucose' : glucose})
# st.dataframe(input_var)
# model = joblib.load('diabetes.pkl')
# prediction = st.button('Press to predict')

# if prediction:
#     predicted = model.predict(input_var)
#     output = None
#     if predicted == 1:
#         output = 'Diabetic'
#     else:
#         output = 'Non-Diabetic'
#     st.success(f'The result of this analysis shows that this individual is {output}')
#     st.balloons()


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
