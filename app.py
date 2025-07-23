import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
# Load model and scaler
model = load_model(r'C:\Users\shari\OneDrive\Desktop\test\salary_predictor_model.keras')
with open(r'C:\Users\shari\OneDrive\Desktop\test\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("👩‍💼 Employee Salary Predictor (>50K or <=50K)")

st.markdown("Enter employee details to predict whether their income exceeds 50K.")

# Input fields (you can expand to more features based on adult.csv)
age = st.number_input("Age", 17, 90, 30)
education_num = st.slider("Education Level (numeric)", 1, 16, 10)
hours_per_week = st.slider("Hours per week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.number_input("Capital Loss", 0, 99999, 0)

# Dummy input for the rest of the features (you can add all if needed)
# Make sure input shape matches model input
input_data = np.zeros((1, 14))  # adult.csv has 14 features excluding income
input_data[0][0] = age
input_data[0][4] = education_num  # Index depends on actual order
input_data[0][10] = hours_per_week
input_data[0][8] = capital_gain
input_data[0][9] = capital_loss

# Scale input
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    pred = model.predict(input_scaled)
    result = ">50K" if pred[0][0] > 0.5 else "<=50K"
    st.success(f"💰 Predicted Salary: {result}")
