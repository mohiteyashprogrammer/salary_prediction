import os
import sys
import pandas as pd
import streamlit as st
from src.pipline.prediction_pipline import PredictPipline

data = pd.read_csv(os.path.join("artifcats","train.csv"))

st.title("Salary Prediction")

job_titles = data['Job Title'].unique()

gender =  ['Male','Female']

education_level =  ['PhD', "Bachelor's Degree", "Master's Degree",'High School']


# Create dropdowns for user input
selected_job_title = st.selectbox("Select a Job Title", sorted(job_titles))
selected_gender = st.selectbox("Select Gender", gender)
selected_education_level = st.selectbox("Select Education Level", education_level)
years_of_experience = st.number_input("Years of Experience")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [selected_gender],
        "Education Level": [selected_education_level],
        "Job Title": [selected_job_title],
        "Years of Experience": [years_of_experience],
    })

    predict_pipline = PredictPipline()
    predicted_salary = predict_pipline.predict(input_data)

    st.header("Predicted Salary Is: $%.0f" % predicted_salary)


