import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("../model/pressure_model.pkl","rb"))
tissue_encoder = pickle.load(open("../model/tissue_encoder.pkl","rb"))
risk_encoder = pickle.load(open("../model/risk_encoder.pkl","rb"))

st.title("AI Surgical Tissue Pressure Risk Prediction")

pressure = st.number_input("Enter Pressure Value (N)",0.0,5.0)

tissue = st.selectbox(
    "Select Tissue Type",
    ["nerve","muscle"]
)

if st.button("Predict Risk"):

    tissue_encoded = tissue_encoder.transform([tissue])[0]

    prediction = model.predict([[pressure,tissue_encoded]])

    risk = risk_encoder.inverse_transform(prediction)

    st.subheader("Predicted Risk Level")
    st.success(risk[0])