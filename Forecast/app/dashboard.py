import streamlit as st
from utils.data_preprocessing import load_data
from models.predict import predict

st.title("Solar Energy Production Forecasting Tool")
data_file = st.file_uploader("Upload your data", type=['csv'])

if data_file:
    data = load_data(data_file)
    st.write("Uploaded Data:")
    st.dataframe(data)
    
    model_path = "models/solar_model.pkl"
    predictions = predict(data, model_path)
    st.write("Predictions:")
    st.write(predictions)
