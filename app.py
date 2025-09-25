import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('LinearRegression_Model.joblib')

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.markdown("Enter the house details to get a price prediction.")

# Create input widgets for the user
square_feet = st.number_input("Square Feet", min_value=500.0, max_value=4000.0, value=2000.0)
num_rooms = st.number_input("Number of Rooms", min_value=2, max_value=7, value=3)
age = st.number_input("Age (years)", min_value=0, max_value=99, value=30)
distance_to_city = st.number_input("Distance to City (km)", min_value=1.0, max_value=30.0, value=15.0)

# Create a button to trigger the prediction
if st.button("Predict Price"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[square_feet, num_rooms, age, distance_to_city]],
                              columns=['square_feet', 'num_rooms', 'age', 'distance_to_city(km)'])

    # Make the prediction
    prediction = model.predict(input_data)
    predicted_price = prediction[0]

    # Display the prediction result
    st.subheader("Prediction:")
    st.success(f"The predicted house price is: ${predicted_price:,.2f}")