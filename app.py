import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Customer Booking Prediction",
    page_icon="✈️",
    layout="centered"
)

# Load trained model
# In a real deployment, ensure 'model.pkl' is available
# For this example, we'll assume it's created and accessible.
# If you run this in Colab, you'd need to create 'model.pkl' first (e.g., pickle.dump(model, open('model.pkl', 'wb')))

# Placeholder for model loading to avoid FileNotFoundError during app.py creation
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the model is saved to disk.")
    st.stop() # Stop the Streamlit app if model not found

# Title
st.title("✈️ Customer Booking Prediction System")

st.markdown("""
This application predicts whether a customer is likely to complete a booking.
""")

# Sidebar
st.sidebar.header("About Project")

st.sidebar.info("""
Machine Learning model built using Random Forest Classifier.

This project predicts customer booking completion behavior.
""")

# Input Section
st.subheader("Enter Customer Details")

purchase_lead = st.number_input(
    "Purchase Lead",
    min_value=0,
    max_value=500,
    value=30
)

length_of_stay = st.number_input(
    "Length of Stay",
    min_value=1,
    max_value=50,
    value=5
)

flight_hour = st.slider(
    "Flight Hour",
    0,
    23,
    12
)

# Predict Button
if st.button("Predict Booking"):

    input_data = pd.DataFrame({
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour]
    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Customer is likely to complete booking")
    else:
        st.error("❌ Customer is unlikely to complete booking")

    st.write(f"Prediction Confidence: {probability:.2%}")

    st.progress(int(probability * 100))

# Footer
st.markdown("---")
st.caption("Built with Streamlit and Scikit-Learn")
