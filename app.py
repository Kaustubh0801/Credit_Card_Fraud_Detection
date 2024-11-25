import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load(r'C:\Users\kaust\Downloads\credit card_fraud_detection_model')

# Streamlit app
st.title("Fraud Detection System")

st.write("""
### Predict the likelihood of a transaction being fraudulent
Enter the details for the 29 features below to predict:
""")

# Dynamically create input fields for all 29 features
user_inputs = []
for i in range(1, 30):  # Features 1 to 29
    value = st.number_input(f"Feature {i}", min_value=0.0, max_value=100.0, value=50.0)
    user_inputs.append(value)

# Convert user inputs into a numpy array
user_data = np.array([user_inputs])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)

    if prediction[0] == 1:
        st.error("This transaction is predicted to be Fraudulent!")
    else:
        st.success("This transaction is predicted to be Genuine.")

    st.write("Prediction Probability:", prediction_proba)
