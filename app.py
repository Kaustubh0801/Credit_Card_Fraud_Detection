import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('credit_card_fraud_detection_model')

# Streamlit app
st.title("Fraud Detection System")

st.write("""
### Predict the likelihood of a transaction being fraudulent
Enter the details for the 29 features below to predict:
""")

# List of feature names based on the dataset
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Create input fields for all 29 features with suggested ranges
user_inputs = []
for feature in feature_names:
    if feature == "Amount":
        value = st.number_input(f"{feature}", min_value=0.0, max_value=10000.0, value=50.0)  # Adjust range for Amount
    else:
        value = st.number_input(f"{feature}", min_value=-100.0, max_value=100.0, value=0.0)  # Adjust range for `V1` to `V28`
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
