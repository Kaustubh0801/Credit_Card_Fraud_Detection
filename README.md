# Credit Card Fraud Detection System

## Overview
The Credit Card Fraud Detection System predicts whether a given transaction is genuine or fraudulent based on features extracted from the transaction data. This project utilizes a machine learning model trained on the Credit Card Fraud Detection Dataset from Kaggle.

The deployed web application allows users to input transaction details interactively and receive predictions about the likelihood of fraud in real time.

## Features
- **Interactive Web Interface**: Powered by Streamlit, enabling users to input feature values and view results dynamically.
- **Machine Learning Model**: Random Forest Classifier to classify transactions as Genuine or Fraudulent.
- **Probability Scores**: Provides probabilities for predictions to indicate the model's confidence.

## Dataset
The dataset used for this project is the Credit Card Fraud Detection Dataset. It contains anonymized transaction data with 30 features:
- **Features V1-V28**: Principal Component Analysis (PCA)-transformed features.
- **Amount**: The transaction amount.
- **Class**: The target variable (0 for Genuine, 1 for Fraudulent).

## Deployment
The application is deployed and accessible at the following link:  
[Credit Card Fraud Detection App](https://creditcardfrauddetection-kahgptm4uf66ulmtquqgdu.streamlit.app/)

## Usage
1. Enter values for the features (V1 to V28 and Amount) using the sliders or input fields provided in the web interface.
2. Click the **Predict** button to receive:
   - A prediction of whether the transaction is `Genuine` or `Fraudulent`.
   - The prediction probability scores.

## Project Files
- `app.py`: Contains the Streamlit web application code.
- `credit_card_fraud_detection_model.pkl`: The trained machine learning model.
- `requirements.txt`: Python dependencies required to run the application.
- `credit_card_fraud_detection.ipynb`: Jupyter Notebook with data exploration, preprocessing, and model training.

## Model Training
The model was trained using the following steps:

### Data Preprocessing:
- Balanced the dataset using under-sampling and oversampling to address the class imbalance.
- Scaled numerical features using StandardScaler.

### Model Selection:
- Various classifiers were evaluated, and the best-performing model was selected.

### Evaluation Metrics:
- Precision, Recall, and F1 Score were used to evaluate performance.

## Technologies Used
- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-learn, NumPy, and Pandas

## Results
The model achieved:
- High accuracy on the test set.
- Effective fraud detection with minimal false negatives.

## License
This project is licensed under the MIT License.
