import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page configuration
st.set_page_config(page_title="Real-Time Fraud Detection System", layout="wide")

# Title
st.title("Real-Time Credit Card Fraud Detection")

# 1. Load Dataset (Assessment 1)
@st.cache_data
def load_data():
    # Loading the provided credit_card_fraud_10k.csv
    df = pd.read_csv('credit_card_fraud_10k.csv')
    return df

df = load_data()

# 2. Data Cleaning & Preprocessing (Assessment 2)
def preprocess_data(data):
    # Cleaning column names (lower casing and removing unwanted characters)
    data.columns = [c.lower().strip() for c in data.columns]
    
    # Handle categorical data: merchant_category
    le = LabelEncoder()
    data['merchant_category_encoded'] = le.fit_transform(data['merchant_category'])
    
    # Define features and target
    features = ['amount', 'transaction_hour', 'merchant_category_encoded', 
                'foreign_transaction', 'location_mismatch', 'device_trust_score', 
                'velocity_last_24h', 'cardholder_age']
    X = data[features]
    y = data['is_fraud']
    
    return X, y, le

X, y, label_encoder = preprocess_data(df)

# 3. Model Training (Assessment 3 - Logistic Regression)
# Assessment 4: Split data into training and testing (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Evaluation (Assessment 4)
st.header("Model Performance Evaluation")
y_pred = model.predict(X_test_scaled)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2%}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2%}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.2%}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(pd.DataFrame(cm, columns=['Predicted Legit', 'Predicted Fraud'], index=['Actual Legit', 'Actual Fraud']))

st.info("Generalization: The model's performance on unseen test data indicates how well it adapts to new transactions. "
        "Misclassifications in the confusion matrix highlight instances where the model failed to correctly identify fraud or flagged legitimate transactions.")

# 5. Real-Time Demonstration
st.divider()
st.header("Real-Time Fraud Simulation")

with st.form("transaction_form"):
    st.write("Enter transaction details to simulate live input:")
    amt = st.number_input("Amount", min_value=0.0, value=100.0)
    hour = st.slider("Transaction Hour", 0, 23, 12)
    category = st.selectbox("Merchant Category", df['merchant_category'].unique())
    foreign = st.selectbox("Foreign Transaction", [0, 1])
    mismatch = st.selectbox("Location Mismatch", [0, 1])
    trust = st.slider("Device Trust Score", 0, 100, 80)
    velocity = st.number_input("Velocity Last 24h", min_value=0, value=1)
    age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=30)
    
    submit = st.form_submit_button("Predict")

if submit:
    start_time = time.time()
    
    # Prepare input
    cat_encoded = label_encoder.transform([category])[0]
    input_data = np.array([[amt, hour, cat_encoded, foreign, mismatch, trust, velocity, age]])
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)
    end_time = time.time()
    latency = end_time - start_time
    
    if prediction[0] == 1:
        st.error("Result: Potential Fraud Detected")
    else:
        st.success("Result: Transaction Legitimate")
        
    st.write(f"Inference Latency: {latency:.4f} seconds")

# 6. AI Exploration Experiment (Option 1: Unexpected Inputs)
st.divider()
st.header("AI Exploration: Unexpected Inputs")
st.write("Testing how the model reacts to extreme scenarios.")

exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.write("Scenario A: Extremely high amount at 3 AM")
    if st.button("Run High Amount Test"):
        # Simulated high risk input
        test_input = np.array([[50000.0, 3, label_encoder.transform(['Travel'])[0], 1, 1, 10, 15, 25]])
        test_scaled = scaler.transform(test_input)
        res = model.predict(test_scaled)
        st.write(f"Prediction: {'Fraud' if res[0] == 1 else 'Legit'}")
        st.write("Reason: High amounts combined with unusual hours and low trust scores often trigger fraud flags in Logistic Regression models.")

with exp_col2:
    st.write("Scenario B: Unusual merchant for age group")
    if st.button("Run Merchant Test"):
        test_input = np.array([[10.0, 14, label_encoder.transform(['Electronics'])[0], 0, 0, 95, 1, 95]])
        test_scaled = scaler.transform(test_input)
        res = model.predict(test_scaled)
        st.write(f"Prediction: {'Fraud' if res[0] == 1 else 'Legit'}")
        st.write("Reason: The model evaluates patterns; low-risk indicators like high trust and local transaction typically result in 'Legit' regardless of age.")

# 7. Deployment Evaluation
st.divider()
st.header("Deployment Evaluation")
st.write("""
Response speed: The system is very fast, processing predictions in less than 1 second.
Lag: There is no noticeable lag during transaction simulation or data processing.
Stability: The Streamlit interface remains stable across multiple input tests.
Challenges: One major challenge is data imbalance, as fraud cases are much rarer than legitimate ones. 
Another challenge is maintaining low real-time latency when scaling the feature engineering process.
""")