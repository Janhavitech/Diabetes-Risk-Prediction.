import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Load the dataset and fit the scaler
dataset = pd.read_csv('diabetes_data_upload.csv')
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0})
dataset['class'] = dataset['class'].map({'Positive': 1, 'Negative': 0})
dataset['Polydipsia'] = dataset['Polydipsia'].map({'Yes': 1, 'No': 0})
dataset['sudden weight loss'] = dataset['sudden weight loss'].map({'Yes': 1, 'No': 0})
dataset['partial paresis'] = dataset['partial paresis'].map({'Yes': 1, 'No': 0})
dataset['Irritability'] = dataset['Irritability'].map({'Yes': 1, 'No': 0})
dataset['Polyphagia'] = dataset['Polyphagia'].map({'Yes': 1, 'No': 0})
dataset['visual blurring'] = dataset['visual blurring'].map({'Yes': 1, 'No': 0})

X = dataset[['Polydipsia', 'sudden weight loss', 'partial paresis', 'Irritability', 'Polyphagia', 'Age', 'visual blurring']]
scaler = StandardScaler()
scaler.fit(X)

# Streamlit UI
st.title("Diabetes Prediction")

polydipsia = st.selectbox("Polydipsia:", [0, 1])
sudden_weight_loss = st.selectbox("Sudden Weight Loss:", [0, 1])
partial_paresis = st.selectbox("Partial Paresis:", [0, 1])
irritability = st.selectbox("Irritability:", [0, 1])
polyphagia = st.selectbox("Polyphagia:", [0, 1])
age = st.slider("Age:", 10, 90, 30)
visual_blurring = st.selectbox("Visual Blurring:", [0, 1])

if st.button("Predict"):
    input_data = np.array([[polydipsia, sudden_weight_loss, partial_paresis, irritability, polyphagia, age, visual_blurring]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    probability = prediction[0][0]

    result = "Positive" if probability > 0.5 else "Negative"
    st.success(f"Prediction: {result} (Probability: {probability:.2f})")
