import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv("diabetes_data_upload.csv")

# Encode categorical variables
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0})
dataset['class'] = dataset['class'].map({'Positive': 1, 'Negative': 0})
dataset['Polyuria'] = dataset['Polyuria'].map({'Yes': 1, 'No': 0})
dataset['Polydipsia'] = dataset['Polydipsia'].map({'Yes': 1, 'No': 0})
dataset['sudden weight loss'] = dataset['sudden weight loss'].map({'Yes': 1, 'No': 0})
dataset['weakness'] = dataset['weakness'].map({'Yes': 1, 'No': 0})
dataset['Polyphagia'] = dataset['Polyphagia'].map({'Yes': 1, 'No': 0})
dataset['Genital thrush'] = dataset['Genital thrush'].map({'Yes': 1, 'No': 0})
dataset['visual blurring'] = dataset['visual blurring'].map({'Yes': 1, 'No': 0})
dataset['Itching'] = dataset['Itching'].map({'Yes': 1, 'No': 0})
dataset['Irritability'] = dataset['Irritability'].map({'Yes': 1, 'No': 0})
dataset['delayed healing'] = dataset['delayed healing'].map({'Yes': 1, 'No': 0})
dataset['partial paresis'] = dataset['partial paresis'].map({'Yes': 1, 'No': 0})
dataset['muscle stiffness'] = dataset['muscle stiffness'].map({'Yes': 1, 'No': 0})
dataset['Alopecia'] = dataset['Alopecia'].map({'Yes': 1, 'No': 0})
dataset['Obesity'] = dataset['Obesity'].map({'Yes': 1, 'No': 0})

# Select features and target
X = dataset[['Polydipsia', 'sudden weight loss', 'partial paresis', 'Irritability', 'Polyphagia', 'Age', 'visual blurring']]
y = dataset['class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Build a simple neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, validation_data=(X_test_scaled, y_test))


# Save trained model
model.save("my_model.h5")

# Save the scaler for later use
import joblib
joblib.dump(scaler, "scaler.pkl")
