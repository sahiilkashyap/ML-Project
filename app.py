import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Wine Quality Classifier", layout="centered")

@st.cache_resource
def train_model():
    data = pd.read_csv("winequality-red.csv")

    # Binary classification: Good (1) if quality >= 7
    data['quality_label'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']
    
    X = data[features]
    y = data['quality_label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=3)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, features, test_accuracy

model, scaler, features, test_accuracy = train_model()

st.title("🍷 Wine Quality Classification")
st.markdown("Enter the features of wine below to predict if it's **Good** or **Bad** quality.")

fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, 7.0)
volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.1, 1.5, 0.5)
citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.5, 15.0, 2.5)
chlorides = st.slider("Chlorides (g/dm³)", 0.01, 0.2, 0.05)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", 1, 75, 15)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide (mg/dm³)", 6, 300, 45)
density = st.slider("Density (g/cm³)", 0.9900, 1.0050, 0.9960)
ph = st.slider("pH", 2.8, 4.0, 3.3)
sulphates = st.slider("Sulphates (g/dm³)", 0.3, 2.0, 0.6)
alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, 10.0)

if st.button("Classify Wine"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                            ph, sulphates, alcohol]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ Good Quality Wine ({prediction_proba:.1f}% confidence)")
    else:
        st.error(f"❌ Bad Quality Wine ({100 - prediction_proba:.1f}% confidence)")

st.markdown("---")
st.markdown(f"**Model Accuracy on Test Set:** {test_accuracy * 100:.2f}%")
st.caption("Developed by Sahil Kashyap | Streamlit + Logistic Regression")
