import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Title of the app
st.title("Heart Disease Prediction App")
st.write("""
This app predicts the likelihood of heart disease based on user input.
Fill in the details below to get a prediction.
""")

# Load the saved trained model
model_filename = "heart_disease_model.pkl"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)


# Function to collect user input
def user_input_features():
    age = st.number_input("Age", min_value=10, max_value=120, value=40, step=1)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox(
        "Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic)",
        [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
    restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.number_input("ST Depression Induced by Exercise (0.0-6.0)", min_value=0.0, max_value=6.0, value=1.0,
                              step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0, step=1)
    thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)", [1, 2, 3])

    # Combine the inputs into a DataFrame
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([input_data])


# Get user input
input_features = user_input_features()

# Show the input data in the app
st.subheader("Your Input Data:")
st.write(input_features)

# Predict based on user input and display the result
# Predict based on user input and display the result
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_features)
    heart_disease_prob = prediction_proba[0][1]  # Probability of class 1 (Heart Disease)

    # Display risk level
    st.subheader("Risk Assessment:")
    if heart_disease_prob < 0.33:
        st.success("🟢 You are at **Low Risk**. No need to see a doctor, but maintain a healthy lifestyle.")
    elif 0.33 <= heart_disease_prob < 0.66:
        st.warning("🟠 You are at **Moderate Risk**. Consider improving your lifestyle and monitoring regularly.")
    else:
        st.error("🔴 You are at **High Risk**. Please consult a doctor as soon as possible.")

    # Display prediction probabilities
    st.subheader("Prediction Probability:")
    st.write(f"Probability of No Heart Disease: {1 - heart_disease_prob:.2f}")
    st.write(f"Probability of Heart Disease: {heart_disease_prob:.2f}")
