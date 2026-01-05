import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Movie Interest Predictor", page_icon="🎬")
st.title("🎬 Movie Interest Predictor (Decision Tree)")

# Load Model and Encoder
model_path = 'movie_model.pkl'
encoder_path = 'interest_encoder.pkl'

if os.path.exists(model_path) and os.path.exists(encoder_path):
    model = joblib.load(model_path)
    le_interest = joblib.load(encoder_path)

    st.write("Enter the details to predict what kind of movies a person might like.")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Female", "Male"])
    
    # Map gender to numeric (0 for Female, 1 for Male)
    gender_num = 1 if gender == "Male" else 0

    if st.button("Predict Interest"):
        # Features: [Age, Gender]
        features = np.array([[age, gender_num]])
        prediction_id = model.predict(features)
        
        # Convert ID back to name (e.g., 'Action')
        result = le_interest.inverse_transform(prediction_id)
        
        st.success(f"### Predicted Interest: **{result[0]}**")
else:
    st.warning("Please upload 'movie_model.pkl' and 'interest_encoder.pkl' to your GitHub repo.")