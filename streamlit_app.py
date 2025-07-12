# streamlit_app.py
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/personality_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Streamlit UI
st.title("🧠 Personality Predictor from Social Media Text")

user_input = st.text_area("✍️ Enter your social media post or comment:")

if st.button("Predict Personality"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        st.success(f"🎯 Predicted Personality Type: **{prediction[0]}**")
