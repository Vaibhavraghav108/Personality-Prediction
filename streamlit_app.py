import streamlit as st
import joblib

# Load model, vectorizer, and label encoder
model = joblib.load("model/personality_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")  # NEW

# Streamlit UI
st.title("🧠 Personality Predictor from Social Media Text")

user_input = st.text_area("✍️ Enter your social media post or comment:")

if st.button("Predict Personality"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        predicted_label = label_encoder.inverse_transform(prediction)  # NEW
        st.success(f"🎯 Predicted Personality Type: **{predicted_label[0]}**")
