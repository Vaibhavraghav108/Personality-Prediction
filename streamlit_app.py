import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the BERT model and your trained classifier + label encoder
bert = SentenceTransformer('all-MiniLM-L6-v2')
model = joblib.load("model/personality_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Optional: MBTI personality meanings
mbti_meanings = {
    'INFJ': "The Advocate",
    'INFP': "The Mediator",
    'ENFP': "The Campaigner",
    'ENFJ': "The Protagonist",
    'INTJ': "The Architect",
    'INTP': "The Logician",
    'ENTP': "The Debater",
    'ENTJ': "The Commander",
    'ISTJ': "The Logistician",
    'ISFJ': "The Defender",
    'ESTJ': "The Executive",
    'ESFJ': "The Consul",
    'ISTP': "The Virtuoso",
    'ISFP': "The Adventurer",
    'ESTP': "The Entrepreneur",
    'ESFP': "The Entertainer",
}

# Streamlit app UI
st.set_page_config(page_title="Personality Predictor", page_icon="🧠")
st.title("🧠 Predict Your MBTI Personality Type")
st.write("Paste your social media post or any text below and we'll predict your MBTI personality type!")

user_input = st.text_area("✍️ Enter your social media post or comment:")

if st.button("Predict Personality"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            # Get BERT embedding
            input_vec = bert.encode([user_input])
            
            # Ensure it's a 2D array for model input
            if isinstance(input_vec, list):
                input_vec = np.array(input_vec)

            # Make prediction
            prediction = model.predict(input_vec)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Display result
            meaning = mbti_meanings.get(predicted_label.upper(), "Unknown Type")
            st.success(f"🎯 Predicted Personality Type: **{predicted_label}** – *{meaning}*")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
