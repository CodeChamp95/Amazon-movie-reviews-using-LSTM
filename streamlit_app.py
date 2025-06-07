import streamlit as st
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model = load_model('lstm_sentiment_model.h5')

# Parameters (should match your training setup)
vocab_size = 5000
maxlen = 200

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^/]+/>', '', text)
    return text

# Encode text
def encode_text(text):
    processed_text = preprocess_text(text)
    encoded = [one_hot(processed_text, vocab_size)]
    padded = pad_sequences(encoded, maxlen=maxlen, padding='post')
    return padded

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below and get its sentiment prediction using a trained LSTM model.")

# Input
user_input = st.text_area("Write your movie review here:", height=300)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        encoded = encode_text(user_input)
        prediction = model.predict(encoded)[0][0]
        sentiment = "Positive 😊" if prediction > 0.7 else "Negative 😞"
        confidence = round(float(prediction) * 100, 2) if prediction > 0.7 else round((1 - float(prediction)) * 100, 2)

        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence}%")
