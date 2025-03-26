from flask import Flask, render_template, request
import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('lstm_sentiment_model.h5')  # Replace with your model path

# Initialize preprocessing tools
# nltk.download('stopwords')
# ps = PorterStemmer()
vocab_size = 5000
maxlen = 200  # Should match your model's input length

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^/]+/>', '', text)
    
    # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # # Stemming
    # text = ' '.join([ps.stem(word) for word in text.split()])
    
    return text

def encode_text(text):
    # One-hot encoding and padding
    processed_text = preprocess_text(text)
    encoded = [one_hot(processed_text, vocab_size)]
    padded = pad_sequences(encoded, maxlen=maxlen, padding='post')
    return padded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        if not text:
            return render_template('index.html', error="Please enter some text")
        
        # Preprocess and encode
        processed_text = preprocess_text(text)
        encoded_text = encode_text(processed_text)
        
        # Make prediction
        prediction = model.predict(encoded_text)
        sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
        confidence = round(float(prediction[0][0]) * 100, 2) if sentiment == 'positive' else round((1 - float(prediction[0][0])) * 100, 2)
        
        return render_template('index.html', 
                             text=text,
                             sentiment=sentiment,
                             confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)