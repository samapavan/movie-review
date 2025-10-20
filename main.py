import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
word_index = imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
model=load_model('movie_review_model.h5')
def decode_review(text):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    return decoded_review
def preprocess_review(review):
    max_features = 10000
    max_len = 500
    encoded_review = [1]  # Start with the start token
    for word in review.split():
        for key, value in imdb.get_word_index().items():
            if value == word:
                encoded_review.append(key + 3)  # Offset by 3
                break
        else:
            encoded_review.append(2)  # Unknown word token
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review
def predect_review_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]
import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")
user_input = st.text_area("Movie Review", "")
if st.button("Classify"):
    preprocess_input=preprocess_review(user_input)
    predection=model.predict(preprocess_input)
    sentiment = 'Positive' if predection[0][0] >= 0.5 else 'Negative'
    st.write(f"Predicted Sentiment: **{sentiment}**")
    st.write(f"Confidence Score: **{predection[0][0]}**")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")
    