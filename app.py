import streamlit as st
import re
import emoji
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)

# Load the trained model and vectorizer
model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Data Cleaning Function
def clean_tweet(tweet):
    tweet = tweet.lower()  # Convert to lowercase
    tweet = emoji.replace_emoji(tweet, replace="")  # Remove emojis
    tweet = re.sub(r"http\S+|www\S+|#\S+|@\S+", "", tweet)  # Remove URLs, hashtags, and mentions
    tweet = re.sub(r'<.*?>', '', tweet)
    tweet = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ\s]", "", tweet)  # Keep Yoruba characters
    tweet = re.sub(r"\d+", "", tweet)  # Remove digits
    tweet = re.sub(r"\s+", " ", tweet).strip()  # Remove extra spaces
    return tweet

# Streamlit App
st.title("Language Detection")
st.write("Enter a sentence in Yoruba to check if it contains offensive language:")

# Text input
user_input = st.text_area("Enter a Yoruba text/tweet:", "")


if st.button("Predict"):
    if user_input:
        # Clean the input text
        cleaned_text = clean_tweet(user_input)

        # Transform the input text using the vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = model.predict(text_vectorized)

        # Display the result
        if prediction[0] == 'Offensive':
            st.error("The text contains offensive language.")
        elif prediction[0] == 'Hate':
            st.error("The text contains hate speech.")
        else:
            st.success("The text does not contain offensive language.")
    else:
        st.warning("Please enter some text.")