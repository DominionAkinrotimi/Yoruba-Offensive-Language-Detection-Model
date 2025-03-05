from flask import Flask, request, jsonify, render_template
import re
import emoji
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data['text']

    # Clean the input text
    cleaned_text = clean_tweet(user_input)

    # Transform the input text using the vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])

    # Make prediction
    prediction = model.predict(text_vectorized)

    # Return the result
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)