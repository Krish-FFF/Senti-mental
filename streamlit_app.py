# requirements.txt
# streamlit
# pandas
# tweepy
# nltk
# transformers
# torch
# scipy

import streamlit as st
import pandas as pd
import tweepy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Download required NLTK resources
nltk.download('vader_lexicon')

# Set up Twitter API client
def twitter_auth(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

# Fetch recent tweets
def fetch_tweets(api, username, count=5):
    tweets = api.user_timeline(screen_name=username, count=count, tweet_mode='extended')
    return [tweet.full_text for tweet in tweets]

# Sentiment analysis using NLTK VADER
def analyze_vader(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Sentiment analysis using Hugging Face RoBERTa
def analyze_roberta(text, tokenizer, model):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits.detach().numpy()[0]
    scores = softmax(scores)
    sentiment = scores.argmax()  # 0=negative, 1=neutral, 2=positive
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    return sentiment_labels[sentiment], scores[sentiment]

# Streamlit UI setup
st.title("🐦 Twitter Sentiment Analyzer")

# Sidebar for API credentials
st.sidebar.header("Twitter API Credentials")
consumer_key = st.sidebar.text_input("Consumer Key")
consumer_secret = st.sidebar.text_input("Consumer Secret")
access_token = st.sidebar.text_input("Access Token")
access_token_secret = st.sidebar.text_input("Access Token Secret")

# Main UI for username input
username = st.text_input("Enter Twitter Username (without @):")
fetch_button = st.button("Analyze Sentiment")

# Load RoBERTa model once
@st.cache_resource
def load_roberta():
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load_roberta()

# When button clicked
if fetch_button:
    if not all([consumer_key, consumer_secret, access_token, access_token_secret, username]):
        st.error("Please provide all API credentials and username.")
    else:
        try:
            api = twitter_auth(consumer_key, consumer_secret, access_token, access_token_secret)
            tweets = fetch_tweets(api, username)

            results = []
            for tweet in tweets:
                vader_score = analyze_vader(tweet)
                roberta_sentiment, roberta_confidence = analyze_roberta(tweet, tokenizer, model)

                results.append({
                    'Tweet': tweet,
                    'VADER Sentiment Score': vader_score,
                    'RoBERTa Sentiment': roberta_sentiment,
                    'RoBERTa Confidence': roberta_confidence
                })

            df = pd.DataFrame(results)
            st.write("### Analysis Results")
            st.dataframe(df)

        except Exception as e:
            st.error(f"An error occurred: {e}")
