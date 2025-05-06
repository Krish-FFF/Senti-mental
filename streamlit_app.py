# requirements.txt
# streamlit
# pandas
# tweepy
# nltk
# transformers
# torch
# scipy
# matplotlib

import streamlit as st
import pandas as pd
import tweepy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt

# Download required NLTK resources
nltk.download('vader_lexicon')

# Twitter API v2 client
def twitter_auth_v2(bearer_token):
    return tweepy.Client(bearer_token=bearer_token)

# Fetch tweets using Twitter API v2
def fetch_tweets_v2(client, username, count=5):
    user = client.get_user(username=username)
    user_id = user.data.id

    tweets = client.get_users_tweets(
        id=user_id, max_results=count, tweet_fields=['text']
    )

    return [tweet.text for tweet in tweets.data]

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

# Sidebar for Twitter API v2
st.sidebar.header("Twitter API v2 Credentials")
bearer_token = st.sidebar.text_input("Bearer Token", type="password")

# Main UI
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
    if not all([bearer_token, username]):
        st.error("Please provide Bearer token and username.")
    else:
        try:
            client = twitter_auth_v2(bearer_token)
            tweets = fetch_tweets_v2(client, username)

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

            # Visualization
            st.write("### Sentiment Distribution")
            sentiment_counts = df['RoBERTa Sentiment'].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'gray', 'red'])
            plt.xlabel('Sentiment')
            plt.ylabel('Number of Tweets')
            plt.title('Sentiment Distribution of Last 5 Tweets')
            plt.xticks(rotation=0)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
