# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to fetch data from social media APIs or user's social handles
def fetch_social_media_data(platform, username=None, api_keys=None, num_posts=500):
    """Fetches data from social media APIs or user's social handles.

    Args:
        platform (str): The social media platform (e.g., 'twitter', 'facebook').
        username (str, optional): The username of the social media account. Required if
            using user's handle. Defaults to None.
        api_keys (dict, optional): A dictionary containing API keys for the platform.
            Required if using APIs. Defaults to None.
        num_posts (int, optional): The number of recent posts to fetch. Defaults to 500.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched posts.
    """

    if platform == 'twitter':
        # Import necessary libraries for Twitter API
        import tweepy

        # Authenticate with Twitter API using API keys or user login
        if api_keys:
            auth = tweepy.OAuthHandler(api_keys['consumer_key'], api_keys['consumer_secret'])
            auth.set_access_token(api_keys['access_token'], api_keys['access_token_secret'])
            api = tweepy.API(auth)
        else:
            # Prompt user for login credentials
            consumer_key = input("Enter your Twitter consumer key: ")
            consumer_secret = input("Enter your Twitter consumer secret: ")
            access_token = input("Enter your Twitter access token: ")
            access_token_secret = input("Enter your Twitter access token secret: ")

            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth)

        # Fetch recent tweets
        tweets = []
        for tweet in tweepy.Cursor(api.user_timeline, screen_name=username, tweet_mode='extended').items(num_posts):
            tweets.append(tweet.full_text)

        # Create a DataFrame from tweets
        df = pd.DataFrame({'Text': tweets})

    elif platform == 'facebook':
        # Import necessary libraries for Facebook API
        import facebook

        # Authenticate with Facebook API using API keys or user login
        if api_keys:
            graph = facebook.GraphAPI(access_token=api_keys['access_token'])
        else:
            # Prompt user for login credentials
            access_token = input("Enter your Facebook access token: ")
            graph = facebook.GraphAPI(access_token=access_token)

        # Fetch recent posts
        posts = []
        results = graph.get_connections(id='me', connection_name='posts')
        for post in results['data']:
            posts.append(post['message'])

        # Create a DataFrame from posts
        df = pd.DataFrame({'Text': posts})

    else:
        raise ValueError("Unsupported platform. Please choose 'twitter' or 'facebook'.")

    return df

# Function to analyze sentiment
def analyze_sentiment(data_source, file_path=None, platform=None, username=None,
                      use_api_keys=False, api_keys=None):
    """Analyzes sentiment of text data from file or social media."""

    # Input validation
    if data_source == "file" and file_path is None:
        raise ValueError("Please provide a file path.")
    if data_source == "social_media" and (username is None or (use_api_keys and any(v is None for v in api_keys.values()))):
        raise ValueError("Please provide username and API keys (if applicable).")

    # Load or fetch data based on data_source
    if data_source == "file":
        df = pd.read_csv(file_path)
    elif data_source == "social_media":
        df = fetch_social_media_data(platform, username, api_keys if use_api_keys else None)
    else:
        raise ValueError("Invalid data source.")

    # Data Preprocessing
    df = df.head(500)  # Limit data for demonstration purposes
    # Assume 'Score' column exists if using a file with pre-labeled data
    if 'Score' in df.columns:
        df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)  # 1 for positive, 0 for negative
    else:
        # If 'Score' column doesn't exist, assume data is unlabeled and set sentiment to None
        df['sentiment'] = None

    # Split data into train and test sets (if sentiment labels are available)
    if df['sentiment'].notna().any():
        X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['sentiment'], test_size=0.2, random_state=42)
        oversampler = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
        X_train_resampled = X_train_resampled.flatten()
    else:
        # If sentiment labels are not available, use the entire dataset for prediction
        X_test = df['Text']

    # VADER Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    vader_results = X_test.apply(lambda text: sia.polarity_scores(text)['compound'])

    # RoBERTa Sentiment analysis
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores[2]  # Return positive score

    roberta_results = X_test.apply(polarity_scores_roberta)

    # Ensemble and Evaluation
    ensemble_scores = (vader_results + roberta_results) / 2
    ensemble_predictions = ensemble_scores.apply(lambda x: 1 if x >= 0.5 else 0)

    # Calculate accuracy and classification report if sentiment labels are available
    if df['sentiment'].notna().any():
        accuracy = accuracy_score(y_test, ensemble_predictions)
        classification_rep = classification_report(y_test, ensemble_predictions)
    else:
        accuracy = None
        classification_rep = None

    results_df = pd.DataFrame({'Text': X_test, 'Sentiment': ensemble_predictions})

    # Return sentiment predictions and accuracy/classification report
    return results_df, accuracy, classification_rep


# Create Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=[
        gr.Radio(["file", "social_media"], label="Data Source"),
        gr.File(label="File Path", visible=True),
        gr.Dropdown(["twitter", "facebook"], label="Platform", visible=False),
        gr.Textbox(label="Username", visible=False),
        gr.Checkbox(label="Use API Keys", visible=False),
        gr.Textbox(label="Consumer Key", visible=False),
        gr.Textbox(label="Consumer Secret", visible=False),
        gr.Textbox(label="Access Token", visible=False),
        gr.Textbox(label="Access Token Secret", visible=False)
    ],
    outputs=[
        gr.Dataframe(label="Sentiment Results"),
        gr.Number(label="Accuracy"),
        gr.Textbox(label="Classification Report")
    ],
    title="Sentiment Analysis Tool",
    description="Analyze sentiment of text data from file or social media."
)

# Launch the interface
iface.launch()
