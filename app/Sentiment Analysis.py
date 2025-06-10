import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st

# Load dataset from CSV
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # If 'Hashtags' are stored as string of hashtags, convert to list
    if 'Hashtags' in df.columns:
        df['Hashtags'] = df['Hashtags'].fillna("").apply(lambda x: re.findall(r"#\w+", x))
    else:
        df['Hashtags'] = [[] for _ in range(len(df))]
    return df

# Clean tweet text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s#@]", "", text)
    return text.lower().strip()

# Sentiment analysis with VADER
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Prepare data
def prepare_data(df):
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    df['Sentiment'] = df['Cleaned_Text'].apply(vader_sentiment)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

# Plotting functions (same as before)
def plot_sentiment_distribution(df):
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Sentiment', order=['Positive', 'Neutral', 'Negative'], palette='viridis')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_top_hashtags(df, top_n=10):
    hashtags_series = df['Hashtags'].explode().dropna().str.lower()
    hashtag_counts = Counter(hashtags_series).most_common(top_n)
    if len(hashtag_counts) == 0:
        st.write("No hashtags found in the dataset.")
        return
    tags, counts = zip(*hashtag_counts)
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(tags), y=list(counts), palette='cubehelix')
    plt.title(f"Top {top_n} Hashtags")
    plt.ylabel("Frequency")
    plt.xlabel("Hashtags")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_sentiment_over_time(df):
    sentiment_time = df.groupby(['Date', 'Sentiment']).size().unstack().fillna(0)
    sentiment_time.plot(kind='line', figsize=(10,6), marker='o')
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_hashtag_trends(df, top_n=5):
    hashtags_series = df['Hashtags'].explode().dropna().str.lower()
    top_hashtags = [tag for tag, _ in Counter(hashtags_series).most_common(top_n)]
    if not top_hashtags:
        st.write("No hashtags found for trend analysis.")
        return

    hashtag_trends = {}
    for tag in top_hashtags:
        df[tag] = df['Hashtags'].apply(lambda x: tag in [h.lower() for h in x] if isinstance(x, list) else False)
        counts = df[df[tag] == True].groupby('Date').size()
        hashtag_trends[tag] = counts

    hashtag_trends_df = pd.DataFrame(hashtag_trends).fillna(0)
    hashtag_trends_df.plot(figsize=(10,6), marker='o')
    plt.title("Top Hashtag Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

# Streamlit dashboard
def main():
    st.title("Media Trend Analysis Dashboard - CSV Version")
    st.write("Analyzing sentiment and trending topics from provided CSV data.")

    uploaded_file = st.file_uploader("C:/Users/Vaibhav/Desktop/sentimentdataset.csv", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = prepare_data(df)

        st.subheader("Sample Data")
        st.dataframe(df[['Date', 'Text', 'Sentiment']].head(10))

        st.subheader("Sentiment Distribution")
        plot_sentiment_distribution(df)

        st.subheader("Top Hashtags")
        plot_top_hashtags(df)

        st.subheader("Sentiment Trends Over Time")
        plot_sentiment_over_time(df)

        st.subheader("Hashtag Trends Over Time")
        plot_hashtag_trends(df)

if __name__ == "__main__":
    main()
