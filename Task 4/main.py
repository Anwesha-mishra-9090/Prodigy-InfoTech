import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Example: Load sample social media data with columns ['date', 'text']
# Replace with your own data source or dataset CSV
# Sample dataframe creation for demo purposes
data = {
    'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'text': [
        'I love the new product! Absolutely fantastic and user-friendly.' if i % 5 != 0 else
        'This product is terrible. Very disappointed and will not buy again.'
        for i in range(100)
    ]
}
df = pd.DataFrame(data)

# Preprocessing & Cleaning Function
def preprocess_text(text):
    # Simple preprocessing: lowercasing and removing excess whitespace
    text = text.lower()
    return text.strip()

df['clean_text'] = df['text'].apply(preprocess_text)

# Apply VADER sentiment analysis
def classify_sentiment(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_text'].apply(classify_sentiment)
df['compound_score'] = df['clean_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Visualization 1: Sentiment distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='sentiment', order=['Positive', 'Neutral', 'Negative'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.show()

# Visualization 2: Sentiment over time (rolling average)
df.set_index('date', inplace=True)
sentiment_daily = df.resample('D').mean(numeric_only=True).compound_score
sentiment_rolling = sentiment_daily.rolling(window=7).mean()

plt.figure(figsize=(12,6))
plt.plot(sentiment_daily, label='Daily Average Compound Score', alpha=0.6)
plt.plot(sentiment_rolling, label='7-Day Rolling Average', linewidth=2)
plt.title("Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average Compound Sentiment Score")
plt.legend()
plt.show()

# Visualization 3: Word clouds for positive and negative sentiments
positive_text = ' '.join(df[df['sentiment'] == 'Positive']['clean_text'])
negative_text = ' '.join(df[df['sentiment'] == 'Negative']['clean_text'])

stopwords = set(STOPWORDS)

def plot_wordcloud(text, title, color):
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color='white',
                          max_words=100,
                          max_font_size=50,
                          scale=3,
                          colormap=color,
                          random_state=42).generate(text)
    plt.figure(figsize=(8,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

plot_wordcloud(positive_text, "Positive Sentiment Word Cloud", 'Greens')
plot_wordcloud(negative_text, "Negative Sentiment Word Cloud", 'Reds')

# Optional: Save the enriched dataframe with sentiments to CSV
df.reset_index().to_csv("social_media_sentiment_analysis.csv", index=False)

print("Sentiment analysis and visualization complete.")
