from enum import Enum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class ReviewStatus(Enum):
    ORGANIC = "Organic"
    SUSPICIOUS = "Suspicious"
    SPAM = "Spam"

def determine_review_status(df):
    # Convert publishedAtDate to datetime if it's not already
    df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'])
    
    # Get the earliest and latest review dates
    earliest_date = df['publishedAtDate'].min()
    latest_date = df['publishedAtDate'].max()
    
    # Calculate the time span of reviews
    time_span = latest_date - earliest_date
    
    # Count reviews per year
    reviews_per_year = len(df) / (time_span.days / 365.25)
    
    # Calculate average time between reviews
    if len(df) > 1:
        sorted_dates = sorted(df['publishedAtDate'])
        time_diffs = [(sorted_dates[i+1] - sorted_dates[i]).total_seconds() 
                        for i in range(len(sorted_dates)-1)]
        avg_time_between_reviews = sum(time_diffs) / len(time_diffs)
    else:
        avg_time_between_reviews = float('inf')
    
    if reviews_per_year <= 10:
        return ReviewStatus.ORGANIC
    elif reviews_per_year > 12:
        # If average time between reviews is less than 1 hour, consider it spam
        if avg_time_between_reviews < 3600:
            return ReviewStatus.SPAM
        else:
            return ReviewStatus.SUSPICIOUS
    else:
        return ReviewStatus.SUSPICIOUS
    
def get_word_cloud(df, targeted_word=None):
    if targeted_word:
        targeted_word = targeted_word.lower()
        
        # Drop NaN values first
        df_filtered = df['Text'].dropna()
        
        # Then apply string operations
        df_filtered = df_filtered[df_filtered.str.lower().str.contains(targeted_word, na=False)]
    else:
        df_filtered = df['Text'].dropna()  # No filtering if no targeted word

    if df_filtered.empty:
        return None
    else:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_filtered)

        word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

        # Return top 10 words with their frequencies
        return dict(list(sorted_word_freq.items())[:10])
