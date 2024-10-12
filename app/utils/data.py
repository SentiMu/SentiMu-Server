from enum import Enum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

class ReviewStatus(Enum):
    ORGANIC = "Organic"
    SUSPICIOUS = "Suspicious"
    SPAM = "Spam"

def determine_review_status(df):
    df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'])
    
    earliest_date = df['publishedAtDate'].min()
    latest_date = df['publishedAtDate'].max()
    time_span = latest_date - earliest_date
    reviews_per_year = len(df) / (time_span.days / 365.25)
    
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
        if avg_time_between_reviews < 3600:
            return ReviewStatus.SPAM
        else:
            return ReviewStatus.SUSPICIOUS
    else:
        return ReviewStatus.SUSPICIOUS
    
def get_word_cloud(df, targeted_word=None):
    if targeted_word:
        targeted_word = targeted_word.lower()
        
        df_filtered = df['Text'].dropna()
        df_filtered = df_filtered[df_filtered.str.lower().str.contains(targeted_word, na=False)]
    else:
        df_filtered = df['Text'].dropna() 

    if df_filtered.empty:
        return None
    else:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_filtered)

        word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

        return dict(list(sorted_word_freq.items())[:10])

def generate_month_categories() -> list[str]:
    current_month = datetime.now().month
    categories = []

    for i in range(1, 12):
        month = (current_month + i - 1) % 12 + 1 
        month_name = datetime(2000, month, 1).strftime("%b")
        categories.append(month_name)

    current_month_name = datetime(2000, current_month, 1).strftime("%b")
    categories.append(current_month_name)

    return categories

def generate_month_numbers() -> list[int]:
    current_month = datetime.now().month
    months = []

    for i in range(1, 12):
        month = (current_month + i - 1) % 12 + 1 
        months.append(month)

    months.append(current_month)
    return months 