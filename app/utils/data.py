from enum import Enum
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

def determine_user_review_status(df):
    """
    Analyzes reviews from a single user to detect spam patterns.
    
    Parameters:
    df: DataFrame with columns:
        - publishedAtDate: datetime string
        - stars: numeric
        - text/cleanedText: string
        - sentiment: string
        - confidence_score: float
    
    Returns:
    Dictionary containing analysis results and ReviewStatus
    """
    # Convert dates to datetime if they're strings
    df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'])
    
    # Sort reviews by time
    df = df.sort_values('publishedAtDate')
    
    # Calculate key metrics
    review_count = len(df)
    time_span = (df['publishedAtDate'].max() - df['publishedAtDate'].min()).total_seconds()
    
    # Initialize results
    spam_signals = 0
    reasons = []
    
    # 1. Time-based Analysis
    
    # Daily analysis
    daily_counts = df.groupby(df['publishedAtDate'].dt.date).size()
    max_daily_reviews = daily_counts.max()
    
    if max_daily_reviews > 4:
        spam_signals += 3
        reasons.append(f"Posted {max_daily_reviews} reviews in one day (>4 threshold)")
    elif max_daily_reviews > 2:
        spam_signals += 1
        reasons.append(f"Posted {max_daily_reviews} reviews in one day (>2 threshold)")
    
    # Yearly analysis
    yearly_counts = df.groupby(df['publishedAtDate'].dt.year).size()
    for year, count in yearly_counts.items():
        if count > 12:
            spam_signals += 2
            reasons.append(f"Posted {count} reviews in {year} (>12 threshold)")
    
    # Short time window analysis
    time_diffs = df['publishedAtDate'].diff().dropna()
    very_quick_reviews = time_diffs <= pd.Timedelta(minutes=5)
    if very_quick_reviews.any():
        quick_review_count = very_quick_reviews.sum()
        if quick_review_count >= 2:
            spam_signals += 2
            reasons.append(f"Posted {quick_review_count} reviews within 5 minutes of each other")
    
    # Text similarity
    text_column = 'cleanedText' if 'cleanedText' in df.columns else 'text'
    unique_texts = df[text_column].nunique()
    text_similarity_ratio = unique_texts / review_count
    
    if text_similarity_ratio < 0.8:
        spam_signals += 1
        reasons.append("High similarity between review texts")
    
    # 3. Additional Pattern Detection
    
    # Check for periodic posting (e.g., exact same time every day)
    time_of_day = df['publishedAtDate'].dt.time
    unique_times = time_of_day.nunique()
    if unique_times == 1 and review_count >= 3:
        spam_signals += 2
        reasons.append("Every reviews posted at exactly same time")
    
    # Check for burst patterns (many reviews in short bursts with long gaps)
    if len(time_diffs) >= 3:
        # Calculate both the median and standard deviation for better insight
        median_time_diff = time_diffs.median().total_seconds()
        std_time_diff = time_diffs.std().total_seconds()
        
        # Define thresholds to identify burst behavior
        burst_std_threshold = 86400  # 1 day in seconds
        burst_median_threshold = 43200  # Half a day in seconds

        # Check if the user exhibits burst-cooldown patterns
        if (std_time_diff > burst_std_threshold or median_time_diff < burst_median_threshold) and max_daily_reviews > 3:
            spam_signals += 1
            reasons.append(
                "Burst pattern detected - high variation in review timing with many reviews in short periods."
            )

    
    # Determine final status
    result = {
        'review_count': review_count,
        'time_span_seconds': time_span,
        'max_daily_reviews': max_daily_reviews,
        'yearly_reviews': yearly_counts.to_dict(),
        'text_similarity_ratio': text_similarity_ratio,
        'spam_signals': spam_signals,
        'reasons': reasons
    }

    if spam_signals >= 4:
        result['status'] = "Spam"
    elif spam_signals >= 1:
        result['status'] = "Suspicious"
    else:
        result['status'] = "Organic"
        reasons.append("No spam signal detected")
    
    return result


# def determine_review_status(df):
#     df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'])
    
#     earliest_date = df['publishedAtDate'].min()
#     latest_date = df['publishedAtDate'].max()
#     time_span = latest_date - earliest_date
#     reviews_per_year = len(df) / (time_span.days / 365.25) if time_span.days > 0 else float('inf')
    
#     if len(df) > 1:
#         sorted_dates = sorted(df['publishedAtDate'])
#         time_diffs = [(sorted_dates[i+1] - sorted_dates[i]).total_seconds() 
#                         for i in range(len(sorted_dates)-1)]
#         avg_time_between_reviews = sum(time_diffs) / len(time_diffs) if time_diffs else float('inf')
#     else:
#         avg_time_between_reviews = float('inf')
    
#     if reviews_per_year <= 10:
#         return ReviewStatus.ORGANIC
#     elif reviews_per_year > 12:
#         if avg_time_between_reviews < 3600:  # Less than an hour between reviews
#             return ReviewStatus.SPAM
#         else:
#             return ReviewStatus.SUSPICIOUS
#     else:
#         return ReviewStatus.SUSPICIOUS
    
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