import pandas as pd
from collections import Counter

class ReviewFeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()
        # Convert published date to datetime
        self.df['publishedAtDate'] = pd.to_datetime(self.df['publishedAtDate'])
        
    def create_temporal_features(self):
        """Create time-based features"""
        self.df['year'] = self.df['publishedAtDate'].dt.year
        self.df['month'] = self.df['publishedAtDate'].dt.month
        self.df['day_of_week'] = self.df['publishedAtDate'].dt.day_name()
        self.df['hour'] = self.df['publishedAtDate'].dt.hour
        self.df['quarter'] = self.df['publishedAtDate'].dt.quarter
        
        # Create time-based aggregations
        temporal_features = {
            'reviews_per_month': self.df.groupby(['year', 'month']).size().to_dict(),
            'reviews_per_quarter': self.df.groupby(['year', 'quarter']).size().to_dict(),
            'reviews_by_day': self.df['day_of_week'].value_counts().to_dict(),
            'reviews_by_hour': self.df['hour'].value_counts().to_dict()
        }
        return temporal_features

    def create_sentiment_features(self):
        """Create sentiment-based features"""
        sentiment_features = {
            'sentiment_distribution': self.df['sentiment'].value_counts().to_dict(),
            'avg_confidence': self.df['confidence_score'].mean(),
            'sentiment_by_quarter': self.df.groupby(['year', 'quarter'])['sentiment'].value_counts().to_dict(),
            'high_confidence_reviews': self.df[self.df['confidence_score'] > 0.8].shape[0],
            'low_confidence_reviews': self.df[self.df['confidence_score'] < 0.4].shape[0]
        }
        return sentiment_features

    def create_text_features(self):
        """Create text-based features"""
        # Word count
        self.df['word_count'] = self.df['cleanedText'].apply(lambda x: len(str(x).split()))
        
        # Get common words (simple implementation)
        all_words = ' '.join(self.df['cleanedText'].astype(str)).split()
        word_freq = Counter(all_words).most_common(20)
        
        text_features = {
            'avg_word_count': self.df['word_count'].mean(),
            'common_words': dict(word_freq),
            'reviews_by_language': self.df['detected_language'].value_counts().to_dict()
        }
        return text_features

    def create_engagement_features(self):
        """Create engagement-based features"""
        engagement_features = {
            'avg_likes': self.df['likesCount'].mean(),
            'total_likes': self.df['likesCount'].sum(),
            'high_engagement_reviews': self.df[self.df['likesCount'] > self.df['likesCount'].mean()].shape[0]
        }
        return engagement_features