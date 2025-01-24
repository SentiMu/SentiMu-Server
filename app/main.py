from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
from dateutil import parser
from utils.data import determine_user_review_status
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import redis
import json
from functools import wraps
from models import (
    ScoreResponse,
    CountResponse,
    OverviewResponse,
    TimeSeriesData,
    TimeSeriesResponse,
    Review,
    ReviewsResponse,
    WordCloudResponse,
    DuplicateReviewsResponse,
    TargetedWordRequest,
)

app = FastAPI()

# Add error handling for Redis connection at startup
try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True
    )
    redis_client.ping()  # Test the connection
except redis.ConnectionError:
    print("Failed to connect to Redis. Make sure Redis is running!")
    redis_client = None

# Cache expiration times (in seconds)
CACHE_EXPIRATION = {
    'total_score': 6,         # 10 minutes
    'reviews_count': 6,       # 10 minutes
    'overview': 6,            # 10 minutes
    'time_series': 6,         # 10 minutes
    'latest_reviews': 6,      # 10 minutes
    'word_cloud': 6,          # 10 minutes
    'duplicate_reviews': 6,   # 10 minutes
}

# CORS setup remains the same
prod = [
    "http://localhost:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=prod,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the DataFrame
df = pd.read_csv('dataset-secret.csv')

# Define stopwords 
STOPWORDS = {"the", "yang", "di", "dan", "ada", "and", "is", "in", "of", "to", "untuk", "dari", "dengan", "yg", "ada", "ini", "atau", "lebih", "menjadi",
            "itu", "saya", "kami", "anda", "mereka", "dia", "aku", "kau", "ia", "kita", "anda", "orang", "akan", "telah", "pun", "tapi", "bisa", "gaes",
            "juga", "lagi", "sudah", "masih", "sekarang", "sementara", "ketika", "kalau", "jika", "seolah", "seolah-olah", "seakan", "karena", "klo", "luas",
            "seakan-akan", "bagi", "ke", "kepada", "oleh", "sangat","banyak", "sedikit", "beberapa", "setiap", "semua", "seluruh", "seperti", "buat", "nya",
            "tidak", "tak", "bukan", "jangan", "enggak", "gak", "tidaklah", "takkan", "takboleh", "boleh", "harus", "mesti", "perlu", "ya", "sampai", "lain"}

# Cache decorator
def redis_cache(cache_key: str, expiration_key: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate dynamic cache key if needed
            final_cache_key = cache_key
            if kwargs:
                final_cache_key = f"{cache_key}:{':'.join([f'{k}={v}' for k, v in kwargs.items()])}"
            
            try:
                # Try to get from cache
                cached_data = redis_client.get(final_cache_key)
                if cached_data:
                    return json.loads(cached_data)
                
                # If not in cache, execute function and cache result
                result = await func(*args, **kwargs)
                redis_client.setex(
                    final_cache_key,
                    CACHE_EXPIRATION[expiration_key],
                    json.dumps(result)
                )
                return result
            except redis.RedisError:
                # If Redis fails, just execute the function
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Placeholder for global targeted word
targeted_word: Optional[str] = None

# Modified API endpoints with Redis caching
@app.post("/set-target-word")
async def set_target_word(request: TargetedWordRequest):
    global targeted_word
    targeted_word = request.word
    return {"message": f"Target word '{targeted_word}' set successfully."}

@app.get("/get-target-word")
async def get_target_word():
    global targeted_word
    if targeted_word:
        return {"target_word": targeted_word}
    return {"message": "No target word has been set yet."}

def filter_reviews_by_targeted_word(df, targeted_word):
    if targeted_word:
        return df[df['text'].str.contains(targeted_word, na=False, case=False)]
    return df

@app.get("/total-score", response_model=ScoreResponse)
@redis_cache("total_score", "total_score")
async def get_total_score():
    global targeted_word
    filtered_df = filter_reviews_by_targeted_word(df, targeted_word)

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail=f"No reviews found for the targeted word '{targeted_word}'.")

    total_score = round(filtered_df['stars'].mean(), 2)
    return {"total_score": total_score}

@app.get("/reviews-count", response_model=CountResponse)
@redis_cache("reviews_count", "reviews_count")
async def get_reviews_count():
    global targeted_word
    filtered_df = filter_reviews_by_targeted_word(df, targeted_word)
    
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail=f"No reviews found for the targeted word '{targeted_word}'.")
    
    count = len(filtered_df)
    return {"reviews_count": count}

@app.get("/latest-reviews", response_model=ReviewsResponse)
@redis_cache("latest_reviews", "latest_reviews")
async def get_latest_reviews():
    global targeted_word
    filtered_df = filter_reviews_by_targeted_word(df, targeted_word)
    latest_reviews = filtered_df.sort_values(by='publishedAtDate', ascending=False).head(5)

    reviews_list = [
        Review(
            id=str(row["publishedAtDate"]),
            name=row["name"],
            published_at=str(row["publishedAtDate"]),
            text=row["text"] if pd.notna(row["text"]) else None,
            rating=row["stars"],
            image_url=row["reviewerPhotoUrl"]
        ).model_dump()
        for _, row in latest_reviews.iterrows()
    ]

    return {"reviews": reviews_list}


@app.get("/duplicate-reviews", response_model=DuplicateReviewsResponse)
@redis_cache("duplicate_reviews", "duplicate_reviews")
async def get_duplicated_reviews():
    global targeted_word

    filtered_df = filter_reviews_by_targeted_word(df, targeted_word)

    # find user with duplicate reviews
    duplicate_reviewers = filtered_df['name'].value_counts()
    duplicate_reviewers = duplicate_reviewers[duplicate_reviewers > 1]

    result = []

    for name, count in duplicate_reviewers.items():
        user_reviews = filtered_df[filtered_df['name'] == name].sort_values(by='publishedAtDate', ascending=False)
        latest_review_date = user_reviews['publishedAtDate'].iloc[0]

        reviews_list = [
            Review(
                id=f"{row['name']}_{row['publishedAtDate']}",
                name=row["name"],
                published_at=str(row["publishedAtDate"]),
                text=row["text"] if pd.notna(row["text"]) else None,
                rating=row["stars"],
                image_url=row["reviewerPhotoUrl"]
            ).model_dump()
            for _, row in user_reviews.iterrows()
        ]

        # Determine the status for the user's reviews
        status = determine_user_review_status(user_reviews)

        result.append({
            "name": name,
            "review_count": count,
            "latest_review_date": str(latest_review_date),
            "image_url": user_reviews['reviewerPhotoUrl'].iloc[0],
            "status": status['status'],
            "reasons": status['reasons'],
            "reviews": sorted(reviews_list, key=lambda x: x['published_at'], reverse=True),
        })

    # Sort by review count and latest review date
    sorted_result = sorted(
        result,
        key=lambda x: (
            x.get("review_count", 0),
            parser.isoparse(x.get("latest_review_date", "1970-01-01T00:00:00.000000Z"))
        ),
        reverse=True
    )
    
    return {"duplicate_reviewers": sorted_result[:5]}

@app.get("/word-cloud/", response_model=WordCloudResponse)
@redis_cache("word_cloud", "word_cloud")
async def get_word_cloud_data():
    global targeted_word

    filtered_df = filter_reviews_by_targeted_word(df, targeted_word)
    filtered_texts = filtered_df['cleanedText'].dropna()

    if filtered_texts.empty:
        return {
            "word_cloud": [
                {
                    "x": f"Not Enough Data for {targeted_word}" if targeted_word else "Not Enough Data",
                    "y": 1,
                    "color": "#7D8998",
                }
            ]
        }

    def remove_stopwords(text):
        words = text.split()
        return " ".join([word for word in words if word.lower() not in STOPWORDS])

    filtered_texts = filtered_texts.apply(remove_stopwords)

    # create a word count vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(filtered_texts)

    # calculate word frequencies
    word_counts = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # sentiment analysis setup
    pretrained_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_id)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_id)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # helper function to find example sentences
    def find_example_sentences(word, texts, max_examples=5):
        return [text for text in texts if word.lower() in text.lower()][:max_examples]

    # calculate word sentiments
    word_sentiments = {}
    for word, _ in sorted_word_counts:
        example_sentences = find_example_sentences(word, filtered_texts)
        if example_sentences:
            sentiments = [sentiment_analysis(sentence)[0]['label'] for sentence in example_sentences]
            pos_count = sentiments.count('positive')
            neg_count = sentiments.count('negative')

            if pos_count > neg_count:
                word_sentiments[word] = '#69AE34'
            elif neg_count > pos_count:
                word_sentiments[word] = '#FE6E73'
            else:
                word_sentiments[word] = '#7D8998'
        else:
            word_sentiments[word] = '#7D8998'

    # prepare word cloud data
    word_cloud_data = [
        {"x": word, "y": count, "color": word_sentiments[word]}
        for word, count in sorted_word_counts
    ]

    return {"word_cloud": word_cloud_data}


@app.get("/overview", response_model=OverviewResponse)
@redis_cache("overview", "overview")
async def get_overview():
    global targeted_word

    filtered_df = filter_reviews_by_targeted_word(df, targeted_word)

    status_counts = filtered_df['sentiment'].value_counts(normalize=True)

    positive = round(status_counts.get('positive', 0) * 100, 2)
    negative = round(status_counts.get('negative', 0) * 100, 2)
    neutral = round(status_counts.get('neutral', 0) * 100, 2)

    return {
        "status": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral
        }
    }


@app.get("/time-series", response_model=TimeSeriesResponse)
@redis_cache("time_series", "time_series")
async def get_time_series():
    global targeted_word
    filtered_df = filter_reviews_by_targeted_word(df, targeted_word)

    data = {
        "Positive Review": [0] * 12,
        "Negative Review": [0] * 12,
        "Neutral Review": [0] * 12,
    }

    filtered_df['publishedAtDate'] = pd.to_datetime(filtered_df['publishedAtDate'])

    current_date = pd.Timestamp.now()

    # Loop through the last 12 months
    for i in range(12):
        # Calculate the target year and month
        target_date = current_date - pd.DateOffset(months=i)
        target_year = target_date.year
        target_month = target_date.month

        # Filter the dataframe by target month and year
        month_df = filtered_df[
            (filtered_df['publishedAtDate'].dt.year == target_year) &
            (filtered_df['publishedAtDate'].dt.month == target_month)
        ]

        # Count the sentiments
        status_counts = month_df['sentiment'].value_counts()

        # Update the data dictionary with counts
        data["Positive Review"][11 - i] = status_counts.get('positive', 0)
        data["Negative Review"][11 - i] = status_counts.get('negative', 0)
        data["Neutral Review"][11 - i] = status_counts.get('neutral', 0)

    # Prepare the time series data
    time_series_data = [
        TimeSeriesData(
            name=name,
            data=[int(x) for x in data[name]]
        ).model_dump()
        for name in data.keys()
    ]

    return {"time_series": time_series_data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 