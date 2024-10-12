from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from utils.data import determine_review_status
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import redis
import json
from functools import wraps

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
    'total_score': 600,         # 10 minutes
    'reviews_count': 600,       # 10 minutes
    'latest_reviews': 600,      # 10 minutes
    'duplicate_reviews': 600,   # 10 minutes
    'word_cloud': 600,          # 10 minutes
    'overview': 600,            # 10 minutes
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
df = pd.read_csv('cleaned_dataset-secret.csv')
label_id = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

# Pydantic models remain the same as in your original code
class Review(BaseModel):
    id: Optional[str]
    name: str
    published_at: str
    text: Optional[str]
    rating: Optional[float]
    image_url: Optional[str]

class ReviewsResponse(BaseModel):
    reviews: List[Review]

class ScoreResponse(BaseModel):
    total_score: float

class CountResponse(BaseModel):
    reviews_count: int

class DuplicateReview(BaseModel):
    name: str
    review_count: int
    latest_review_date: Optional[str]
    status: Optional[str]
    image_url: Optional[str]
    reviews: List[Review]

class DuplicateReviewsResponse(BaseModel):
    duplicate_reviewers: List[DuplicateReview]

class WordCloudData(BaseModel):
    x: str
    y: int
    color: Optional[str]

class WordCloudResponse(BaseModel):
    word_cloud: List[WordCloudData]

class OverviewData(BaseModel):
    positive: float
    negative: float
    neutral: float

class OverviewResponse(BaseModel):
    status: OverviewData

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

# Modified API endpoints with Redis caching
@app.get("/total-score", response_model=ScoreResponse)
@redis_cache("total_score", "total_score")
async def get_total_score():
    total_score = float(df['totalScore'].head(1).values[0])
    return {"total_score": total_score}

@app.get("/reviews-count", response_model=CountResponse)
@redis_cache("reviews_count", "reviews_count")
async def get_reviews_count():
    count = int(df['reviewsCount'].head(1).values[0])
    return {"reviews_count": count}

@app.get("/latest-reviews/{n}", response_model=ReviewsResponse)
@redis_cache("latest_reviews", "latest_reviews")
async def get_latest_reviews(n: int = 5):
    if n < 1:
        raise HTTPException(status_code=400, detail="Number of reviews must be at least 1")
    
    latest_reviews = df.sort_values(by='publishedAtDate', ascending=False).head(n)
    
    reviews_list = [
        Review(
            id=row["publishedAtDate"],
            name=row["name"],
            published_at=row["publishedAtDate"],
            text=row["originalText"] if pd.notna(row["originalText"]) else None,
            rating=row["stars"],
            image_url=row["reviewerPhotoUrl"]
        ).model_dump()
        for _, row in latest_reviews.iterrows()
    ]
    
    return {"reviews": reviews_list}

@app.get("/duplicate-reviews", response_model=DuplicateReviewsResponse)
@redis_cache("duplicate_reviews", "duplicate_reviews")
async def get_duplicated_reviews():
    duplicate_reviewers = df['name'].value_counts()
    duplicate_reviewers = duplicate_reviewers[duplicate_reviewers > 1]
    
    result = []
    
    for name, count in duplicate_reviewers.items():
        user_reviews = df[df['name'] == name].sort_values(by='publishedAtDate', ascending=False)
        latest_review_date = user_reviews['publishedAtDate'].iloc[0]
        
        reviews_list = [
            Review(
                id=str(row["publishedAtDate"]),
                name=row["name"],
                published_at=str(row["publishedAtDate"]),
                text=row["originalText"] if pd.notna(row["originalText"]) else None,
                rating=row["stars"],
                image_url=row["reviewerPhotoUrl"]
            ).model_dump()
            for _, row in user_reviews.iterrows()
        ]
        
        status = determine_review_status(user_reviews)
        
        result.append({
            "name": name,
            "review_count": count,
            "latest_review_date": str(latest_review_date),
            "image_url": user_reviews['reviewerPhotoUrl'].iloc[0],
            "status": status.value,
            "reviews": sorted(reviews_list, key=lambda x: x['published_at'], reverse=True)
        })
    
    sorted_result = sorted(
        result,
        key=lambda x: (
            x["review_count"],
            datetime.strptime(x["latest_review_date"], "%Y-%m-%dT%H:%M:%S.%fZ")
        ),
        reverse=True
    )
    
    return {"duplicate_reviewers": sorted_result[:5]}

@app.get("/word-cloud/", response_model=WordCloudResponse)
@redis_cache("word_cloud", "word_cloud")
async def get_word_cloud_data(targeted_word: Optional[str] = None):
    filtered_texts = df['Text'].dropna()
    
    if targeted_word:
        filtered_texts = filtered_texts[filtered_texts.str.lower().str.contains(targeted_word.lower(), na=False)]
        
    if filtered_texts.empty:
        return {"word_cloud": []}
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(filtered_texts)
    
    word_counts = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    pretrained_id = "mdhugol/indonesia-bert-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_id)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_id)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def find_example_sentences(word, texts, max_examples=5):
        examples = []
        for text in texts:
            if word.lower() in text.lower():
                examples.append(text)
                if len(examples) >= max_examples:
                    break
        return examples

    word_sentiments = {}
    for word, _ in sorted_word_counts:
        example_sentences = find_example_sentences(word, filtered_texts)
        
        if example_sentences:
            sentiments = [label_id[sentiment_analysis(sentence)[0]['label']] for sentence in example_sentences]
            
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

    word_cloud_data = [
        {
            "x": word,
            "y": count,
            "color": word_sentiments[word]
        }
        for word, count in sorted_word_counts
    ]
    
    return {"word_cloud": word_cloud_data}

@app.get("/overview", response_model=OverviewResponse)
@redis_cache("overview", "overview")
async def get_overview():
    # Get normalized value counts once
    status_counts = df['status'].value_counts(normalize=True)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)