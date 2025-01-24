from pydantic import BaseModel
from typing import List, Optional

# Pydantic models that match the data structure needed by frontend
class TargetedWordRequest(BaseModel):
    word: str
    
class ScoreResponse(BaseModel):
    total_score: float

class CountResponse(BaseModel):
    reviews_count: int

class OverviewData(BaseModel):
    positive: float
    negative: float
    neutral: float

class OverviewResponse(BaseModel):
    status: OverviewData

class TimeSeriesData(BaseModel):
    name: str
    data: List[int]

class TimeSeriesResponse(BaseModel):
    time_series: List[TimeSeriesData]

class Review(BaseModel):
    id: Optional[str]
    name: str
    published_at: str
    text: Optional[str]
    rating: Optional[float]
    image_url: Optional[str]

class ReviewsResponse(BaseModel):
    reviews: List[Review]

class WordCloudData(BaseModel):
    x: str
    y: int
    color: Optional[str]

class WordCloudResponse(BaseModel):
    word_cloud: List[WordCloudData]

class DuplicateReviewData(BaseModel):
    name: str
    review_count: int
    latest_review_date: Optional[str]
    status: Optional[str]
    reasons: Optional[List[str]]
    image_url: Optional[str]
    reviews: List[Review]

class DuplicateReviewsResponse(BaseModel):
    duplicate_reviewers: List[DuplicateReviewData]