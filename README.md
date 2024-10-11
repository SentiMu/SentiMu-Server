# SentiMu Analytics Backend

This repository contains the backend server for the SentiMu Analytics Dashboard platform, built using FastAPI and Redis. The server provides RESTful API endpoints for managing and analyzing customer reviews using a pre-trained BERT model for sentiment analysis.

## 🚀 Features

- RESTful API endpoints for reviews management
- Sentiment analysis using BERT model
- Word cloud generation from review content
- Spam review detection
- Redis caching for improved performance
- Automatic OpenAPI documentation

## 🔧 Tech Stack

- FastAPI
- Redis
- Transformers (Hugging Face)
- PyTorch
- Pydantic (data validation)

## 📋 Prerequisites

Please refer to `requirements.txt` for a complete list of dependencies. Key requirements include:

```
fastapi==0.115.0
uvicorn==0.31.1
transformers==4.45.2
torch==2.4.1
pandas==2.2.3
pydantic==2.9.2
scikit-learn==1.5.2
redis==5.1.1
```

## 🏃‍♂️ Running the Server

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
uvicorn app.main:app --reload
```

The server will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## 📝 API Endpoints

### Reviews
- `GET /total-score` - Get Total Score
- `GET /reviews-count` - Get Reviews Count
- `GET /latest-reviews/{n}` - Get Latest Reviews
- `GET /duplicate-reviews` - Get Duplicated Reviews

### Analytics
- `GET /word-cloud/` - Get Word Cloud Data

## 🏗️ Project Structure

```
.
├── app/
│   ├── main.py
│   └── utils/
│       ├── data.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 🔄 Caching

The server uses Redis for caching frequently accessed data:
- Word cloud results (24 hours)
- Sentiment analysis results (24 hours)
- Authentication tokens (1 hour)

## 🔍 Performance Considerations

- The BERT model is loaded once at startup
- Redis caching reduces computation overhead
- Batch processing for multiple reviews

## 🤝 Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## 🔮 Future Improvements

- [ ] Add more analytics features
- [ ] Improve caching strategy
- [ ] Add export functionality for analytics results