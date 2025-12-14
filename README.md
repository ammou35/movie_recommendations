# Movie Recommendation System

A comprehensive movie recommendation system combining supervised learning, unsupervised learning, and deep learning approaches to provide accurate movie quality predictions and personalized recommendations.

## Project Overview

This project implements a hybrid recommendation system that:

- Predicts movie quality scores using supervised machine learning
- Groups similar movies using unsupervised clustering
- Generates recommendations using deep learning embeddings
- Combines all three approaches for optimal results

## Features

### 1. Supervised Learning (Quality Prediction)

- Predicts movie quality scores (0-100) based on features
- Compares multiple models (Linear Regression, Ridge, Random Forest)
- Automatic hyperparameter optimization using GridSearchCV
- Achieves 56.6% variance explained (R² = 0.5657)
- 89.8% accuracy within ±10 points

### 2. Unsupervised Learning (Clustering & Similarity)

- Groups movies into clusters based on features
- Finds similar movies using cosine similarity
- Enables content-based recommendations

### 3. Deep Learning (Embeddings)

- Generates movie embeddings from text features
- Captures semantic relationships between movies
- Enables advanced similarity search

### 4. Data Processing

- Unified data loader for all models
- Feature engineering pipeline
- Handles 9,660 movies with 31+ features
- Parquet caching for fast loading

## Project Structure

```
movie_recommendations/
├── data/
│   ├── raw/                    # Raw CSV data
│   └── processed/              # Processed parquet files
├── src/
│   ├── models/
│   │   ├── supervised/         # Quality prediction model
│   │   │   ├── main.py        # Main entry point
│   │   │   ├── optimizer.py   # Hyperparameter optimization
│   │   │   ├── trainer.py     # Model training
│   │   │   ├── predictor.py   # Inference
│   │   │   └── feature_engineering.py
│   │   ├── unsupervised/      # Clustering & similarity
│   │   └── deep_learning/     # Embedding models
│   └── utils/
│       └── movie_data_loader.py  # Unified data loader
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_supervised_model_results.ipynb
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd movie_recommendations
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure data is in place:

```bash
# Place your imdb_movies.csv in data/raw/
ls data/raw/imdb_movies.csv
```

## Quick Start

### Train Supervised Model

Run the complete pipeline (train, compare, optimize, save):

```bash
python src/models/supervised/main.py
```

This will:

1. Load and prepare data
2. Train 3 models (Linear, Ridge, Random Forest)
3. Compare performance
4. Optimize best model with GridSearchCV
5. Save final model to `src/models/saved_models/supervised_model.pkl`

### Use Trained Model

```python
from models.supervised import QualityPredictor

# Load model
predictor = QualityPredictor('src/models/saved_models/supervised_model.pkl')

# Make prediction
score = predictor.predict_single(movie_features)
print(f"Predicted quality: {score:.1f}/100")
```

### Explore Data

Open and run the Jupyter notebooks:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Dataset

**Source**: IMDB Movies Dataset

**Key Features**:

- Temporal: year, month, day, movie_age
- Financial: budget, revenue, ROI, profit
- Text: overview_length, crew_size
- Categorical: genres (one-hot), language, country, status
