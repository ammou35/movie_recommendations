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

### 3. Deep Learning (Semantic Embeddings)

- Generates movie embeddings from text features
- Captures semantic relationships between movies
- Enables advanced similarity search

### 4. Hybrid Recommender (The Final Product)

- **Combines** Semantic, Quality, and Clustering models
- **Weighted Scoring**: 60% Similarity, 20% Quality, 20% Cluster Affinity
- **Robust**: Handles missing data and aligns different model indices
- **Modular**: individual components can be swapped or retrained

### 5. Data Processing

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
│   │   │   └── feature_engineering.py # Feature engineering implementation
│   │   ├── semantic/      # Semantic Embeddings
│   │   │   ├── main.py        # Main entry point
│   │   │   ├── embedder.py # Embedder implementation
│   │   │   └── text_preprocessor.py # Text preprocessor implementation
│   │   │   └── recommender.py # Recommender implementation
│   │   ├── clustering/        # Movie Clustering
│   │   │   ├── main.py        # Main entry point
│   │   │   ├── movie_clusterer.py # Clusterer implementation
│   │   │   └── clustering_movie_recommender.py # Clustering recommender implementation
│   │   ├── hybrid/        # Hybrid Recommender
│   │   │   ├── main.py        # Main entry point
│   │   │   └── hybrid_recommender.py # Hybrid recommender implementation
│   └── utils/
│       └── movie_data_loader.py  # Unified data loader
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_supervised_model_results.ipynb
│   ├── 03_clustering_model_results.ipynb
│   ├── 04_semantic_model_results.ipynb
│   ├── 05_hybrid_model_visualization.ipynb
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

### Train Clustering Model

Run the clustering pipeline (train, save):

```bash
python src/models/clustering/main.py
```

This will:

1. Load and prepare data
2. Train MovieClusterer (default k=30)
3. Save model to `src/models/saved_models/movie_clusterer.pkl`

### Train Semantic Model

Run the semantic pipeline (generate embeddings):

```bash
python src/models/semantic/main.py
```

This will:

1. Load and preprocess movie descriptions
2. Generate embeddings using `all-MiniLM-L6-v2`
3. Save embeddings to `data/processed/semantic_embeddings.parquet`

### Interactive Mode

Test recommendations interactively in the terminal:

```bash
python src/models/semantic/main.py --interactive
```

### Run Hybrid Model

```bash
python src/models/hybrid/main.py "john wick"
```

This will:

1. Load and prepare data
2. Generate recommendations using hybrid recommender
3. Print top 10 recommendations

### Explore Data

Open and run the Jupyter notebooks:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_supervised_model_results.ipynb
jupyter notebook notebooks/03_clustering_model_results.ipynb
jupyter notebook notebooks/04_semantic_model_results.ipynb
jupyter notebook notebooks/05_hybrid_model_visualization.ipynb
```

## Dataset

**Source**: IMDB Movies Dataset

**Key Features**:

- Temporal: year, month, day, movie_age
- Financial: budget, revenue, ROI, profit
- Text: overview_length, crew_size
- Categorical: genres (one-hot), language, country, status
