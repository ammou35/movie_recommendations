# Semantic Movie Recommendation System

Text-based movie recommendations using semantic embeddings from movie descriptions.

## Overview

This module generates movie recommendations based on the semantic similarity of movie plot descriptions using SentenceTransformers embeddings.

## Features

- **Text Preprocessing**: Cleans and normalizes movie descriptions
- **Semantic Embeddings**: Generates 384-dimensional embeddings using `all-MiniLM-L6-v2`
- **Similarity-based Recommendations**: Finds movies with similar plots
- **Custom Description Search**: Get recommendations from a text description
- **Efficient Storage**: Saves embeddings in parquet and numpy formats

## Quick Start

### 1. Generate Embeddings
```bash
# Run the complete pipeline
python src/models/semantic/main.py
```

This will:
- Load and preprocess ~10,000 movie descriptions
- Generate semantic embeddings
- Save results to `data/processed/semantic_embeddings.parquet`
- Test recommendations on sample movies

### 2. Interactive Mode
```bash
# Test recommendations interactively
python src/models/semantic/main.py --interactive
```

### 3. Use in Code
```python
from src.models.semantic import SemanticEmbedder, SemanticRecommender
import pandas as pd

# Load saved embeddings
df = pd.read_parquet('data/processed/semantic_embeddings.parquet')
embed_cols = [col for col in df.columns if col.startswith('embed_')]
embeddings = df[embed_cols].values
df_clean = df.drop(columns=embed_cols)

# Create recommender
recommender = SemanticRecommender(df_clean, embeddings)

# Get recommendations
recs = recommender.get_similar_movies('Avatar', top_n=5)
print(recs[['names', 'similarity_score', 'score']])
```

## Visualization

Explore the analysis notebook:
```bash
jupyter notebook notebooks/03_semantic_embeddings_analysis.ipynb
```

The notebook includes:
- Text preprocessing analysis
- Embedding statistics
- t-SNE visualizations (colored by score and genre)
- Similarity heatmaps
- Genre-based analysis
- Interactive recommendation testing

## Module Structure
```
src/models/semantic/
├── __init__.py              # Module exports
├── main.py                  # Main entry point and pipeline
├── text_preprocessor.py     # Text cleaning and preprocessing
├── embedder.py              # Embedding generation
└── recommender.py           # Recommendation system
```

## Model Details

- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Embedding Dimension**: 384
- **Similarity Metric**: Cosine Similarity
- **Preprocessing**: Lowercase, URL removal, special character filtering

## Performance

- **Processing**: ~10,000 movies in ~30 seconds
- **Storage**: ~16MB (parquet with metadata)
- **Recommendation Speed**: Instant (with precomputed embeddings)

## Examples

**Similar to Avatar:**
- Battle: Los Angeles (0.489 similarity)
- Doom: Annihilation (0.453)
- Alien Resurrection (0.445)

**Similar to Toy Story:**
- Toy Story 2 (0.741 similarity)
- Toy Story 3 (0.740)
- Toy Story 4 (0.596)

## Requirements

See `requirements.txt` for dependencies.