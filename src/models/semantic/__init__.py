"""
Semantic Movie Recommendation System

Text-based movie recommendations using sentence embeddings.
"""

from .text_preprocessor import (
    TextPreprocessor, 
    clean_text, 
    preprocess_descriptions,
    extract_genre_names,
    prepare_genre_text
)
from .embedder import SemanticEmbedder
from .recommender import SemanticRecommender

__all__ = [
    'TextPreprocessor',
    'SemanticEmbedder',
    'SemanticRecommender',
    'clean_text',
    'preprocess_descriptions',
    'extract_genre_names',
    'prepare_genre_text',
]