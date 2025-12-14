"""
Text Preprocessor for Semantic Embeddings

Cleans and normalizes movie descriptions for semantic analysis.
"""

import pandas as pd
import numpy as np
import re
import ast
from typing import List, Union


def clean_text(text: str) -> str:
    """
    Clean and normalize text for semantic analysis.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and normalized text
    """
    if pd.isna(text) or text == '':
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces and apostrophes
    # This preserves natural language structure for embeddings
    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_genre_names(genre_str: str) -> List[str]:
    """
    Extract genre names from genre string.
    
    Handles different formats:
    - "Action, Drama, Thriller" (IMDB format)
    - "[{'id': 18, 'name': 'Drama'}, ...]" (MovieLens format)
    - NaN/empty/Unknown
    
    Args:
        genre_str: Genre string in various formats
        
    Returns:
        List of genre names
    """
    if pd.isna(genre_str) or genre_str == '' or genre_str == 'Unknown':
        return []
    
    # Try MovieLens format first (list of dicts)
    try:
        parsed = ast.literal_eval(str(genre_str))
        if isinstance(parsed, list):
            return [
                g.get("name", "").strip()
                for g in parsed
                if isinstance(g, dict) and g.get("name")
            ]
    except (ValueError, SyntaxError):
        pass
    
    # Try IMDB format (comma-separated)
    if isinstance(genre_str, str):
        genres = [g.strip() for g in genre_str.split(',')]
        return [g for g in genres if g and g != 'Unknown']
    
    return []


def prepare_genre_text(genre_str: str) -> str:
    """
    Prepare genre text for embedding.
    
    Converts genres to underscore format for better tokenization.
    Example: "Science Fiction" -> "science_fiction"
    
    Args:
        genre_str: Raw genre string
        
    Returns:
        Formatted genre text (empty string if no genres)
    """
    genre_list = extract_genre_names(genre_str)
    if not genre_list:
        return ""
    
    # Convert to underscore format for better tokenization
    # "Science Fiction" -> "science_fiction" (single token)
    genre_tokens = [g.lower().replace(" ", "_") for g in genre_list]
    return " ".join(genre_tokens)


def preprocess_descriptions(
    descriptions: Union[pd.Series, List[str]], 
    min_length: int = 10
) -> pd.Series:
    """
    Preprocess a series of text descriptions.
    
    Args:
        descriptions: Series or list of text descriptions
        min_length: Minimum character length for valid descriptions
        
    Returns:
        Series of cleaned descriptions
    """
    if isinstance(descriptions, list):
        descriptions = pd.Series(descriptions)
    
    # Apply cleaning function
    cleaned = descriptions.apply(clean_text)
    
    # Filter out very short descriptions (likely not meaningful)
    cleaned = cleaned[cleaned.str.len() >= min_length]
    
    return cleaned


def get_text_statistics(descriptions: pd.Series) -> dict:
    """
    Get statistics about text descriptions.
    
    Args:
        descriptions: Series of text descriptions
        
    Returns:
        Dictionary with text statistics
    """
    word_counts = descriptions.str.split().str.len()
    char_counts = descriptions.str.len()
    
    stats = {
        'total_descriptions': len(descriptions),
        'avg_word_count': word_counts.mean(),
        'median_word_count': word_counts.median(),
        'min_word_count': word_counts.min(),
        'max_word_count': word_counts.max(),
        'avg_char_count': char_counts.mean(),
        'median_char_count': char_counts.median(),
        'min_char_count': char_counts.min(),
        'max_char_count': char_counts.max(),
    }
    
    return stats


class TextPreprocessor:
    """
    Text preprocessing pipeline for semantic embeddings.
    """
    
    def __init__(
        self, 
        min_length: int = 10,
        include_title: bool = True,
        include_genres: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            min_length: Minimum character length for valid descriptions
            include_title: Whether to include movie title in embedding
            include_genres: Whether to include genres in embedding
        """
        self.min_length = min_length
        self.include_title = include_title
        self.include_genres = include_genres
        self.stats = None
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess movie data for semantic embeddings.
        
        Combines title (optional), genres (optional), and overview into
        a single text WITHOUT labels for optimal embedding.
        
        Format: "{title} {genres} {overview}"
        Example: "toy story animation comedy family led by woody..."
        
        Args:
            df: DataFrame with 'overview', 'names', and 'genre' columns
            
        Returns:
            DataFrame with 'overview_processed' column containing combined text
        """
        df = df.copy()
        
        # Start with cleaned overview
        df['overview_processed'] = df['overview'].apply(clean_text)
        
        # Prepare components
        components = []
        
        # Add title if requested
        if self.include_title and 'names' in df.columns:
            df['title_processed'] = df['names'].apply(clean_text)
            components.append('title_processed')
        
        # Add genres if requested
        if self.include_genres and 'genre' in df.columns:
            df['genres_text'] = df['genre'].apply(prepare_genre_text)
            components.append('genres_text')
        
        # Always include overview last
        components.append('overview_processed')
        
        # Combine all components WITHOUT labels
        if len(components) > 1:
            df['overview_processed'] = df.apply(
                lambda row: ' '.join(
                    str(row[col]) for col in components 
                    if pd.notna(row.get(col)) and str(row.get(col, '')).strip()
                ).strip(),
                axis=1
            )
        
        # Remove rows with empty processed text
        df = df[df['overview_processed'].str.len() >= self.min_length].reset_index(drop=True)
        
        # Calculate statistics
        self.stats = get_text_statistics(df['overview_processed'])
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the same preprocessing.
        
        Args:
            df: DataFrame with 'overview', 'names', and 'genre' columns
            
        Returns:
            DataFrame with 'overview_processed' column containing combined text
        """
        df = df.copy()
        
        # Start with cleaned overview
        df['overview_processed'] = df['overview'].apply(clean_text)
        
        # Prepare components
        components = []
        
        # Add title if requested
        if self.include_title and 'names' in df.columns:
            df['title_processed'] = df['names'].apply(clean_text)
            components.append('title_processed')
        
        # Add genres if requested
        if self.include_genres and 'genre' in df.columns:
            df['genres_text'] = df['genre'].apply(prepare_genre_text)
            components.append('genres_text')
        
        # Always include overview last
        components.append('overview_processed')
        
        # Combine all components WITHOUT labels
        if len(components) > 1:
            df['overview_processed'] = df.apply(
                lambda row: ' '.join(
                    str(row[col]) for col in components 
                    if pd.notna(row.get(col)) and str(row.get(col, '')).strip()
                ).strip(),
                axis=1
            )
        
        df = df[df['overview_processed'].str.len() >= self.min_length].reset_index(drop=True)
        
        return df
    
    def get_stats(self) -> dict:
        """Get preprocessing statistics."""
        return self.stats if self.stats else {}