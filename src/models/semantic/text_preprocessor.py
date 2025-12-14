"""
Text Preprocessor for Semantic Embeddings

Cleans and normalizes movie descriptions for semantic analysis.
"""

import pandas as pd
import re
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
    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


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
    
    cleaned = descriptions.apply(clean_text)
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
    
    def __init__(self, min_length: int = 10):
        """
        Initialize the preprocessor.
        
        Args:
            min_length: Minimum character length for valid descriptions
        """
        self.min_length = min_length
        self.stats = None
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess movie data for semantic embeddings.
        
        Args:
            df: DataFrame with 'overview' column
            
        Returns:
            DataFrame with cleaned 'overview_processed' column
        """
        df = df.copy()
        
        df['overview_processed'] = preprocess_descriptions(
            df['overview'], 
            min_length=self.min_length
        )
        
        df = df[df['overview_processed'].str.len() > 0].reset_index(drop=True)
        self.stats = get_text_statistics(df['overview_processed'])
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the same preprocessing.
        
        Args:
            df: DataFrame with 'overview' column
            
        Returns:
            DataFrame with cleaned 'overview_processed' column
        """
        df = df.copy()
        
        df['overview_processed'] = preprocess_descriptions(
            df['overview'], 
            min_length=self.min_length
        )
        
        df = df[df['overview_processed'].str.len() > 0].reset_index(drop=True)
        
        return df
    
    def get_stats(self) -> dict:
        """Get preprocessing statistics."""
        return self.stats if self.stats else {}