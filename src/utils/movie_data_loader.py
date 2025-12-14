"""
MovieDataLoader - Unified data loading for all models

Provides clean, consistent data access for supervised, unsupervised, and deep learning models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


class MovieDataLoader:
    """
    Unified data loader for movie recommendation system.

    """
    
    def __init__(
        self, 
        data_path: str = None,
        processed_data_path: str = None,
        cache_processed: bool = True,
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to raw CSV file. If None, defaults to project_root/data/raw/imdb_movies.csv.
            processed_data_path: Path to processed parquet file. If None, defaults to project_root/data/processed/clean_data.parquet.
            cache_processed: If True, processed data will be loaded/saved from/to processed_data_path.
        """
        self.project_root = get_project_root()
        
        if data_path is None:
            self.data_path = self.project_root / "data" / "raw" / "imdb_movies.csv"
        else:
            self.data_path = Path(data_path) if Path(data_path).is_absolute() else self.project_root / data_path
        
        if processed_data_path is None:
            self.processed_data_path = self.project_root / "data" / "processed" / "clean_data.parquet"
        else:
            self.processed_data_path = Path(processed_data_path) if Path(processed_data_path).is_absolute() else self.project_root / processed_data_path
        
        self.cache_processed = cache_processed
        self.raw_data_path = str(self.data_path)
        self.processed_dir = self.processed_data_path.parent
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self._clean_data_cache = None
    
    def get_clean_data(self) -> pd.DataFrame:
        """
        Get cleaned base dataset (all columns, all movies).
        
        This is the foundation dataset that all other methods use.
        Data is cached after first load for performance.
        
        Returns:
            DataFrame with cleaned movie data
            
        """
        if self._clean_data_cache is not None:
            return self._clean_data_cache.copy()
        
        parquet_path = self.processed_dir / "clean_data.parquet"
        if parquet_path.exists():
            self._clean_data_cache = pd.read_parquet(parquet_path)
            return self._clean_data_cache.copy()
        
        df = self._load_and_clean_raw_data()
        
        df.to_parquet(parquet_path, index=False)
        
        self._clean_data_cache = df
        return df.copy()
    
    def get_supervised_data(self) -> pd.DataFrame:
        """
        Get data prepared for supervised learning (quality prediction).
        
        Returns:
            DataFrame ready for supervised learning (9,660 movies)
            
        """
        df = self.get_clean_data()
        
        return df
    
    def get_clustering_data(self, save: bool = False) -> pd.DataFrame:
        """
        Get data prepared for clustering (content-based similarity).
        
        Args:
            save: If True, save to clustering_data.parquet
            
        Returns:
            DataFrame with clustering features
            
        """
        df = self.get_clean_data()
        
        feature_cols = ['year', 'score', 'budget_x', 'revenue']
        clustering_df = df[feature_cols].copy()
        
        if save:
            save_path = self.processed_dir / "clustering_data.parquet"
            clustering_df.to_parquet(save_path, index=False)
        
        return clustering_df
    
    def get_embedding_data(self, save: bool = False) -> pd.DataFrame:
        """
        Get data prepared for embeddings (semantic similarity).
        
        Returns:
            DataFrame with embedding features
        """
        df = self.get_clean_data()
        
        feature_cols = ['year', 'score', 'budget_x', 'revenue']
        clustering_df = df[feature_cols].copy()
        
        if save:
            save_path = self.processed_dir / "clustering_data.parquet"
            clustering_df.to_parquet(save_path, index=False)
        
        return clustering_df
    
    def get_semantic_data(self, save: bool = False) -> pd.DataFrame:
        """
        Get data prepared for semantic embeddings (text-based similarity).
        
        Returns data with text descriptions and metadata needed for
        semantic analysis using sentence transformers.
        
        Args:
            save: If True, save to semantic_data.parquet
            
        Returns:
            DataFrame with text and metadata columns for semantic analysis
        """
        df = self.get_clean_data()
        
        semantic_cols = ['names', 'overview', 'genre', 'score', 'year', 'revenue']
        
        # Filter to only movies with overview text
        semantic_df = df[semantic_cols].copy()
        semantic_df = semantic_df.dropna(subset=['overview'])
        semantic_df = semantic_df[semantic_df['overview'].str.strip() != '']
        
        if save:
            save_path = self.processed_dir / "semantic_data.parquet"
            semantic_df.to_parquet(save_path, index=False)
        
        return semantic_df
    
    def get_custom_data(self, columns: List[str]) -> pd.DataFrame:
        """
        Get custom subset of columns.
        
        Args:
            columns: List of column names to include
            
        Returns:
            DataFrame with specified columns
            
        """
        df = self.get_clean_data()
        return df[columns].copy()
    
    def get_stats(self, detailed: bool = False) -> dict:
        """
        Get statistics about the dataset.
        
        Args:
            detailed: If True, include detailed analysis (duplicates, quality metrics)
        
        Returns:    
            Dictionary with dataset statistics
            
        """
        df = self.get_clean_data()
        
        stats = {
            'total_movies': len(df),
            'total_columns': len(df.columns),
            'score_mean': df['score'].mean(),
            'score_std': df['score'].std(),
            'score_min': df['score'].min(),
            'score_max': df['score'].max(),
            'missing_budget': df['budget_x'].isna().sum(),
            'missing_revenue': df['revenue'].isna().sum(),
            'year_min': int(df['year'].min()) if 'year' in df.columns and pd.notna(df['year'].min()) else 'N/A',
            'year_max': int(df['year'].max()) if 'year' in df.columns and pd.notna(df['year'].max()) else 'N/A'
        }
        
        if detailed:
            dup_names_mask = df['names'].duplicated(keep=False) & df['names'].notna()
            stats['duplicate_titles'] = df.loc[dup_names_mask, 'names'].nunique()
            stats['duplicate_title_rows'] = dup_names_mask.sum()
            
            stats['budget_zero'] = (df['budget_x'] == 0).sum()
            stats['revenue_zero'] = (df['revenue'] == 0).sum()
            stats['score_zero'] = (df['score'] == 0).sum()
            
            stats['nan_per_column'] = df.isna().sum().to_dict()
            stats['rows_with_any_nan'] = df.isna().any(axis=1).sum()
            
            budget_zero_mask = df['budget_x'] == 0
            revenue_zero_mask = df['revenue'] == 0
            row_has_nan_mask = df.isna().any(axis=1)
            
            bad_mask = budget_zero_mask | revenue_zero_mask | row_has_nan_mask
            good_mask = ~bad_mask
            
            stats['clean_movies'] = good_mask.sum()
            stats['clean_percentage'] = (good_mask.sum() / len(df)) * 100
            
            if 'genre' in df.columns:
                all_genres = []
                for genres_str in df['genre'].dropna():
                    all_genres.extend([g.strip() for g in str(genres_str).split(',')])
                
                from collections import Counter
                genre_counts = Counter(all_genres)
                stats['total_unique_genres'] = len(genre_counts)
                stats['top_5_genres'] = dict(genre_counts.most_common(5))
            
            if 'orig_lang' in df.columns:
                stats['unique_languages'] = df['orig_lang'].nunique()
                stats['top_3_languages'] = df['orig_lang'].value_counts().head(3).to_dict()
            
            if 'country' in df.columns:
                stats['unique_countries'] = df['country'].nunique()
                stats['top_3_countries'] = df['country'].value_counts().head(3).to_dict()
        
        return stats
    
    def print_stats(self, detailed: bool = False) -> None:
        """
        Print formatted statistics about the dataset.
        
        Args:
            detailed: If True, print detailed analysis
            
        """
        stats = self.get_stats(detailed=detailed)
        
        print("="*80)
        print("MOVIE DATASET STATISTICS")
        print("="*80)
        
        print(f"Basic Information:")
        print(f"  Total movies: {stats['total_movies']:,}")
        print(f"  Total columns: {stats['total_columns']}")
        print(f"  Year range: {stats['year_min']} to {stats['year_max']}")
        
        print(f"Score Statistics:")
        print(f"  Mean: {stats['score_mean']:.2f}")
        print(f"  Std Dev: {stats['score_std']:.2f}")
        print(f"  Range: {stats['score_min']:.1f} - {stats['score_max']:.1f}")
        
        print(f"Missing Data:")
        print(f"  Missing budget: {stats['missing_budget']}")
        print(f"  Missing revenue: {stats['missing_revenue']}")
        
        if detailed:
            print(f"Detailed Analysis:")
            print(f"  Duplicate titles: {stats['duplicate_titles']} ({stats['duplicate_title_rows']} rows)")
            print(f"  Budget = 0: {stats['budget_zero']}")
            print(f"  Revenue = 0: {stats['revenue_zero']}")
            print(f"  Score = 0: {stats['score_zero']}")
            print(f"  Rows with any NaN: {stats['rows_with_any_nan']}")
            
            print(f"Data Quality:")
            print(f"  Clean movies (no zeros, no NaN): {stats['clean_movies']} ({stats['clean_percentage']:.1f}%)")
            
            if 'top_5_genres' in stats:
                print(f"Top 5 Genres:")
                for genre, count in stats['top_5_genres'].items():
                    print(f"  {genre}: {count}")
            
            if 'top_3_languages' in stats:
                print(f"Top 3 Languages:")
                for lang, count in stats['top_3_languages'].items():
                    print(f"  {lang}: {count}")
        
        print("="*80)
    
    def _load_and_clean_raw_data(self) -> pd.DataFrame:
        """
        Internal method to load and clean raw CSV data.
        
        Performs:
        - Load CSV
        - Remove missing scores
        - Convert data types
        - Handle missing values
        - Parse dates
        
        Returns:
            Cleaned DataFrame
        """
        df = pd.read_csv(self.raw_data_path, na_values=["", " "], quotechar='"')
        
        df = df.dropna(subset=['score'])
        
        numeric_cols = ['score', 'budget_x', 'revenue']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'date_x' in df.columns:
            df['date_x'] = df['date_x'].astype(str).str.strip()
            df['date_x'] = pd.to_datetime(df['date_x'], format='%m/%d/%Y', errors='coerce')
            
            idx = df.columns.get_loc("date_x")
            df.insert(idx, "year", df['date_x'].dt.year)
        
        df = df.drop(columns=["date_x", "status", "orig_title"], errors="ignore")
        
        for col in ['budget_x', 'revenue']:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        categorical_cols = ['genre', 'orig_lang', 'country', 'overview', 'crew']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        return df


# functions for quick access
def load_supervised_data() -> pd.DataFrame:
    """Quick function to load supervised learning data."""
    return MovieDataLoader().get_supervised_data()


def load_clustering_data() -> pd.DataFrame:
    """Quick function to load clustering data."""
    return MovieDataLoader().get_clustering_data()


def load_embedding_data() -> pd.DataFrame:
    """Quick function to load embedding data."""
    return MovieDataLoader().get_embedding_data()

def load_semantic_data() -> pd.DataFrame:
    """Quick function to load semantic data."""
    return MovieDataLoader().get_semantic_data()

