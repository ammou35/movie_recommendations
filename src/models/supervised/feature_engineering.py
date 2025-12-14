"""
Feature engineering for supervised learning model.
Creates numerical and categorical features from raw movie data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from typing import Tuple, List
from datetime import datetime



class FeatureEngineer:
    """
    Feature engineering for movie quality prediction.
    Transforms raw features into model-ready features.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.genre_encoder = MultiLabelBinarizer()
        self.fitted = False
        self.feature_names = []
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from release year.
        
        Args:
            df: DataFrame with 'year' column (already extracted from date_x)
            
        Returns:
            DataFrame with added temporal features
        """
        df['release_year'] = df['year']
        
        current_year = datetime.now().year
        df['movie_age'] = current_year - df['release_year']
        
        df['is_recent'] = (df['movie_age'] <= 5).astype(int)
        
        df['release_year'] = df['release_year'].fillna(df['release_year'].median())
        df['movie_age'] = df['movie_age'].fillna(df['movie_age'].median())
        
        return df
    
    def extract_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract financial features from budget and revenue.
        
        Args:
            df: DataFrame with 'budget_x' and 'revenue' columns
            
        Returns:
            DataFrame with added financial features
        """
        df['revenue_to_budget_ratio'] = df['revenue'] / (df['budget_x'] + 1)
        
        df['log_revenue'] = np.log1p(df['revenue'])
        df['log_budget'] = np.log1p(df['budget_x'])
        
        df['has_budget'] = (df['budget_x'] > 0).astype(int)
        df['has_revenue'] = (df['revenue'] > 0).astype(int)
        
        df['profit'] = df['revenue'] - df['budget_x']
        df['log_profit'] = np.log1p(df['profit'].clip(lower=0))
        
        return df
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from text fields.
        
        Args:
            df: DataFrame with 'overview' and 'crew' columns
            
        Returns:
            DataFrame with added text features
        """
        df['overview_length'] = df['overview'].fillna('').apply(lambda x: len(str(x).split()))
        
        df['crew_size'] = df['crew'].fillna('').apply(lambda x: len(str(x).split(',')))
        
        df['has_overview'] = (df['overview_length'] > 0).astype(int)
        
        return df
    
    def encode_genres(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Encode genres using multi-label binarization.
        
        Args:
            df: DataFrame with 'genre' column
            top_n: Number of top genres to encode
            
        Returns:
            DataFrame with genre features
        """
        df['genre_list'] = df['genre'].fillna('Unknown').apply(
            lambda x: [g.strip() for g in str(x).split(',')]
        )
        
        df['genre_count'] = df['genre_list'].apply(len)
        
        all_genres = []
        for genres in df['genre_list']:
            all_genres.extend(genres)
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(top_n)]
        
        for genre in top_genres:
            col_name = f'genre_{genre.lower().replace(" ", "_")}'
            df[col_name] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded categorical features
        """
        
        df['is_english'] = (df['orig_lang'] == 'English').astype(int)
        
        df['is_us_production'] = (df['country'] == 'US').astype(int)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw data.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        
        df = self.extract_temporal_features(df)
        df = self.extract_financial_features(df)
        df = self.extract_text_features(df)
        df = self.encode_genres(df, top_n=10)
        df = self.encode_categorical(df)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature column names for modeling.
        
        Returns:
            List of feature column names
        """
        numerical_features = [
            'release_year', 'movie_age', 'is_recent',
            'budget_x', 'revenue', 'revenue_to_budget_ratio',
            'log_revenue', 'log_budget', 'has_budget', 'has_revenue',
            'profit', 'log_profit',
            'overview_length', 'crew_size', 'has_overview',
            'genre_count',
            'is_english', 'is_us_production'
        ]
        
        return numerical_features
    
    def prepare_features_and_target(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'score'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and target vector y.
        
        Args:
            df: DataFrame with all features
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        
        feature_cols = self.get_feature_columns()
        genre_cols = [col for col in df.columns if col.startswith('genre_') and col != 'genre_list']
        feature_cols.extend(genre_cols)
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        X = X.fillna(0)
        
        return X, y
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Scaled feature array
        """
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Scaled feature array
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        X_scaled = self.scaler.transform(X)
        return X_scaled
