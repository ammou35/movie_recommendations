"""
Model training for supervised learning.
Trains and evaluates regression models for movie quality prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Tuple, Dict
import os


class QualityModelTrainer:
    """
    Trainer for movie quality prediction models.
    Supports multiple regression algorithms.
    """
    
    def __init__(self, model_type: str = 'ridge'):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('linear', 'ridge', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self.feature_importance = None
        
        # Initialize model based on type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.model.fit(X_train, y_train)
        
        # Extract feature importance for tree-based models
        if self.model_type == 'random_forest':
            self.feature_importance = self.model.feature_importances_
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        set_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            set_name: Name of the dataset ('train', 'val', 'test')
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        metrics = {
            f'{set_name}_rmse': rmse,
            f'{set_name}_mae': mae,
            f'{set_name}_r2': r2
        }
        
        self.metrics.update(metrics)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted quality scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metrics
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.metrics = model_data.get('metrics', {})
        self.feature_importance = model_data.get('feature_importance', None)
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Dict[str, float]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Dictionary of all metrics
        """
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, test_size, val_size
        )
        
        # Train model
        self.train(X_train, y_train)
        
        # Evaluate on all sets
        self.evaluate(X_train, y_train, 'train')
        self.evaluate(X_val, y_val, 'val')
        self.evaluate(X_test, y_test, 'test')
        
        return self.metrics
