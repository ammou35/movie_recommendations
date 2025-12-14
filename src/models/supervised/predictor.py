"""
Predictor for movie quality scores.
Loads trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Union


class QualityPredictor:
    """
    Predictor for movie quality scores.
    Uses trained supervised learning model.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model from disk."""
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data['model']
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quality scores for movies.
        
        Args:
            X: Feature matrix (must be preprocessed and scaled)
            
        Returns:
            Array of predicted quality scores
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        predictions = self.model.predict(X)
        
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def predict_single(self, features: np.ndarray) -> float:
        """
        Predict quality score for a single movie.
        
        Args:
            features: Feature vector for one movie
            
        Returns:
            Predicted quality score
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        prediction = self.predict(features)[0]
        return float(prediction)
    
    def predict_batch(self, X: np.ndarray, return_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict quality scores for multiple movies.
        
        Args:
            X: Feature matrix
            return_dataframe: If True, return DataFrame with predictions
            
        Returns:
            Predictions as array or DataFrame
        """
        predictions = self.predict(X)
        
        if return_dataframe:
            return pd.DataFrame({
                'predicted_quality_score': predictions
            })
        
        return predictions
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_data.get('model_type', 'unknown'),
            'metrics': self.model_data.get('metrics', {}),
            'has_feature_importance': self.model_data.get('feature_importance') is not None
        }

