"""
Supervised learning module for movie quality prediction.

Usage:
    # Train model
    python src/models/supervised/main.py
    
    # Use model
    from models.supervised import QualityPredictor
    predictor = QualityPredictor('path/to/model.pkl')
"""

from .predictor import QualityPredictor

__all__ = ['QualityPredictor']
