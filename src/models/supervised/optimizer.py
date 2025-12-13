"""
Model Optimizer - Hyperparameter Tuning

Handles hyperparameter optimization using GridSearchCV.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any


class ModelOptimizer:
    """
    Optimizes model hyperparameters using GridSearchCV.
    """
    
    # Parameter grids for each model type
    PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        },
        'linear': {}
    }
    
    def __init__(self, model_type: str):
        """
        Initialize optimizer for specific model type.
        
        Args:
            model_type: 'linear', 'ridge', or 'random_forest'
        """
        self.model_type = model_type
        self.param_grid = self.PARAM_GRIDS.get(model_type, {})
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def _get_base_model(self):
        """Get base model instance."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(random_state=42)
        elif self.model_type == 'ridge':
            return Ridge()
        else:
            return LinearRegression()
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        test_size: float = 0.2
    ) -> Tuple[Any, Dict[str, float], Dict[str, Any]]:
        """
        Optimize hyperparameters using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            test_size: Test set proportion
            
        Returns:
            Tuple of (optimized_model, metrics, best_params)
        """
        if not self.param_grid:
            base_model = self._get_base_model()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            base_model.fit(X_train, y_train)
            
            y_pred = base_model.predict(X_test)
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
            return base_model, metrics, {}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        base_model = self._get_base_model()
        grid_search = GridSearchCV(
            base_model,
            self.param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'cv_score': self.best_score
        }
        
        return self.best_model, metrics, self.best_params


def optimize_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    initial_metrics: Dict[str, float]
) -> Tuple[Any, Dict[str, float], Dict[str, Any]]:
    """
    Convenience function to optimize a model.
    
    Args:
        model_type: 'linear', 'ridge', or 'random_forest'
        X: Feature matrix
        y: Target vector
        initial_metrics: Metrics from initial training
        
    Returns:
        Tuple of (optimized_model, metrics, best_params)
    """
    optimizer = ModelOptimizer(model_type)
    optimized_model, metrics, best_params = optimizer.optimize(X, y)
    
    if best_params:
        improvement = metrics['r2'] - initial_metrics['test_r2']
        print(f"   Optimized R²: {metrics['r2']:.4f} (Δ {improvement:+.4f})")
        print(f"   Optimized MAE: {metrics['mae']:.4f} (Δ {metrics['mae'] - initial_metrics['test_mae']:+.4f})")
    
    return optimized_model, metrics, best_params
