"""
MAIN ENTRY POINT - Supervised Quality Prediction Model

This is the ONLY file you need to run for the supervised model.
It handles everything: training, comparison, optimization, and saving.

Usage:
    python src/models/supervised/main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.movie_data_loader import MovieDataLoader
from models.supervised.feature_engineering import FeatureEngineer
from models.supervised.trainer import QualityModelTrainer
import numpy as np
from models.supervised.optimizer import optimize_model
import joblib


def main():
    """
    Complete supervised model pipeline:
    1. Load and prepare data
    2. Train and compare all models
    3. Select best model
    4. Save final model
    """
    print("="*80)
    print("SUPERVISED MODEL - COMPLETE PIPELINE")
    print("="*80)
    
    # ========== STEP 1: DATA PREPARATION ==========
    print("\n[STEP 1/4] Loading and preparing data...")
    loader = MovieDataLoader()
    df = loader.get_supervised_data()
    
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    X, y = engineer.prepare_features_and_target(df)
    X_scaled = engineer.fit_transform(X)
    
    print(f" Data ready: {len(df):,} movies, {len(engineer.feature_names)} features")
    
    # ========== STEP 2: TRAIN & COMPARE MODELS ==========
    print("\n[STEP 2/4] Training and comparing models...")
    
    models_to_test = {
        'Linear Regression': 'linear',
        'Ridge Regression': 'ridge',
        'Random Forest': 'random_forest'
    }
    
    results = {}
    trained_models = {}
    
    for model_name, model_type in models_to_test.items():
        print(f"  Training {model_name}...", end=" ")
        trainer = QualityModelTrainer(model_type=model_type)
        metrics = trainer.train_and_evaluate(X_scaled, y.values)
        results[model_name] = metrics
        trained_models[model_name] = trainer
        print(f" R²={metrics['test_r2']:.4f}")
    
    # ========== STEP 3: SELECT BEST MODEL ==========
    print("\n[STEP 3/5] Selecting best model...")
    
    # Find best by R² score
    best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    best_model_type = models_to_test[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n BEST MODEL: {best_model_name}")
    print(f"   Initial R²: {best_metrics['test_r2']:.4f}")
    print(f"   Initial MAE: {best_metrics['test_mae']:.4f}")
    
    # ========== STEP 4: OPTIMIZE HYPERPARAMETERS ==========
    print(f"\n[STEP 4/5] Optimizing {best_model_name} hyperparameters...")
        
    # Optimize the best model
    final_model, optimized_metrics, best_params = optimize_model(
        model_type=best_model_type,
        X=X_scaled,
        y=y.values,
        initial_metrics=best_metrics
    )
    
    # Calculate final accuracy
    predictions = final_model.predict(X_scaled)
    errors = np.abs(y.values - predictions)
    within_10 = (errors <= 10).sum() / len(errors) * 100
    
    # ========== STEP 5: SAVE FINAL MODEL ==========
    print(f"\n[STEP 5/5] Saving final optimized model...")
    
    final_path = 'src/models/saved_models/quality_predictor_best.pkl'
    
    # Save with metadata
    model_data = {
        'model': final_model,
        'model_type': best_model_name,
        'scaler': engineer.scaler,
        'feature_names': engineer.feature_names,
        'metrics': {
            'test_r2': optimized_metrics.get('r2', best_metrics['test_r2']),
            'test_rmse': optimized_metrics.get('rmse', best_metrics['test_rmse']),
            'test_mae': optimized_metrics.get('mae', best_metrics['test_mae']),
            'accuracy_within_10': within_10
        },
        'feature_importance': final_model.feature_importances_ if hasattr(final_model, 'feature_importances_') else None,
        'best_params': best_params if best_params else None
    }
    
    joblib.dump(model_data, final_path)
    
    print(f" Model saved to: {final_path}")
    


if __name__ == "__main__":
    main()
