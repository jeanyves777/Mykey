"""
Random Forest Model for Ensemble
=================================

Random Forest classifier optimized for high-frequency trading signals.
Captures non-linear patterns and provides feature importance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .base_model import BaseMLModel


class RandomForestModel(BaseMLModel):
    """
    Random Forest classifier for trading signal prediction.

    Strengths:
    - Handles non-linear relationships
    - Robust to outliers
    - Provides feature importance
    - Less prone to overfitting with proper tuning
    """

    def __init__(self, model_name: str = "random_forest"):
        """Initialize Random Forest model with default HFT-optimized parameters."""
        super().__init__(model_name, model_type='classifier')

        # Default hyperparameters optimized for HFT - FAST TRAINING
        self.hyperparameters = {
            'n_estimators': 100,  # Reduced for speed
            'max_depth': 10,  # Reduced for speed
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced',
            'bootstrap': True,
            'oob_score': True,
            'warm_start': False  # Could be set True for incremental training
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of training metrics
        """
        self.feature_names = list(X_train.columns)

        # Create model with hyperparameters
        self.model = RandomForestClassifier(**self.hyperparameters)

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)

        metrics = {
            'train_accuracy': train_accuracy,
            'oob_score': self.model.oob_score_ if self.hyperparameters['oob_score'] else 0.0
        }

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics['val_accuracy'] = val_metrics['accuracy']
            metrics['val_f1'] = val_metrics['f1']
            metrics['val_auc'] = val_metrics['auc']

        self.training_metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Ensure feature alignment
        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Random Forest."""
        if not self.is_trained:
            return None

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds

        Returns:
            Cross-validation metrics
        """
        model = RandomForestClassifier(**self.hyperparameters)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        return {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores.tolist()
        }

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                             param_grid: Optional[Dict] = None) -> Dict:
        """
        Tune hyperparameters using RandomizedSearchCV.

        Args:
            X: Features
            y: Labels
            param_grid: Parameter grid to search

        Returns:
            Best parameters found
        """
        from sklearn.model_selection import RandomizedSearchCV

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [5, 10, 20, 30],
                'min_samples_leaf': [5, 10, 15, 20],
                'max_features': ['sqrt', 'log2', 0.3, 0.5]
            }

        model = RandomForestClassifier(
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )

        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=50,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42
        )

        search.fit(X, y)

        self.hyperparameters.update(search.best_params_)

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_
        }
