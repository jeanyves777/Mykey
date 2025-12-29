"""
Random Forest Model for Forex ML System
=======================================

Random Forest classifier for ensemble voting.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""

    def __init__(self, config=None):
        """Initialize Random Forest model."""
        super().__init__('random_forest', config)
        self.model = self.build_model()

    def build_model(self) -> RandomForestClassifier:
        """Build Random Forest model."""
        if self.config and hasattr(self.config, 'ml'):
            return RandomForestClassifier(
                n_estimators=self.config.ml.rf_n_estimators,
                max_depth=self.config.ml.rf_max_depth,
                min_samples_split=self.config.ml.rf_min_samples_split,
                min_samples_leaf=self.config.ml.rf_min_samples_leaf,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the Random Forest model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_importances_ = self.model.feature_importances_

        # Calculate training metrics
        metrics = self.evaluate(X_train, y_train)
        metrics = {f'train_{k}': v for k, v in metrics.items()}

        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
