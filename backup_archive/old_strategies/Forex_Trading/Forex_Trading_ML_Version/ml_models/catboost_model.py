"""
CatBoost Model for Forex ML System
==================================

CatBoost classifier for ensemble voting.
"""

import numpy as np
from typing import Dict, Optional
from catboost import CatBoostClassifier

from .base_model import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost model implementation."""

    def __init__(self, config=None):
        """Initialize CatBoost model."""
        super().__init__('catboost', config)
        self.model = self.build_model()

    def build_model(self) -> CatBoostClassifier:
        """Build CatBoost model."""
        if self.config and hasattr(self.config, 'ml'):
            return CatBoostClassifier(
                iterations=self.config.ml.cat_iterations,
                depth=self.config.ml.cat_depth,
                learning_rate=self.config.ml.cat_learning_rate,
                random_state=42,
                verbose=0,
                auto_class_weights='Balanced',
                thread_count=-1
            )
        else:
            return CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=0,
                auto_class_weights='Balanced',
                thread_count=-1
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the CatBoost model."""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
            verbose=False
        )
        self.is_trained = True
        self.feature_importances_ = self.model.feature_importances_

        # Calculate metrics
        metrics = self.evaluate(X_train, y_train)
        metrics = {f'train_{k}': v for k, v in metrics.items()}

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
