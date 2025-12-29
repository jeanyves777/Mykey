"""
LightGBM Model for Forex ML System
==================================

LightGBM classifier for ensemble voting.
"""

import numpy as np
from typing import Dict, Optional
import lightgbm as lgb

from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""

    def __init__(self, config=None):
        """Initialize LightGBM model."""
        super().__init__('lightgbm', config)
        self.model = self.build_model()

    def build_model(self) -> lgb.LGBMClassifier:
        """Build LightGBM model."""
        if self.config and hasattr(self.config, 'ml'):
            return lgb.LGBMClassifier(
                n_estimators=self.config.ml.lgb_n_estimators,
                max_depth=self.config.ml.lgb_max_depth,
                learning_rate=self.config.ml.lgb_learning_rate,
                num_leaves=self.config.ml.lgb_num_leaves,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                verbose=-1
            )
        else:
            return lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                verbose=-1
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the LightGBM model."""
        callbacks = [lgb.early_stopping(50, verbose=False)] if X_val is not None else None
        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
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
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
