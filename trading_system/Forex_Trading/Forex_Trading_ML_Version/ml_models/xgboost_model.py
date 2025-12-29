"""
XGBoost Model for Forex ML System
=================================

XGBoost classifier for ensemble voting.
"""

import numpy as np
from typing import Dict, Optional
import xgboost as xgb

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    def __init__(self, config=None):
        """Initialize XGBoost model."""
        super().__init__('xgboost', config)
        self.model = self.build_model()

    def build_model(self) -> xgb.XGBClassifier:
        """Build XGBoost model."""
        if self.config and hasattr(self.config, 'ml'):
            return xgb.XGBClassifier(
                n_estimators=self.config.ml.xgb_n_estimators,
                max_depth=self.config.ml.xgb_max_depth,
                learning_rate=self.config.ml.xgb_learning_rate,
                subsample=self.config.ml.xgb_subsample,
                colsample_bytree=self.config.ml.xgb_colsample_bytree,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the XGBoost model."""
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
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
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        return self.model.predict_proba(X)
