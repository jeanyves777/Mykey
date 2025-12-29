"""
Base Model for Forex ML System
==============================

Abstract base class for all ML models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
import joblib
import os


class BaseModel(ABC):
    """Abstract base class for ML models."""

    def __init__(self, name: str, config=None):
        """
        Initialize base model.

        Args:
            name: Model name identifier
            config: Configuration object
        """
        self.name = name
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_importances_: Optional[np.ndarray] = None

    @abstractmethod
    def build_model(self) -> Any:
        """Build and return the model instance."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features

        Returns:
            Array of class probabilities
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }

        return metrics

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f'{self.name}_model.pkl')
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_importances': self.feature_importances_
        }, model_path)

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Directory path containing model
        """
        model_path = os.path.join(path, f'{self.name}_model.pkl')
        data = joblib.load(model_path)
        self.model = data['model']
        self.is_trained = data['is_trained']
        self.feature_importances_ = data.get('feature_importances')

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances if available."""
        return self.feature_importances_

    def __str__(self) -> str:
        return f"{self.name} (trained={self.is_trained})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})>"
