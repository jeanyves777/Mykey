"""
Base Model Interface for ML Ensemble
=====================================

Abstract base class that all ML models must implement.
Provides common functionality for training, prediction, and evaluation.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import joblib
import json


class BaseMLModel(ABC):
    """
    Abstract base class for all ML models in the ensemble.

    All models must implement:
    - train(): Train the model on data
    - predict(): Make predictions
    - predict_proba(): Return prediction probabilities
    - save()/load(): Model persistence
    """

    def __init__(self, model_name: str, model_type: str = 'classifier'):
        """
        Initialize the base model.

        Args:
            model_name: Unique name for the model
            model_type: 'classifier' or 'regressor'
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
        self.hyperparameters = {}

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
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
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of prediction probabilities (n_samples, n_classes)
        """
        pass

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores if available.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, log_loss
        )

        predictions = self.predict(X)
        proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y, predictions, average='weighted', zero_division=0),
        }

        # Handle binary classification AUC
        try:
            if proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(y, proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y, proba, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.5

        try:
            metrics['log_loss'] = log_loss(y, proba)
        except:
            metrics['log_loss'] = float('inf')

        return metrics

    def save(self, filepath: str):
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, str(save_path))

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'hyperparameters': self.hyperparameters
        }

        meta_path = save_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, filepath: str):
        """
        Load the model from disk.

        Args:
            filepath: Path to the saved model
        """
        load_path = Path(filepath)

        # Load model
        self.model = joblib.load(str(load_path))

        # Load metadata
        meta_path = load_path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            self.model_name = metadata.get('model_name', self.model_name)
            self.model_type = metadata.get('model_type', self.model_type)
            self.is_trained = metadata.get('is_trained', True)
            self.feature_names = metadata.get('feature_names', [])
            self.training_metrics = metadata.get('training_metrics', {})
            self.hyperparameters = metadata.get('hyperparameters', {})

    def set_hyperparameters(self, params: Dict[str, Any]):
        """
        Set hyperparameters for the model.

        Args:
            params: Dictionary of hyperparameters
        """
        self.hyperparameters.update(params)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.model_name}, trained={self.is_trained})"
