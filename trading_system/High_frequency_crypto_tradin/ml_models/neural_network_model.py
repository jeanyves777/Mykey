"""
Neural Network Model for Ensemble
===================================

Deep learning model for trading signal prediction.
Uses scikit-learn's MLPClassifier for broad compatibility,
with optional PyTorch backend for advanced features.

This implementation prioritizes reliability over complexity.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseMLModel


class NeuralNetworkModel(BaseMLModel):
    """
    Neural Network classifier for trading signal prediction.

    Uses scikit-learn's MLPClassifier which:
    - Works on all Python versions (3.8-3.13+)
    - No GPU dependency issues
    - Fast training for moderate datasets
    - Supports early stopping

    For larger datasets or GPU acceleration, can switch to PyTorch backend.
    """

    def __init__(self, model_name: str = "neural_network", use_pytorch: bool = False):
        """Initialize Neural Network model with HFT-optimized parameters."""
        super().__init__(model_name, model_type='classifier')

        self.use_pytorch = use_pytorch

        # Default hyperparameters - FAST TRAINING
        self.hyperparameters = {
            'hidden_layer_sizes': (128, 64, 32),  # Network architecture
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,  # L2 regularization
            'batch_size': 256,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 100,  # Max epochs
            'early_stopping': True,
            'n_iter_no_change': 10,
            'validation_fraction': 0.1,
            'random_state': 42,
            'verbose': False,
        }

        self.scaler = None
        self.label_encoder = None

    def _build_sklearn_model(self):
        """Build scikit-learn MLPClassifier."""
        from sklearn.neural_network import MLPClassifier

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hyperparameters['hidden_layer_sizes'],
            activation=self.hyperparameters['activation'],
            solver=self.hyperparameters['solver'],
            alpha=self.hyperparameters['alpha'],
            batch_size=self.hyperparameters['batch_size'],
            learning_rate=self.hyperparameters['learning_rate'],
            learning_rate_init=self.hyperparameters['learning_rate_init'],
            max_iter=self.hyperparameters['max_iter'],
            early_stopping=self.hyperparameters['early_stopping'],
            n_iter_no_change=self.hyperparameters['n_iter_no_change'],
            validation_fraction=self.hyperparameters['validation_fraction'],
            random_state=self.hyperparameters['random_state'],
            verbose=self.hyperparameters['verbose'],
        )

        return self.model

    def _build_pytorch_model(self, input_dim: int, num_classes: int):
        """Build PyTorch model if available."""
        try:
            import torch
            import torch.nn as nn

            hidden_sizes = self.hyperparameters['hidden_layer_sizes']

            layers = []
            prev_size = input_dim

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, num_classes))

            self.pytorch_model = nn.Sequential(*layers)
            return self.pytorch_model

        except ImportError:
            return None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Train the Neural Network model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of training metrics
        """
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import accuracy_score, f1_score

        self.feature_names = list(X_train.columns)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Build and train sklearn model
        self._build_sklearn_model()

        # Train
        self.model.fit(X_scaled, y_encoded)

        self.is_trained = True

        # Calculate training metrics
        train_predictions = self.model.predict(X_scaled)
        train_accuracy = accuracy_score(y_encoded, train_predictions)
        train_f1 = f1_score(y_encoded, train_predictions, average='weighted')

        metrics = {
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'epochs_trained': self.model.n_iter_,
            'final_loss': self.model.loss_,
        }

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_scaled = self.scaler.transform(X_val)

            val_predictions = self.model.predict(X_val_scaled)
            metrics['val_accuracy'] = accuracy_score(y_val_encoded, val_predictions)
            metrics['val_f1'] = f1_score(y_val_encoded, val_predictions, average='weighted')

            # Calculate AUC
            try:
                from sklearn.metrics import roc_auc_score
                val_proba = self.model.predict_proba(X_val_scaled)
                metrics['val_auc'] = roc_auc_score(
                    y_val_encoded, val_proba, multi_class='ovr', average='weighted'
                )
            except:
                metrics['val_auc'] = 0.5

        self.training_metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Ensure we use the same features
        if set(self.feature_names).issubset(X.columns):
            X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)
        predictions_encoded = self.model.predict(X_scaled)

        return self.label_encoder.inverse_transform(predictions_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Ensure we use the same features
        if set(self.feature_names).issubset(X.columns):
            X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance using weight magnitudes from first layer.
        """
        if not self.is_trained:
            return None

        try:
            # Get first layer weights
            first_layer_weights = np.abs(self.model.coefs_[0])
            importance_scores = first_layer_weights.mean(axis=1)

            # Normalize
            importance_scores = importance_scores / importance_scores.sum()

            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)

            return importance
        except Exception as e:
            return None

    def get_training_history(self) -> Optional[Dict]:
        """Get training history."""
        if not self.is_trained:
            return None

        return {
            'loss_curve': self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else [],
            'n_iter': self.model.n_iter_,
            'best_loss': self.model.best_loss_ if hasattr(self.model, 'best_loss_') else None,
        }

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                             param_grid: Optional[Dict] = None) -> Dict:
        """
        Tune hyperparameters using grid search.

        Args:
            X: Features
            y: Labels
            param_grid: Parameter grid to search

        Returns:
            Best parameters found
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        if param_grid is None:
            param_grid = {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01],
            }

        # Prepare data
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Grid search
        base_model = MLPClassifier(
            max_iter=50,
            early_stopping=True,
            random_state=42,
            verbose=False,
        )

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_scaled, y_encoded)

        # Update hyperparameters with best found
        self.hyperparameters.update(grid_search.best_params_)

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
        }

    def save(self, filepath: str):
        """Save model with scaler and label encoder."""
        import joblib
        from pathlib import Path

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained,
        }, str(save_path))

    def load(self, filepath: str):
        """Load model with scaler and label encoder."""
        import joblib
        from pathlib import Path

        load_path = Path(filepath)

        data = joblib.load(str(load_path))

        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.hyperparameters = data['hyperparameters']
        self.training_metrics = data['training_metrics']
        self.is_trained = data['is_trained']


# Test
if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([-1, 0, 1], n_samples))

    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Train
    model = NeuralNetworkModel()
    metrics = model.train(X_train, y_train, X_val, y_val)

    print("Neural Network Training Results:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy: {metrics.get('val_accuracy', 'N/A')}")
    print(f"  Val F1: {metrics.get('val_f1', 'N/A')}")
    print(f"  Epochs: {metrics['epochs_trained']}")
