"""
Neural Network Model for Forex ML System
========================================

Deep learning model for ensemble voting.
"""

import numpy as np
from typing import Dict, Optional, List
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from .base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """Neural Network model implementation using TensorFlow/Keras."""

    def __init__(self, config=None):
        """Initialize Neural Network model."""
        super().__init__('neural_network', config)
        self.model = None
        self.hidden_layers = [256, 128, 64, 32]
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.epochs = 100
        self.batch_size = 64

        if config and hasattr(config, 'ml'):
            self.hidden_layers = config.ml.nn_hidden_layers
            self.dropout_rate = config.ml.nn_dropout_rate
            self.learning_rate = config.ml.nn_learning_rate
            self.epochs = config.ml.nn_epochs
            self.batch_size = config.ml.nn_batch_size

        self.num_classes = 3

    def build_model(self, input_dim: int = None) -> 'keras.Model':
        """Build Neural Network model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("TensorFlow is required for Neural Network model. Install with: pip install tensorflow")

        if input_dim is None:
            return None

        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))

        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the Neural Network model."""
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for Neural Network model")

        # Build model with correct input dimension
        self.model = self.build_model(input_dim=X_train.shape[1])

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        self.is_trained = True

        # Feature importances not directly available for NN
        # Use a placeholder
        self.feature_importances_ = np.ones(X_train.shape[1]) / X_train.shape[1]

        # Calculate metrics
        metrics = self.evaluate(X_train, y_train)
        metrics = {f'train_{k}': v for k, v in metrics.items()}

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet")
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X, verbose=0)

    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(path, exist_ok=True)

        if self.model is not None:
            model_path = os.path.join(path, f'{self.name}_model.keras')
            self.model.save(model_path)

        # Save metadata
        import json
        meta_path = os.path.join(path, f'{self.name}_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'is_trained': self.is_trained,
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate
            }, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for Neural Network model")

        model_path = os.path.join(path, f'{self.name}_model.keras')
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            self.is_trained = True

        # Load metadata
        import json
        meta_path = os.path.join(path, f'{self.name}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.is_trained = meta.get('is_trained', True)
