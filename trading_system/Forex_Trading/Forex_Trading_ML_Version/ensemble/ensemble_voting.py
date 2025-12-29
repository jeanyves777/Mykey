"""
Ensemble Voting System for Forex ML System
==========================================

5-model ensemble with weighted voting and dynamic weight adjustment.
"""

import warnings
import os
import json

# Suppress sklearn and ML library warnings during inference
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=UserWarning, module='catboost')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/WARNING

import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib

from trading_system.Forex_Trading.Forex_Trading_ML_Version.ml_models.random_forest_model import RandomForestModel
from trading_system.Forex_Trading.Forex_Trading_ML_Version.ml_models.xgboost_model import XGBoostModel
from trading_system.Forex_Trading.Forex_Trading_ML_Version.ml_models.lightgbm_model import LightGBMModel
from trading_system.Forex_Trading.Forex_Trading_ML_Version.ml_models.catboost_model import CatBoostModel
from trading_system.Forex_Trading.Forex_Trading_ML_Version.ml_models.neural_network_model import NeuralNetworkModel
from trading_system.Forex_Trading.Forex_Trading_ML_Version.ensemble.dynamic_weighting import DynamicWeightManager


class EnsembleVotingSystem:
    """5-model ensemble voting system for Forex predictions."""

    # Class labels
    SELL = 0
    HOLD = 1
    BUY = 2

    def __init__(self, config=None):
        """
        Initialize ensemble voting system.

        Args:
            config: TradingConfig object
        """
        self.config = config

        # ML config defaults
        self.confidence_threshold = 0.60
        self.min_model_agreement = 3
        self.voting_method = 'soft'
        self.use_dynamic_weights = True

        if config and hasattr(config, 'ml'):
            self.confidence_threshold = config.ml.confidence_threshold
            self.min_model_agreement = config.ml.min_model_agreement
            self.voting_method = config.ml.voting_method
            self.use_dynamic_weights = config.ml.use_dynamic_weights

        # Initialize models
        self.models: Dict[str, 'BaseModel'] = {
            'random_forest': RandomForestModel(config),
            'xgboost': XGBoostModel(config),
            'lightgbm': LightGBMModel(config),
            'catboost': CatBoostModel(config),
            'neural_network': NeuralNetworkModel(config)
        }

        # Initial weights
        initial_weights = {
            'random_forest': 0.20,
            'xgboost': 0.25,
            'lightgbm': 0.20,
            'catboost': 0.20,
            'neural_network': 0.15
        }

        if config and hasattr(config, 'ml'):
            initial_weights = config.ml.model_weights

        # Weight manager
        self.weight_manager = DynamicWeightManager(
            model_names=list(self.models.keys()),
            initial_weights=initial_weights
        )

        # Prediction tracking
        self.last_predictions: Dict[str, int] = {}
        self.last_probabilities: Dict[str, np.ndarray] = {}
        self.last_confidence: float = 0.0
        self.last_agreement: int = 0

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of model name -> training metrics
        """
        results = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                metrics = model.train(X_train, y_train, X_val, y_val)
                results[name] = metrics
                print(f"  Train Accuracy: {metrics.get('train_accuracy', 0):.4f}")
                if 'val_accuracy' in metrics:
                    print(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
            except Exception as e:
                print(f"  Error training {name}: {e}")
                results[name] = {'error': str(e)}

        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.

        Args:
            X: Features

        Returns:
            Tuple of (predictions, confidences, agreements)
        """
        n_samples = X.shape[0]
        all_predictions = np.zeros((n_samples, len(self.models)))
        all_probabilities = np.zeros((n_samples, len(self.models), 3))

        # Get predictions from each model
        for i, (name, model) in enumerate(self.models.items()):
            if model.is_trained:
                try:
                    all_predictions[:, i] = model.predict(X)
                    all_probabilities[:, i, :] = model.predict_proba(X)
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
                    all_predictions[:, i] = self.HOLD
                    all_probabilities[:, i, :] = [0.0, 1.0, 0.0]  # HOLD

        # Ensemble voting
        if self.voting_method == 'soft':
            predictions, confidences = self._soft_vote(all_probabilities)
        else:
            predictions, confidences = self._hard_vote(all_predictions)

        # Calculate agreement
        agreements = self._calculate_agreement(all_predictions)

        return predictions, confidences, agreements

    def predict_single(self, X: np.ndarray) -> Tuple[int, float, int, Dict]:
        """
        Make prediction for a single sample with detailed output.

        Args:
            X: Features (single sample or batch)

        Returns:
            Tuple of (prediction, confidence, agreement, details)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get individual model predictions
        model_preds = {}
        model_probs = {}

        weights = self.weight_manager.get_weights()

        trained_model_names = []
        for name, model in self.models.items():
            if model.is_trained:
                trained_model_names.append(name)
                try:
                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0]
                    model_preds[name] = int(pred)
                    model_probs[name] = proba.tolist()
                except Exception as e:
                    model_preds[name] = self.HOLD
                    model_probs[name] = [0.0, 1.0, 0.0]

        # Store for tracking
        self.last_predictions = model_preds
        self.last_probabilities = model_probs

        # Ensemble prediction
        if self.voting_method == 'soft':
            # Weighted probability averaging
            weighted_proba = np.zeros(3)
            for name, proba in model_probs.items():
                weighted_proba += np.array(proba) * weights.get(name, 0.2)

            prediction = int(np.argmax(weighted_proba))
            confidence = float(weighted_proba[prediction])
        else:
            # Majority voting
            votes = list(model_preds.values())
            prediction = int(max(set(votes), key=votes.count))
            confidence = votes.count(prediction) / len(votes)

        # Agreement (how many models agree with final prediction)
        agreement = sum(1 for p in model_preds.values() if p == prediction)

        self.last_confidence = confidence
        self.last_agreement = agreement

        # Build details
        details = {
            'model_predictions': model_preds,
            'model_probabilities': model_probs,
            'weights': weights,
            'ensemble_prediction': prediction,
            'confidence': confidence,
            'agreement': agreement,
            'passes_threshold': confidence >= self.confidence_threshold and agreement >= self.min_model_agreement
        }

        return prediction, confidence, agreement, details

    def _soft_vote(self, all_probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted soft voting based on probabilities."""
        weights = self.weight_manager.get_weights()
        weight_array = np.array([weights[name] for name in self.models.keys()])

        # Weighted average of probabilities
        weighted_proba = np.average(all_probabilities, axis=1, weights=weight_array)

        predictions = np.argmax(weighted_proba, axis=1)
        confidences = np.max(weighted_proba, axis=1)

        return predictions, confidences

    def _hard_vote(self, all_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Majority voting."""
        predictions = []
        confidences = []

        for i in range(all_predictions.shape[0]):
            votes = all_predictions[i, :]
            prediction = int(max(set(votes), key=list(votes).count))
            confidence = list(votes).count(prediction) / len(votes)

            predictions.append(prediction)
            confidences.append(confidence)

        return np.array(predictions), np.array(confidences)

    def _calculate_agreement(self, all_predictions: np.ndarray) -> np.ndarray:
        """Calculate model agreement for each prediction."""
        agreements = []

        for i in range(all_predictions.shape[0]):
            votes = all_predictions[i, :]
            most_common = max(set(votes), key=list(votes).count)
            agreement = sum(1 for v in votes if v == most_common)
            agreements.append(agreement)

        return np.array(agreements)

    def update_weights(self, prediction: int, actual: int) -> None:
        """Update model weights based on prediction outcome."""
        if self.use_dynamic_weights:
            self.weight_manager.update(self.last_predictions, actual)

    def should_trade(self, confidence: float, agreement: int) -> bool:
        """Check if confidence and agreement meet thresholds."""
        return confidence >= self.confidence_threshold and agreement >= self.min_model_agreement

    def get_signal_name(self, prediction: int) -> str:
        """Convert prediction to signal name."""
        if prediction == self.BUY:
            return 'BUY'
        elif prediction == self.SELL:
            return 'SELL'
        else:
            return 'HOLD'

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        os.makedirs(path, exist_ok=True)

        # Save each model
        for name, model in self.models.items():
            model_path = os.path.join(path, name)
            model.save(model_path)

        # Save weights
        weights_path = os.path.join(path, 'weights.json')
        self.weight_manager.save(weights_path)

        # Save metadata
        meta = {
            'confidence_threshold': self.confidence_threshold,
            'min_model_agreement': self.min_model_agreement,
            'voting_method': self.voting_method,
            'use_dynamic_weights': self.use_dynamic_weights
        }
        meta_path = os.path.join(path, 'ensemble_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Ensemble saved to {path}")

    def load(self, path: str) -> None:
        """Load ensemble from disk."""
        # Load each model
        for name, model in self.models.items():
            model_path = os.path.join(path, name)
            if os.path.exists(model_path):
                try:
                    model.load(model_path)
                    print(f"  Loaded {name}")
                except Exception as e:
                    print(f"  Error loading {name}: {e}")

        # Load weights
        weights_path = os.path.join(path, 'weights.json')
        if os.path.exists(weights_path):
            self.weight_manager.load(weights_path)

        # Load metadata
        meta_path = os.path.join(path, 'ensemble_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.confidence_threshold = meta.get('confidence_threshold', 0.60)
                self.min_model_agreement = meta.get('min_model_agreement', 3)
                self.voting_method = meta.get('voting_method', 'soft')
                self.use_dynamic_weights = meta.get('use_dynamic_weights', True)

        print(f"Ensemble loaded from {path}")

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weight_manager.get_weights()

    def get_model_status(self) -> Dict[str, bool]:
        """Get training status of each model."""
        return {name: model.is_trained for name, model in self.models.items()}

    def __str__(self) -> str:
        status = self.get_model_status()
        trained = sum(status.values())
        return f"EnsembleVotingSystem({trained}/{len(self.models)} models trained, method={self.voting_method})"
