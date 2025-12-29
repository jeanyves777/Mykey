"""
Dynamic Weight Manager for Forex ML System
==========================================

Manages model weights based on recent performance.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
import json
import os


class DynamicWeightManager:
    """Manages dynamic model weights based on performance."""

    def __init__(self, model_names: List[str], initial_weights: Dict[str, float] = None,
                 window_size: int = 100, min_weight: float = 0.05, max_weight: float = 0.50):
        """
        Initialize dynamic weight manager.

        Args:
            model_names: List of model names
            initial_weights: Initial weight distribution
            window_size: Number of predictions to consider for weight updates
            min_weight: Minimum weight per model
            max_weight: Maximum weight per model
        """
        self.model_names = model_names
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initialize weights
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            equal_weight = 1.0 / len(model_names)
            self.weights = {name: equal_weight for name in model_names}

        # Performance tracking
        self.prediction_history: Dict[str, deque] = {
            name: deque(maxlen=window_size) for name in model_names
        }
        self.actual_history: deque = deque(maxlen=window_size)

    def update(self, model_predictions: Dict[str, int], actual: int) -> None:
        """
        Update weights based on prediction accuracy.

        Args:
            model_predictions: Dictionary of model name -> prediction
            actual: Actual outcome
        """
        self.actual_history.append(actual)

        for name in self.model_names:
            if name in model_predictions:
                correct = int(model_predictions[name] == actual)
                self.prediction_history[name].append(correct)

        # Recalculate weights after enough samples
        if len(self.actual_history) >= 20:
            self._recalculate_weights()

    def _recalculate_weights(self) -> None:
        """Recalculate weights based on recent accuracy."""
        accuracies = {}
        total_accuracy = 0.0

        for name in self.model_names:
            if len(self.prediction_history[name]) > 0:
                accuracy = sum(self.prediction_history[name]) / len(self.prediction_history[name])
                # Add small epsilon to avoid zero weights
                accuracy = max(accuracy, 0.01)
                accuracies[name] = accuracy
                total_accuracy += accuracy

        if total_accuracy > 0:
            # Normalize to get weights
            for name in self.model_names:
                raw_weight = accuracies.get(name, 0.01) / total_accuracy
                # Clip to min/max bounds
                self.weights[name] = np.clip(raw_weight, self.min_weight, self.max_weight)

            # Renormalize to ensure weights sum to 1
            total = sum(self.weights.values())
            for name in self.model_names:
                self.weights[name] /= total

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()

    def get_model_accuracies(self) -> Dict[str, float]:
        """Get recent accuracy for each model."""
        accuracies = {}
        for name in self.model_names:
            if len(self.prediction_history[name]) > 0:
                accuracies[name] = sum(self.prediction_history[name]) / len(self.prediction_history[name])
            else:
                accuracies[name] = 0.0
        return accuracies

    def save(self, path: str) -> None:
        """Save weights to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'weights': self.weights,
                'accuracies': self.get_model_accuracies()
            }, f, indent=2)

    def load(self, path: str) -> None:
        """Load weights from JSON."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.weights = data.get('weights', self.weights)

    def reset(self) -> None:
        """Reset weights to equal distribution."""
        equal_weight = 1.0 / len(self.model_names)
        self.weights = {name: equal_weight for name in self.model_names}

        for name in self.model_names:
            self.prediction_history[name].clear()
        self.actual_history.clear()

    def __str__(self) -> str:
        weights_str = ", ".join([f"{k}: {v:.2%}" for k, v in self.weights.items()])
        return f"DynamicWeightManager({weights_str})"
