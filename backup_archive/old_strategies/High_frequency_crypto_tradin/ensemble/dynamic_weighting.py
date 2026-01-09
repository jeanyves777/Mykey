"""
Dynamic Weight Manager for Ensemble Models
============================================

Manages and updates model weights based on recent performance.
Implements adaptive weighting strategies for the ensemble.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ModelPerformance:
    """Track individual model performance metrics."""
    model_name: str
    correct_predictions: int = 0
    total_predictions: int = 0
    profit_contribution: float = 0.0
    recent_accuracy: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions

    @property
    def rolling_accuracy(self) -> float:
        """Calculate rolling accuracy from recent predictions."""
        if len(self.recent_accuracy) == 0:
            return 0.5
        return np.mean(list(self.recent_accuracy))

    @property
    def sharpe_contribution(self) -> float:
        """Calculate Sharpe-like metric from recent returns."""
        if len(self.recent_returns) < 10:
            return 0.0
        returns = np.array(self.recent_returns)
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)


class DynamicWeightManager:
    """
    Manages dynamic weights for ensemble models based on recent performance.

    Weighting Strategies:
    1. Accuracy-based: Weight by recent prediction accuracy
    2. Profit-based: Weight by profit contribution
    3. Sharpe-based: Weight by risk-adjusted returns
    4. Confidence-based: Weight by prediction confidence
    5. Hybrid: Combine multiple strategies
    """

    def __init__(self,
                 model_names: List[str],
                 initial_weights: Optional[Dict[str, float]] = None,
                 strategy: str = 'hybrid',
                 learning_rate: float = 0.1,
                 min_weight: float = 0.05,
                 max_weight: float = 0.5):
        """
        Initialize the Dynamic Weight Manager.

        Args:
            model_names: List of model names in the ensemble
            initial_weights: Initial weight distribution (default: equal)
            strategy: Weighting strategy ('accuracy', 'profit', 'sharpe', 'confidence', 'hybrid')
            learning_rate: Rate at which weights adapt (0-1)
            min_weight: Minimum weight for any model
            max_weight: Maximum weight for any model
        """
        self.model_names = model_names
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initialize weights
        if initial_weights is None:
            equal_weight = 1.0 / len(model_names)
            self.weights = {name: equal_weight for name in model_names}
        else:
            self.weights = initial_weights.copy()

        # Initialize performance tracking
        self.performance = {
            name: ModelPerformance(model_name=name)
            for name in model_names
        }

        # Historical weight tracking
        self.weight_history = []
        self.update_count = 0

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()

    def get_weight_array(self) -> np.ndarray:
        """Get weights as numpy array in model order."""
        return np.array([self.weights[name] for name in self.model_names])

    def update_prediction_result(self,
                                  model_name: str,
                                  was_correct: bool,
                                  actual_return: float = 0.0,
                                  confidence: float = 1.0):
        """
        Update performance metrics after a prediction.

        Args:
            model_name: Name of the model
            was_correct: Whether the prediction was correct
            actual_return: Actual return from the trade
            confidence: Model's confidence in the prediction
        """
        if model_name not in self.performance:
            return

        perf = self.performance[model_name]
        perf.total_predictions += 1

        if was_correct:
            perf.correct_predictions += 1

        perf.recent_accuracy.append(1 if was_correct else 0)
        perf.recent_returns.append(actual_return * confidence)
        perf.profit_contribution += actual_return

    def update_weights(self):
        """
        Update model weights based on recent performance.

        Called periodically to adapt weights to model performance.
        """
        self.update_count += 1

        if self.strategy == 'accuracy':
            new_weights = self._accuracy_based_weights()
        elif self.strategy == 'profit':
            new_weights = self._profit_based_weights()
        elif self.strategy == 'sharpe':
            new_weights = self._sharpe_based_weights()
        elif self.strategy == 'confidence':
            new_weights = self._confidence_based_weights()
        else:  # hybrid
            new_weights = self._hybrid_weights()

        # Apply learning rate for smooth transitions
        for name in self.model_names:
            self.weights[name] = (
                (1 - self.learning_rate) * self.weights[name] +
                self.learning_rate * new_weights[name]
            )

        # Apply weight constraints
        self._apply_constraints()

        # Record history
        self.weight_history.append({
            'update': self.update_count,
            'weights': self.weights.copy()
        })

    def _accuracy_based_weights(self) -> Dict[str, float]:
        """Calculate weights based on rolling accuracy."""
        accuracies = {}
        for name in self.model_names:
            acc = self.performance[name].rolling_accuracy
            # Transform to weight (better than random = higher weight)
            accuracies[name] = max(acc - 0.3, 0.1)  # Baseline at 30% accuracy

        # Normalize
        total = sum(accuracies.values())
        return {name: acc / total for name, acc in accuracies.items()}

    def _profit_based_weights(self) -> Dict[str, float]:
        """Calculate weights based on profit contribution."""
        profits = {}
        for name in self.model_names:
            profit = self.performance[name].profit_contribution
            # Shift to positive and add baseline
            profits[name] = max(profit + 1, 0.1)

        total = sum(profits.values())
        return {name: p / total for name, p in profits.items()}

    def _sharpe_based_weights(self) -> Dict[str, float]:
        """Calculate weights based on Sharpe-like metric."""
        sharpes = {}
        for name in self.model_names:
            sharpe = self.performance[name].sharpe_contribution
            # Transform to weight
            sharpes[name] = max(sharpe + 1, 0.1)

        total = sum(sharpes.values())
        return {name: s / total for name, s in sharpes.items()}

    def _confidence_based_weights(self) -> Dict[str, float]:
        """Calculate weights based on prediction confidence calibration."""
        # This requires tracking confidence vs accuracy
        # For now, use accuracy as proxy
        return self._accuracy_based_weights()

    def _hybrid_weights(self) -> Dict[str, float]:
        """
        Calculate hybrid weights combining multiple strategies.

        Combines:
        - 40% accuracy-based
        - 30% profit-based
        - 30% Sharpe-based
        """
        acc_weights = self._accuracy_based_weights()
        profit_weights = self._profit_based_weights()
        sharpe_weights = self._sharpe_based_weights()

        hybrid = {}
        for name in self.model_names:
            hybrid[name] = (
                0.4 * acc_weights[name] +
                0.3 * profit_weights[name] +
                0.3 * sharpe_weights[name]
            )

        # Normalize
        total = sum(hybrid.values())
        return {name: w / total for name, w in hybrid.items()}

    def _apply_constraints(self):
        """Apply min/max weight constraints and renormalize."""
        # Apply min/max constraints
        for name in self.model_names:
            self.weights[name] = np.clip(
                self.weights[name],
                self.min_weight,
                self.max_weight
            )

        # Renormalize to sum to 1
        total = sum(self.weights.values())
        for name in self.model_names:
            self.weights[name] /= total

    def reset_performance(self):
        """Reset all performance tracking."""
        for name in self.model_names:
            self.performance[name] = ModelPerformance(model_name=name)

    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of model performance."""
        data = []
        for name in self.model_names:
            perf = self.performance[name]
            data.append({
                'model': name,
                'weight': self.weights[name],
                'total_predictions': perf.total_predictions,
                'accuracy': perf.accuracy,
                'rolling_accuracy': perf.rolling_accuracy,
                'profit_contribution': perf.profit_contribution,
                'sharpe_contribution': perf.sharpe_contribution
            })

        return pd.DataFrame(data)

    def save(self, filepath: str):
        """Save weight manager state."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model_names': self.model_names,
            'weights': self.weights,
            'strategy': self.strategy,
            'learning_rate': self.learning_rate,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'update_count': self.update_count,
            'performance': {
                name: {
                    'correct_predictions': p.correct_predictions,
                    'total_predictions': p.total_predictions,
                    'profit_contribution': p.profit_contribution
                }
                for name, p in self.performance.items()
            }
        }

        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: str):
        """Load weight manager state."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.model_names = state['model_names']
        self.weights = state['weights']
        self.strategy = state['strategy']
        self.learning_rate = state['learning_rate']
        self.min_weight = state['min_weight']
        self.max_weight = state['max_weight']
        self.update_count = state['update_count']

        for name, perf_data in state['performance'].items():
            if name in self.performance:
                self.performance[name].correct_predictions = perf_data['correct_predictions']
                self.performance[name].total_predictions = perf_data['total_predictions']
                self.performance[name].profit_contribution = perf_data['profit_contribution']
