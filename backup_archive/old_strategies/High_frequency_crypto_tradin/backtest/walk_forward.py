"""
Walk-Forward Validation Framework
===================================

Implements walk-forward analysis for robust out-of-sample testing:
1. Divide data into multiple train/test windows
2. Train on each window, test on the next
3. Aggregate results for realistic performance estimation
4. Prevent look-ahead bias and overfitting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: np.ndarray
    actual: np.ndarray
    probabilities: Optional[np.ndarray] = None


class WalkForwardValidator:
    """
    Walk-Forward Validation for time series ML models.

    This approach ensures:
    - No look-ahead bias (only uses past data to predict future)
    - Robust out-of-sample testing
    - Detection of model degradation over time
    - Realistic simulation of live trading
    """

    def __init__(self,
                 n_splits: int = 5,
                 train_ratio: float = 0.8,
                 min_train_size: int = 5000,
                 gap: int = 0,
                 expanding: bool = False):
        """
        Initialize Walk-Forward Validator.

        Args:
            n_splits: Number of walk-forward windows
            train_ratio: Ratio of data for training in each window
            min_train_size: Minimum training samples required
            gap: Gap between train and test (to simulate real-world delay)
            expanding: If True, use expanding window (includes all prior data)
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.min_train_size = min_train_size
        self.gap = gap
        self.expanding = expanding
        self.windows: List[WalkForwardWindow] = []
        self.results: List[WalkForwardResult] = []

    def create_windows(self, n_samples: int) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows.

        Args:
            n_samples: Total number of samples

        Returns:
            List of WalkForwardWindow objects
        """
        self.windows = []

        # Calculate window size
        window_size = n_samples // self.n_splits
        train_size = int(window_size * self.train_ratio)
        test_size = window_size - train_size

        # Ensure minimum training size
        if train_size < self.min_train_size:
            # Adjust number of splits
            adjusted_splits = n_samples // (self.min_train_size + test_size)
            if adjusted_splits < 2:
                raise ValueError(f"Not enough data for walk-forward. Need at least {self.min_train_size * 2} samples")
            self.n_splits = max(2, adjusted_splits)
            window_size = n_samples // self.n_splits
            train_size = int(window_size * self.train_ratio)
            test_size = window_size - train_size

        for i in range(self.n_splits):
            if self.expanding:
                # Expanding window: include all prior data
                train_start = 0
                train_end = (i + 1) * window_size - test_size - self.gap
            else:
                # Rolling window
                train_start = i * window_size
                train_end = train_start + train_size

            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples:
                break

            window = WalkForwardWindow(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_end - train_start,
                test_size=test_end - test_start
            )

            self.windows.append(window)

        return self.windows

    def validate(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 train_func: Callable,
                 predict_func: Callable,
                 predict_proba_func: Optional[Callable] = None,
                 verbose: bool = True) -> Dict:
        """
        Run walk-forward validation.

        Args:
            X: Features DataFrame
            y: Target Series
            train_func: Function to train model: train_func(X_train, y_train)
            predict_func: Function to predict: predict_func(X_test) -> predictions
            predict_proba_func: Optional function for probabilities
            verbose: Print progress

        Returns:
            Dictionary with aggregated results
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        n_samples = len(X)
        self.create_windows(n_samples)
        self.results = []

        all_predictions = []
        all_actuals = []
        all_probas = []

        if verbose:
            print("=" * 60)
            print("Walk-Forward Validation")
            print(f"Total samples: {n_samples}")
            print(f"Number of windows: {len(self.windows)}")
            print("=" * 60)

        for window in self.windows:
            if verbose:
                print(f"\nWindow {window.window_id + 1}/{len(self.windows)}")
                print(f"  Train: {window.train_start} - {window.train_end} ({window.train_size} samples)")
                print(f"  Test: {window.test_start} - {window.test_end} ({window.test_size} samples)")

            # Extract data for this window
            X_train = X.iloc[window.train_start:window.train_end]
            y_train = y.iloc[window.train_start:window.train_end]
            X_test = X.iloc[window.test_start:window.test_end]
            y_test = y.iloc[window.test_start:window.test_end]

            # Train model
            train_metrics = train_func(X_train, y_train)

            # Predict
            predictions = predict_func(X_test)
            probabilities = None
            if predict_proba_func is not None:
                try:
                    probabilities = predict_proba_func(X_test)
                except:
                    pass

            # Calculate test metrics
            test_metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'f1': f1_score(y_test, predictions, average='weighted', zero_division=0),
                'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, predictions, average='weighted', zero_division=0)
            }

            if verbose:
                print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"  Test F1: {test_metrics['f1']:.4f}")

            # Store result
            result = WalkForwardResult(
                window_id=window.window_id,
                train_metrics=train_metrics if isinstance(train_metrics, dict) else {},
                test_metrics=test_metrics,
                predictions=predictions,
                actual=y_test.values,
                probabilities=probabilities
            )
            self.results.append(result)

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            if probabilities is not None:
                all_probas.extend(probabilities)

        # Aggregate results
        aggregated = self._aggregate_results(all_predictions, all_actuals, all_probas)

        if verbose:
            print("\n" + "=" * 60)
            print("Aggregated Results")
            print("=" * 60)
            print(f"Overall Accuracy: {aggregated['accuracy']:.4f}")
            print(f"Overall F1: {aggregated['f1']:.4f}")
            print(f"Accuracy Std: {aggregated['accuracy_std']:.4f}")
            print(f"Performance Degradation: {aggregated['degradation']:.4f}")

        return aggregated

    def _aggregate_results(self,
                           all_predictions: List,
                           all_actuals: List,
                           all_probas: List) -> Dict:
        """Aggregate results from all windows."""
        from sklearn.metrics import accuracy_score, f1_score

        # Overall metrics
        accuracy = accuracy_score(all_actuals, all_predictions)
        f1 = f1_score(all_actuals, all_predictions, average='weighted', zero_division=0)

        # Per-window metrics
        window_accuracies = [r.test_metrics['accuracy'] for r in self.results]
        window_f1s = [r.test_metrics['f1'] for r in self.results]

        # Calculate degradation (trend in performance)
        if len(window_accuracies) > 1:
            x = np.arange(len(window_accuracies))
            slope = np.polyfit(x, window_accuracies, 1)[0]
            degradation = slope  # Negative = performance getting worse
        else:
            degradation = 0

        # Consistency check
        consistency = 1 - np.std(window_accuracies) / (np.mean(window_accuracies) + 1e-10)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'accuracy_std': np.std(window_accuracies),
            'f1_std': np.std(window_f1s),
            'accuracy_min': min(window_accuracies),
            'accuracy_max': max(window_accuracies),
            'degradation': degradation,
            'consistency': consistency,
            'window_accuracies': window_accuracies,
            'window_f1s': window_f1s,
            'n_predictions': len(all_predictions),
            'n_windows': len(self.results)
        }

    def get_combined_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all predictions and actuals combined."""
        all_preds = []
        all_actual = []

        for result in self.results:
            all_preds.extend(result.predictions)
            all_actual.extend(result.actual)

        return np.array(all_preds), np.array(all_actual)

    def get_window_results(self) -> pd.DataFrame:
        """Get results for each window as DataFrame."""
        data = []
        for result in self.results:
            row = {
                'window': result.window_id,
                **{f'test_{k}': v for k, v in result.test_metrics.items()},
                **{f'train_{k}': v for k, v in result.train_metrics.items()}
            }
            data.append(row)

        return pd.DataFrame(data)

    def plot_results(self, save_path: Optional[str] = None):
        """Plot walk-forward validation results."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Accuracy over time
            ax1 = axes[0, 0]
            window_ids = [r.window_id for r in self.results]
            accuracies = [r.test_metrics['accuracy'] for r in self.results]
            ax1.plot(window_ids, accuracies, 'b-o', label='Test Accuracy')
            ax1.axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.4f}')
            ax1.set_xlabel('Window')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy Over Walk-Forward Windows')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # F1 over time
            ax2 = axes[0, 1]
            f1_scores = [r.test_metrics['f1'] for r in self.results]
            ax2.plot(window_ids, f1_scores, 'g-o', label='Test F1')
            ax2.axhline(y=np.mean(f1_scores), color='r', linestyle='--', label=f'Mean: {np.mean(f1_scores):.4f}')
            ax2.set_xlabel('Window')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('F1 Score Over Walk-Forward Windows')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Train vs Test comparison
            ax3 = axes[1, 0]
            train_acc = [r.train_metrics.get('train_accuracy', 0) for r in self.results]
            test_acc = [r.test_metrics['accuracy'] for r in self.results]
            x = np.arange(len(window_ids))
            width = 0.35
            ax3.bar(x - width/2, train_acc, width, label='Train', alpha=0.8)
            ax3.bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
            ax3.set_xlabel('Window')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Train vs Test Accuracy')
            ax3.set_xticks(x)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Cumulative predictions
            ax4 = axes[1, 1]
            all_preds, all_actual = self.get_combined_predictions()
            cumulative_correct = np.cumsum(all_preds == all_actual)
            cumulative_total = np.arange(1, len(all_preds) + 1)
            rolling_accuracy = cumulative_correct / cumulative_total
            ax4.plot(rolling_accuracy, 'b-', alpha=0.7)
            ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Prediction #')
            ax4.set_ylabel('Cumulative Accuracy')
            ax4.set_title('Cumulative Accuracy Over All Windows')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")


def walk_forward_validate_ensemble(ensemble,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    n_splits: int = 5,
                                    verbose: bool = True) -> Dict:
    """
    Convenience function to run walk-forward validation on ensemble.

    Args:
        ensemble: EnsembleVotingSystem instance
        X: Features
        y: Targets
        n_splits: Number of walk-forward windows
        verbose: Print progress

    Returns:
        Validation results dictionary
    """
    validator = WalkForwardValidator(n_splits=n_splits)

    def train_func(X_train, y_train):
        metrics = ensemble.train(X_train, y_train, verbose=False)
        return metrics.get('ensemble', {})

    def predict_func(X_test):
        return ensemble.predict(X_test)

    def predict_proba_func(X_test):
        return ensemble.predict_proba(X_test)

    results = validator.validate(
        X, y,
        train_func=train_func,
        predict_func=predict_func,
        predict_proba_func=predict_proba_func,
        verbose=verbose
    )

    return results


if __name__ == "__main__":
    # Test walk-forward validation
    print("Testing Walk-Forward Validation...")

    from sklearn.ensemble import RandomForestClassifier

    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))

    # Simple model for testing
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_func(X_train, y_train):
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        return {'train_accuracy': train_acc}

    def predict_func(X_test):
        return model.predict(X_test)

    # Run validation
    validator = WalkForwardValidator(n_splits=5)
    results = validator.validate(X, y, train_func, predict_func)

    print("\nWindow Results:")
    print(validator.get_window_results())
