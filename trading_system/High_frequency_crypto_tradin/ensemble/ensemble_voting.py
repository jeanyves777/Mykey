"""
Ensemble Voting System for High-Frequency Crypto Trading
==========================================================

Combines 5 powerful ML models using weighted voting:
1. Random Forest - Captures non-linear patterns
2. XGBoost - Gradient boosting excellence
3. LightGBM - Fast and memory efficient
4. CatBoost - Handles categorical features
5. Neural Network - Deep learning patterns

The final prediction is based on weighted voting of all models,
with dynamic weight adjustment based on recent performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib
import json
from datetime import datetime

from ..ml_models import (
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    NeuralNetworkModel
)
from .dynamic_weighting import DynamicWeightManager


class EnsembleVotingSystem:
    """
    Ensemble voting system that combines 5 ML models.

    Supports:
    - Soft voting (probability averaging)
    - Hard voting (majority vote)
    - Dynamic weight adjustment based on performance
    - Confidence thresholds for trade signals
    """

    def __init__(self,
                 voting_method: str = 'soft',
                 confidence_threshold: float = 0.6,
                 min_agreement: int = 3,
                 use_dynamic_weights: bool = True,
                 buy_probability_threshold: float = 0.22):
        """
        Initialize the Ensemble Voting System.

        Args:
            voting_method: 'soft' (probability) or 'hard' (majority)
            confidence_threshold: Minimum confidence for trade signal
            min_agreement: Minimum models that must agree for hard voting
            use_dynamic_weights: Whether to use dynamic weight adjustment
            buy_probability_threshold: For binary classification, predict BUY
                                      if P(BUY) > this threshold (default 0.22)
        """
        self.voting_method = voting_method
        self.confidence_threshold = confidence_threshold
        self.min_agreement = min_agreement
        self.use_dynamic_weights = use_dynamic_weights
        self.buy_probability_threshold = buy_probability_threshold

        # Initialize models
        self.models = {
            'random_forest': RandomForestModel('random_forest'),
            'xgboost': XGBoostModel('xgboost'),
            'lightgbm': LightGBMModel('lightgbm'),
            'catboost': CatBoostModel('catboost'),
            'neural_network': NeuralNetworkModel('neural_network')
        }

        self.model_names = list(self.models.keys())

        # Initialize weight manager
        initial_weights = {
            'random_forest': 0.20,
            'xgboost': 0.25,
            'lightgbm': 0.20,
            'catboost': 0.20,
            'neural_network': 0.15
        }

        self.weight_manager = DynamicWeightManager(
            model_names=self.model_names,
            initial_weights=initial_weights,
            strategy='hybrid'
        )

        # Tracking
        self.is_trained = False
        self.training_history = []
        self.prediction_history = []
        self.feature_names = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble with comprehensive logging.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress

        Returns:
            Dictionary of training metrics for each model
        """
        import time

        self.feature_names = list(X_train.columns)
        all_metrics = {}
        total_models = len(self.models)
        successful = 0
        failed = 0

        if verbose:
            print("")
            print("  " + "-" * 56)
            print("  TRAINING INDIVIDUAL MODELS")
            print("  " + "-" * 56)
            print(f"  Training samples: {len(X_train):,}")
            print(f"  Validation samples: {len(X_val) if X_val is not None else 0:,}")
            print(f"  Features: {len(self.feature_names)}")
            print("")

        for idx, (name, model) in enumerate(self.models.items(), 1):
            model_start = time.time()

            if verbose:
                print(f"  [{idx}/{total_models}] Training {name.upper()}...")
                print(f"        Status: Starting...", end='\r')

            try:
                metrics = model.train(X_train, y_train, X_val, y_val)
                all_metrics[name] = metrics
                successful += 1

                model_time = time.time() - model_start

                if verbose:
                    # Clear the line and print results
                    print(" " * 60, end='\r')
                    print(f"  [{idx}/{total_models}] {name.upper()}: COMPLETE ({model_time:.1f}s)")
                    print(f"        Train Accuracy: {metrics.get('train_accuracy', 0):.4f}")
                    if 'val_accuracy' in metrics:
                        print(f"        Val Accuracy:   {metrics['val_accuracy']:.4f}")
                        print(f"        Val F1 Score:   {metrics.get('val_f1', 0):.4f}")
                        print(f"        Val AUC:        {metrics.get('val_auc', 0):.4f}")
                    if 'best_iteration' in metrics:
                        print(f"        Best Iteration: {metrics['best_iteration']}")
                    if 'epochs_trained' in metrics:
                        print(f"        Epochs Trained: {metrics['epochs_trained']}")
                    print("")

            except Exception as e:
                model_time = time.time() - model_start
                failed += 1

                if verbose:
                    print(" " * 60, end='\r')
                    print(f"  [{idx}/{total_models}] {name.upper()}: FAILED ({model_time:.1f}s)")
                    print(f"        Error: {str(e)[:50]}...")
                    print("")

                all_metrics[name] = {'error': str(e), 'train_accuracy': 0}

        self.is_trained = True

        # Record training
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': all_metrics,
            'samples': len(X_train)
        })

        # Training summary
        if verbose:
            print("  " + "-" * 56)
            print("  MODEL TRAINING SUMMARY")
            print("  " + "-" * 56)
            print(f"  Models trained successfully: {successful}/{total_models}")
            print(f"  Models failed: {failed}/{total_models}")

        # Evaluate ensemble
        if X_val is not None and y_val is not None:
            if verbose:
                print("")
                print("  Evaluating ensemble performance...")

            ensemble_metrics = self.evaluate(X_val, y_val)
            all_metrics['ensemble'] = ensemble_metrics

            if verbose:
                print("")
                print("  " + "-" * 56)
                print("  ENSEMBLE PERFORMANCE")
                print("  " + "-" * 56)
                print(f"  Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
                print(f"  Ensemble F1 Score: {ensemble_metrics['f1']:.4f}")
                print(f"  Ensemble Precision: {ensemble_metrics['precision']:.4f}")
                print(f"  Ensemble Recall: {ensemble_metrics['recall']:.4f}")
                print(f"  Ensemble AUC: {ensemble_metrics['auc']:.4f}")

                # Print per-class metrics (CRITICAL for detecting HOLD bias)
                per_class = ensemble_metrics.get('per_class', {})
                if per_class:
                    print("")
                    print("  Per-Class Metrics:")
                    print("  " + "-" * 50)
                    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
                    print("  " + "-" * 50)
                    for cls_name in ['SELL', 'HOLD', 'BUY']:
                        if cls_name in per_class:
                            m = per_class[cls_name]
                            print(f"  {cls_name:<8} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10,}")

                # Print prediction distribution (to detect HOLD bias)
                pred_dist = ensemble_metrics.get('prediction_distribution', {})
                if pred_dist:
                    print("")
                    print("  Prediction Distribution:")
                    total_preds = sum(pred_dist.values())
                    for cls_name in ['SELL', 'HOLD', 'BUY']:
                        if cls_name in pred_dist:
                            count = pred_dist[cls_name]
                            pct = 100 * count / total_preds if total_preds > 0 else 0
                            print(f"    {cls_name}: {count:,} ({pct:.1f}%)")

                # Print confusion matrix nicely
                cm = ensemble_metrics.get('confusion_matrix', [])
                if cm and len(cm) > 0:
                    print("")
                    print("  Confusion Matrix:")
                    n_classes = len(cm)
                    if n_classes == 3:
                        labels = ['SELL', 'HOLD', 'BUY ']
                    elif n_classes == 2:
                        labels = ['NEG ', 'POS ']
                    else:
                        labels = [f'C{i}  ' for i in range(n_classes)]

                    print("            Predicted")
                    print("            " + "  ".join([f"{l:>6}" for l in labels[:len(cm[0])]]))
                    for i, row in enumerate(cm):
                        label = labels[i] if i < len(labels) else f'C{i}  '
                        print(f"  Actual {label}: " + "  ".join([f"{v:>6}" for v in row]))

        return all_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble voting.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")

        if self.voting_method == 'soft':
            return self._soft_vote(X)
        else:
            return self._hard_vote(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")

        return self._weighted_average_proba(X)

    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Make predictions with confidence scores and model agreement.

        Args:
            X: Features to predict on

        Returns:
            Tuple of (predictions, confidences, agreement_details)
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")

        # Get probabilities from all models
        all_probas = {}
        all_predictions = {}

        for name, model in self.models.items():
            try:
                all_probas[name] = model.predict_proba(X)
                raw_preds = model.predict(X)
                # Map binary predictions to signal format for consistency
                # Binary: 0=SELL -> -1, 1=BUY -> 1
                n_classes = all_probas[name].shape[1]
                if n_classes == 2:
                    all_predictions[name] = np.where(raw_preds == 0, -1, 1)
                else:
                    all_predictions[name] = raw_preds
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")

        # Calculate weighted average probabilities
        weights = self.weight_manager.get_weights()
        weighted_proba = np.zeros_like(list(all_probas.values())[0])

        total_weight = 0
        for name, proba in all_probas.items():
            weighted_proba += weights[name] * proba
            total_weight += weights[name]

        weighted_proba /= total_weight

        # Final predictions - need to map indices to proper signals
        n_classes = weighted_proba.shape[1]
        raw_indices = np.argmax(weighted_proba, axis=1)
        confidences = np.max(weighted_proba, axis=1)

        if n_classes == 2:
            # Binary classification: index 0=SELL, index 1=BUY
            # Map to signals: SELL=-1, BUY=1 (no HOLD in binary)
            predictions = np.where(raw_indices == 0, -1, 1)
        else:
            # 3-class: indices 0,1,2 -> signals -1,0,1
            predictions = raw_indices - 1

        # Model agreement
        agreement_details = {
            'individual_predictions': all_predictions,
            'individual_probas': {k: v.tolist() if len(v) < 10 else 'too_large'
                                  for k, v in all_probas.items()},
            'weights': weights,
            'agreement_count': self._count_agreement(all_predictions, predictions)
        }

        return predictions, confidences, agreement_details

    def _soft_vote(self, X: pd.DataFrame) -> np.ndarray:
        """Soft voting using weighted probability average.

        For binary classification (BUY/NO_TRADE), we use the configurable
        buy_probability_threshold to decide when to trade. The model tends
        to favor NO_TRADE, so we predict BUY if P(BUY) > threshold (not argmax!).

        This lets TP/SL manage risk instead of the ML being too conservative.
        """
        proba = self._weighted_average_proba(X)
        n_classes = proba.shape[1]

        if n_classes == 2:
            # Binary classification for HF scalping:
            # - Class 0 = NO_TRADE (don't enter)
            # - Class 1 = BUY (enter long position)
            #
            # DON'T USE ARGMAX! The model heavily favors NO_TRADE (~80%).
            # Instead, use the configurable buy_probability_threshold.
            buy_proba = proba[:, 1]  # Probability of BUY

            # Predict BUY (1) if P(BUY) > threshold, else NO_TRADE (0)
            predictions = (buy_proba > self.buy_probability_threshold).astype(int)
            return predictions
        else:
            # 3-class: argmax gives indices 0, 1, 2 but we need -1, 0, 1
            return np.argmax(proba, axis=1) - 1

    def _hard_vote(self, X: pd.DataFrame) -> np.ndarray:
        """Hard voting using weighted majority.

        For binary classification, we use a low threshold on weighted
        BUY votes to allow more trades (same philosophy as soft vote).
        """
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except:
                continue

        if not predictions:
            raise ValueError("No models available for prediction")

        # Detect number of classes from first prediction
        first_preds = list(predictions.values())[0]
        unique_classes = np.unique(first_preds)

        # Weighted voting
        weights = self.weight_manager.get_weights()
        n_samples = len(X)

        if len(unique_classes) == 2 and set(unique_classes).issubset({0, 1}):
            # Binary classification: 0=NO_TRADE, 1=BUY
            n_classes = 2
            class_offset = 0
        else:
            # 3-class: -1, 0, 1
            n_classes = 3
            class_offset = 1

        vote_matrix = np.zeros((n_samples, n_classes))

        for name, preds in predictions.items():
            for i, pred in enumerate(preds):
                class_idx = int(pred) + class_offset
                class_idx = max(0, min(n_classes - 1, class_idx))  # Clamp to valid range
                vote_matrix[i, class_idx] += weights[name]

        if n_classes == 2:
            # Binary classification for HF scalping:
            # DON'T USE ARGMAX! Use the configurable buy_probability_threshold.
            # If BUY weight > threshold of total weight, predict BUY.
            total_weight = vote_matrix.sum(axis=1, keepdims=True)
            buy_weight_ratio = vote_matrix[:, 1] / (total_weight.flatten() + 1e-10)
            final_predictions = (buy_weight_ratio > self.buy_probability_threshold).astype(int)
        else:
            # 3-class: indices 0,1,2 -> signals -1,0,1
            raw_indices = np.argmax(vote_matrix, axis=1)
            final_predictions = raw_indices - class_offset

        return final_predictions

    def _weighted_average_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate weighted average of prediction probabilities."""
        weights = self.weight_manager.get_weights()
        all_probas = []
        used_weights = []

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                all_probas.append(proba)
                used_weights.append(weights[name])
            except Exception as e:
                print(f"Warning: {name} predict_proba failed: {e}")

        if not all_probas:
            raise ValueError("All models failed to predict")

        # Normalize weights
        used_weights = np.array(used_weights)
        used_weights /= used_weights.sum()

        # Weighted average
        weighted_proba = np.zeros_like(all_probas[0])
        for proba, weight in zip(all_probas, used_weights):
            weighted_proba += weight * proba

        return weighted_proba

    def _count_agreement(self, all_predictions: Dict, final_predictions: np.ndarray) -> List[int]:
        """Count how many models agree with final prediction for each sample."""
        n_samples = len(final_predictions)
        agreement_counts = np.zeros(n_samples, dtype=int)

        for name, preds in all_predictions.items():
            agreement_counts += (preds == final_predictions).astype(int)

        return agreement_counts.tolist()

    def get_trade_signal(self, X: pd.DataFrame) -> Tuple[int, float, Dict]:
        """
        Get trading signal from ensemble with comprehensive details.

        Args:
            X: Current market features (single row or recent window)

        Returns:
            Tuple of (signal, confidence, details)
            signal: 1 (buy), -1 (sell), 0 (hold)
            confidence: Confidence score 0-1
            details: Additional information including individual model predictions
        """
        if len(X) == 0:
            return 0, 0.0, {'error': 'Empty input'}

        # Use last row if multiple
        X_current = X.iloc[[-1]] if len(X) > 1 else X

        predictions, confidences, details = self.predict_with_confidence(X_current)

        signal = int(predictions[0])
        confidence = float(confidences[0])

        # Extract individual model predictions for logging
        individual_preds = {}
        individual_confs = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(X_current)
                proba = model.predict_proba(X_current)
                raw_pred = int(pred[0])

                # Map binary predictions to signal format
                # Binary: 0=SELL -> -1, 1=BUY -> 1
                n_classes = proba.shape[1] if len(proba.shape) > 1 else 2
                if n_classes == 2:
                    mapped_pred = -1 if raw_pred == 0 else 1
                else:
                    mapped_pred = raw_pred  # Already -1, 0, 1

                individual_preds[name] = mapped_pred
                individual_confs[name] = float(np.max(proba[0]))
            except Exception as e:
                individual_preds[name] = 0
                individual_confs[name] = 0.0

        # Add to details for logging
        details['individual_predictions'] = individual_preds
        details['individual_confidences'] = individual_confs

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            signal = 0  # Hold if not confident enough

        # Check minimum agreement for hard voting
        if self.voting_method == 'hard':
            agreement = details['agreement_count'][0]
            if agreement < self.min_agreement:
                signal = 0

        details['final_signal'] = signal
        details['confidence'] = confidence

        return signal, confidence, details

    def update_performance(self, model_results: Dict[str, bool],
                           actual_return: float):
        """
        Update model performance tracking after trade result.

        Args:
            model_results: Dict mapping model name to whether it was correct
            actual_return: Actual return from the trade
        """
        for name, was_correct in model_results.items():
            self.weight_manager.update_prediction_result(
                name, was_correct, actual_return
            )

        # Update weights periodically
        if self.weight_manager.update_count % 10 == 0:
            self.weight_manager.update_weights()

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate ensemble performance with per-class metrics.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary of evaluation metrics including per-class precision/recall
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
            classification_report
        )

        predictions = self.predict(X)
        proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y, predictions, average='weighted', zero_division=0),
        }

        try:
            if proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(y, proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y, proba, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.5

        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        metrics['confusion_matrix'] = cm.tolist()

        # Per-class metrics (IMPORTANT for detecting HOLD bias)
        unique_classes = sorted(y.unique())
        per_class_precision = precision_score(y, predictions, average=None, zero_division=0, labels=unique_classes)
        per_class_recall = recall_score(y, predictions, average=None, zero_division=0, labels=unique_classes)
        per_class_f1 = f1_score(y, predictions, average=None, zero_division=0, labels=unique_classes)

        # Store per-class metrics
        metrics['per_class'] = {}
        class_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        for i, cls in enumerate(unique_classes):
            cls_name = class_names.get(cls, f'CLASS_{cls}')
            metrics['per_class'][cls_name] = {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i]),
                'support': int((y == cls).sum())
            }

        # Prediction distribution (to detect HOLD bias)
        pred_counts = {}
        for cls in unique_classes:
            cls_name = class_names.get(cls, f'CLASS_{cls}')
            pred_counts[cls_name] = int((predictions == cls).sum())
        metrics['prediction_distribution'] = pred_counts

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get aggregated feature importance from all models.

        Returns:
            DataFrame with feature names and importance scores
        """
        weights = self.weight_manager.get_weights()
        all_importance = {}

        for name, model in self.models.items():
            try:
                importance = model.get_feature_importance()
                if importance is not None:
                    for _, row in importance.iterrows():
                        feature = row['feature']
                        if feature not in all_importance:
                            all_importance[feature] = 0
                        all_importance[feature] += weights[name] * row['importance']
            except:
                continue

        if not all_importance:
            return pd.DataFrame()

        # Normalize
        total = sum(all_importance.values())
        for feature in all_importance:
            all_importance[feature] /= total

        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in all_importance.items()
        ]).sort_values('importance', ascending=False)

        return importance_df

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weight_manager.get_weights()

    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary for all models."""
        return self.weight_manager.get_performance_summary()

    def save(self, save_dir: str):
        """
        Save the entire ensemble.

        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for name, model in self.models.items():
            model_path = save_path / name
            model.save(str(model_path))

        # Save weight manager
        self.weight_manager.save(str(save_path / 'weights.json'))

        # Save ensemble metadata
        metadata = {
            'voting_method': self.voting_method,
            'confidence_threshold': self.confidence_threshold,
            'min_agreement': self.min_agreement,
            'use_dynamic_weights': self.use_dynamic_weights,
            'buy_probability_threshold': self.buy_probability_threshold,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }

        with open(save_path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Ensemble saved to {save_path}")

    def load(self, load_dir: str):
        """
        Load the entire ensemble.

        Args:
            load_dir: Directory containing saved models
        """
        load_path = Path(load_dir)

        # Load each model
        for name, model in self.models.items():
            model_path = load_path / name
            try:
                model.load(str(model_path))
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

        # Load weight manager
        weight_path = load_path / 'weights.json'
        if weight_path.exists():
            self.weight_manager.load(str(weight_path))

        # Load metadata
        meta_path = load_path / 'ensemble_metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            self.voting_method = metadata.get('voting_method', 'soft')
            self.confidence_threshold = metadata.get('confidence_threshold', 0.6)
            self.min_agreement = metadata.get('min_agreement', 3)
            self.use_dynamic_weights = metadata.get('use_dynamic_weights', True)
            self.buy_probability_threshold = metadata.get('buy_probability_threshold', 0.22)
            self.feature_names = metadata.get('feature_names', [])
            self.is_trained = metadata.get('is_trained', True)

        print(f"Ensemble loaded from {load_path}")


if __name__ == "__main__":
    # Test ensemble
    print("Testing Ensemble Voting System...")

    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([-1, 0, 1], n_samples))

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Create and train ensemble
    ensemble = EnsembleVotingSystem()
    metrics = ensemble.train(X_train, y_train, X_val, y_val)

    # Get trade signal
    signal, confidence, details = ensemble.get_trade_signal(X_val.iloc[:1])
    print(f"\nTrade Signal: {signal}, Confidence: {confidence:.4f}")
