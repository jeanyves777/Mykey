"""
CatBoost Model for Ensemble
============================

CatBoost classifier optimized for high-frequency trading signals.
Handles categorical features well and has built-in overfitting prevention.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseMLModel


class CatBoostModel(BaseMLModel):
    """
    CatBoost classifier for trading signal prediction.

    Strengths:
    - Excellent handling of categorical features
    - Built-in overfitting prevention (ordered boosting)
    - Fast prediction time
    - No need for label encoding
    """

    def __init__(self, model_name: str = "catboost"):
        """Initialize CatBoost model with HFT-optimized parameters."""
        super().__init__(model_name, model_type='classifier')

        # Default hyperparameters optimized for HFT - FAST TRAINING
        self.hyperparameters = {
            'iterations': 200,  # Reduced for speed
            'depth': 6,  # Reduced for speed
            'learning_rate': 0.1,  # Increased for faster convergence
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'border_count': 128,
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1,
            'task_type': 'CPU',
            'auto_class_weights': 'Balanced',
            'early_stopping_rounds': 20  # Early stopping for speed
        }

    def _get_model(self):
        """Lazy load CatBoost to avoid import issues."""
        try:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**self.hyperparameters)
        except ImportError:
            raise ImportError("CatBoost not installed. Run: pip install catboost")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Train the CatBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of training metrics
        """
        self.feature_names = list(X_train.columns)

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)

        # Create model
        self.model = self._get_model()

        # Prepare eval set
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = (X_val, y_val_encoded)
        else:
            eval_set = None

        # Train with early stopping
        fit_params = {'verbose': False}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = 20

        self.model.fit(X_train, y_encoded, **fit_params)

        self.is_trained = True

        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions.flatten() == y_encoded)

        metrics = {
            'train_accuracy': train_accuracy,
            'best_iteration': self.model.get_best_iteration() if eval_set else self.hyperparameters['iterations']
        }

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            metrics['val_accuracy'] = np.mean(val_predictions.flatten() == y_val_encoded)

            val_metrics = self.evaluate(X_val, y_val)
            metrics['val_f1'] = val_metrics['f1']
            metrics['val_auc'] = val_metrics['auc']

        self.training_metrics = metrics
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        predictions_encoded = self.model.predict(X).flatten()
        return self.label_encoder.inverse_transform(predictions_encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from CatBoost."""
        if not self.is_trained:
            return None

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)

        return importance

    def get_shap_values(self, X: pd.DataFrame):
        """
        Get SHAP values for model interpretability.

        Args:
            X: Features to explain

        Returns:
            SHAP values array
        """
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            return shap_values
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
            return None

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                             param_grid: Optional[Dict] = None) -> Dict:
        """
        Tune hyperparameters using RandomizedSearchCV.

        Args:
            X: Features
            y: Labels
            param_grid: Parameter grid to search

        Returns:
            Best parameters found
        """
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.preprocessing import LabelEncoder
        from catboost import CatBoostClassifier

        if param_grid is None:
            param_grid = {
                'iterations': [100, 200, 300, 500],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 10],
                'random_strength': [0.5, 1, 2],
                'bagging_temperature': [0.5, 1, 2]
            }

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = CatBoostClassifier(
            loss_function='MultiClass',
            random_seed=42,
            verbose=False,
            thread_count=-1,
            auto_class_weights='Balanced'
        )

        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=30,
            cv=3,
            scoring='f1_weighted',
            n_jobs=1,  # CatBoost manages its own threads
            random_state=42
        )

        search.fit(X, y_encoded)

        self.hyperparameters.update(search.best_params_)

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_
        }

    def save(self, filepath: str):
        """Save model with label encoder."""
        import joblib
        from pathlib import Path

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model using CatBoost's native format
        self.model.save_model(str(save_path.with_suffix('.cbm')))

        # Save metadata
        joblib.dump({
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics
        }, str(save_path.with_suffix('.meta')))

    def load(self, filepath: str):
        """Load model with label encoder."""
        import joblib
        from catboost import CatBoostClassifier
        from pathlib import Path

        load_path = Path(filepath)

        # Load model
        self.model = CatBoostClassifier()
        self.model.load_model(str(load_path.with_suffix('.cbm')))

        # Load metadata
        meta = joblib.load(str(load_path.with_suffix('.meta')))
        self.label_encoder = meta['label_encoder']
        self.feature_names = meta['feature_names']
        self.hyperparameters = meta['hyperparameters']
        self.training_metrics = meta['training_metrics']

        self.is_trained = True
