"""
LightGBM Model for Ensemble
============================

LightGBM classifier optimized for high-frequency trading signals.
Faster training than XGBoost with leaf-wise tree growth.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseMLModel


class LightGBMModel(BaseMLModel):
    """
    LightGBM classifier for trading signal prediction.

    Strengths:
    - Faster training than XGBoost
    - Lower memory usage
    - Handles categorical features natively
    - Leaf-wise growth for better accuracy
    """

    def __init__(self, model_name: str = "lightgbm"):
        """Initialize LightGBM model with HFT-optimized parameters."""
        super().__init__(model_name, model_type='classifier')

        # Default hyperparameters optimized for HFT - FAST TRAINING
        self.hyperparameters = {
            'n_estimators': 200,  # Reduced for speed
            'max_depth': 8,  # Reduced for speed
            'learning_rate': 0.1,  # Increased for faster convergence
            'num_leaves': 31,  # Reduced for speed
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary',  # Binary for BUY/NO_TRADE
            'metric': 'binary_logloss',
            'is_unbalance': True,  # Handle class imbalance
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True,
            'early_stopping_rounds': 20  # Early stopping for speed
        }

    def _get_model(self):
        """Lazy load LightGBM to avoid import issues."""
        try:
            import lightgbm as lgb
            params = self.hyperparameters.copy()

            # Remove LightGBM-specific params that aren't in sklearn API
            sklearn_params = {
                'n_estimators': params.pop('n_estimators', 300),
                'max_depth': params.pop('max_depth', 10),
                'learning_rate': params.pop('learning_rate', 0.05),
                'num_leaves': params.pop('num_leaves', 50),
                'min_child_samples': params.pop('min_child_samples', 20),
                'subsample': params.pop('subsample', 0.8),
                'colsample_bytree': params.pop('colsample_bytree', 0.8),
                'reg_alpha': params.pop('reg_alpha', 0.1),
                'reg_lambda': params.pop('reg_lambda', 1.0),
                'n_jobs': params.pop('n_jobs', -1),
                'random_state': params.pop('random_state', 42),
                'verbose': params.pop('verbose', -1),
                'force_col_wise': params.pop('force_col_wise', True)
            }

            return lgb.LGBMClassifier(**sklearn_params)
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Train the LightGBM model.

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

        # Prepare callbacks for early stopping
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

        # Prepare eval set for early stopping
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_encoded)]
        else:
            eval_set = None
            callbacks = []  # No early stopping without validation set

        # Train
        self.model.fit(
            X_train, y_encoded,
            eval_set=eval_set,
            callbacks=callbacks
        )

        self.is_trained = True

        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_encoded)

        metrics = {
            'train_accuracy': train_accuracy,
            'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else self.hyperparameters.get('n_estimators', 300)
        }

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            metrics['val_accuracy'] = np.mean(val_predictions == y_val_encoded)

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

        # Use booster directly if loaded from file
        if hasattr(self, 'booster') and self.booster is not None:
            probas = self.booster.predict(X)
            if len(probas.shape) == 1:
                predictions_encoded = (probas > 0.5).astype(int)
            else:
                predictions_encoded = np.argmax(probas, axis=1)
        else:
            predictions_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        # Use booster directly if loaded from file
        if hasattr(self, 'booster') and self.booster is not None:
            probas = self.booster.predict(X)
            if len(probas.shape) == 1:
                # Binary classification - return [1-p, p] format
                return np.column_stack([1 - probas, probas])
            return probas
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from LightGBM.

        Args:
            importance_type: 'gain', 'split', or 'weight'

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            return None

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

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
        import lightgbm as lgb

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 8, 10, 15, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [20, 31, 50, 70, 100],
                'min_child_samples': [10, 20, 30, 50],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
            }

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = lgb.LGBMClassifier(
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )

        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=50,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
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

        # Save model
        self.model.booster_.save_model(str(save_path.with_suffix('.lgb')))

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
        import lightgbm as lgb
        from pathlib import Path

        load_path = Path(filepath)

        # Load metadata first to get model params
        meta = joblib.load(str(load_path.with_suffix('.meta')))
        self.label_encoder = meta['label_encoder']
        self.feature_names = meta['feature_names']
        self.hyperparameters = meta['hyperparameters']
        self.training_metrics = meta['training_metrics']

        # Load booster directly - don't use sklearn wrapper
        self.booster = lgb.Booster(model_file=str(load_path.with_suffix('.lgb')))
        self.model = None  # We'll use booster directly

        self.is_trained = True
