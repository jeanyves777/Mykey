"""
XGBoost Model for Ensemble
===========================

XGBoost classifier optimized for high-frequency trading signals.
Known for winning Kaggle competitions and handling tabular data well.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseMLModel


class XGBoostModel(BaseMLModel):
    """
    XGBoost classifier for trading signal prediction.

    Strengths:
    - Excellent performance on tabular data
    - Handles missing values natively
    - Regularization prevents overfitting
    - Feature importance with SHAP values
    """

    def __init__(self, model_name: str = "xgboost"):
        """Initialize XGBoost model with HFT-optimized parameters."""
        super().__init__(model_name, model_type='classifier')

        # Default hyperparameters optimized for HFT - FAST TRAINING
        self.hyperparameters = {
            'n_estimators': 200,  # Reduced for speed
            'max_depth': 6,  # Reduced for speed
            'learning_rate': 0.1,  # Increased for faster convergence
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',  # Binary for BUY/NO_TRADE
            'eval_metric': 'logloss',
            'scale_pos_weight': 4.0,  # Upweight BUY class (minority)
            'use_label_encoder': False,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0,
            'early_stopping_rounds': 20  # Early stopping for speed
        }

    def _get_model(self):
        """Lazy load XGBoost to avoid import issues."""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(**self.hyperparameters)
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of training metrics
        """
        self.feature_names = list(X_train.columns)

        # Encode labels if necessary
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)

        # Detect binary vs multi-class and set appropriate objective
        n_classes = len(np.unique(y_encoded))
        if n_classes == 2:
            self.hyperparameters['objective'] = 'binary:logistic'
            self.hyperparameters['eval_metric'] = 'logloss'
        else:
            self.hyperparameters['objective'] = 'multi:softprob'
            self.hyperparameters['eval_metric'] = 'mlogloss'

        # Create model
        self.model = self._get_model()

        # Prepare eval set for early stopping
        eval_set = [(X_train, y_encoded)]
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set.append((X_val, y_val_encoded))

        # Train with early stopping
        fit_params = {
            'eval_set': eval_set,
            'verbose': False
        }

        # Only use early stopping if we have validation data
        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=20)

        self.model.fit(X_train, y_encoded, **fit_params)

        self.is_trained = True

        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_encoded)

        metrics = {
            'train_accuracy': train_accuracy,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.hyperparameters['n_estimators']
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

        predictions_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost."""
        if not self.is_trained:
            return None

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
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
        import xgboost as xgb

        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2, 0.3]
            }

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
            verbosity=0
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
        self.model.save_model(str(save_path.with_suffix('.xgb')))

        # Save label encoder and metadata
        joblib.dump({
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics
        }, str(save_path.with_suffix('.meta')))

    def load(self, filepath: str):
        """Load model with label encoder."""
        import joblib
        import xgboost as xgb
        from pathlib import Path

        load_path = Path(filepath)

        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(load_path.with_suffix('.xgb')))

        # Load metadata
        meta = joblib.load(str(load_path.with_suffix('.meta')))
        self.label_encoder = meta['label_encoder']
        self.feature_names = meta['feature_names']
        self.hyperparameters = meta['hyperparameters']
        self.training_metrics = meta['training_metrics']

        self.is_trained = True
