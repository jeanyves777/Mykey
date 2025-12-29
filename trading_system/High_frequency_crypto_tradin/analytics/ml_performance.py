"""
ML Ensemble Performance Analytics Module
=========================================

Comprehensive analysis specifically designed for ML ensemble models:
- Per-model performance breakdown
- Ensemble voting analysis
- Prediction confidence distribution
- Class-wise metrics (BUY/SELL/HOLD)
- Feature importance analysis
- Walk-forward validation metrics
- Real vs Synthetic data quality
- Model weight evolution

This is tailored for the High-Frequency Crypto Trading ML System.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json


@dataclass
class ModelPerformance:
    """Performance metrics for a single model in the ensemble."""
    name: str
    train_accuracy: float
    val_accuracy: float
    train_f1: float
    val_f1: float
    auc: float
    weight: float
    training_time: float  # seconds
    best_iteration: int = 0

    # Per-class metrics
    sell_precision: float = 0.0
    sell_recall: float = 0.0
    sell_f1: float = 0.0
    hold_precision: float = 0.0
    hold_recall: float = 0.0
    hold_f1: float = 0.0
    buy_precision: float = 0.0
    buy_recall: float = 0.0
    buy_f1: float = 0.0

    @property
    def overfit_ratio(self) -> float:
        """Train/Val accuracy ratio - values > 1.1 suggest overfitting."""
        if self.val_accuracy > 0:
            return self.train_accuracy / self.val_accuracy
        return 0.0


@dataclass
class EnsembleAnalysis:
    """Analysis of ensemble voting behavior."""
    total_predictions: int
    unanimous_predictions: int  # All models agree
    majority_predictions: int  # 3+ models agree
    split_predictions: int  # No clear majority

    avg_confidence: float
    high_conf_predictions: int  # confidence > 0.65
    low_conf_predictions: int  # confidence < 0.50

    # Class distribution
    sell_predictions: int
    hold_predictions: int
    buy_predictions: int

    # Actual class distribution
    sell_actual: int
    hold_actual: int
    buy_actual: int


@dataclass
class DataQualityReport:
    """Data quality assessment for training."""
    total_samples: int
    total_features: int

    # Data integrity
    missing_values_pct: float
    zero_volume_pct: float
    outlier_pct: float
    ohlcv_violations: int
    time_gaps: int

    # Feature quality
    constant_features: int
    near_constant_features: int
    high_correlation_pairs: int

    # Target distribution
    sell_pct: float
    hold_pct: float
    buy_pct: float

    # Data source
    is_synthetic: bool
    real_data_days: int
    synthetic_data_days: int


@dataclass
class WalkForwardMetrics:
    """Walk-forward validation results."""
    n_windows: int
    overall_accuracy: float
    overall_f1: float
    accuracy_std: float
    f1_std: float

    # Per-window performance
    window_accuracies: List[float]
    window_f1s: List[float]

    # Degradation analysis
    performance_degradation: float  # Negative = improving over time
    consistency_score: float  # 1 - CV(accuracy)

    # Robustness
    min_accuracy: float
    max_accuracy: float


@dataclass
class MLPerformanceReport:
    """Comprehensive ML ensemble performance report."""

    # Training metadata
    training_date: datetime
    training_duration_minutes: float
    python_version: str
    data_directory: str

    # Model performance
    models: List[ModelPerformance]
    ensemble_accuracy: float
    ensemble_f1: float
    ensemble_auc: float

    # Ensemble analysis
    ensemble_analysis: EnsembleAnalysis

    # Data quality
    data_quality: DataQualityReport

    # Walk-forward validation
    walk_forward: WalkForwardMetrics

    # Confusion matrix
    confusion_matrix: List[List[int]]

    # Feature analysis
    top_features: List[Tuple[str, float]]  # (name, importance)

    # Model weights
    model_weights: Dict[str, float]


class MLPerformanceAnalyzer:
    """
    Analyzes ML ensemble training results and generates comprehensive reports.
    """

    def __init__(self,
                 training_metrics: Dict = None,
                 ensemble = None,
                 feature_engineer = None,
                 validation_results: Dict = None,
                 data_quality_report = None):
        """
        Initialize with training outputs.

        Parameters
        ----------
        training_metrics : Dict
            Metrics from train_ensemble() including per-model results
        ensemble : EnsembleVotingSystem
            Trained ensemble model
        feature_engineer : FeatureEngineer
            Feature engineering object with column info
        validation_results : Dict
            Walk-forward validation results
        data_quality_report : DataQualityReport
            Data quality assessment
        """
        self.training_metrics = training_metrics or {}
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer
        self.validation_results = validation_results or {}
        self.data_quality_report = data_quality_report

    def analyze(self) -> MLPerformanceReport:
        """Perform comprehensive analysis and return report."""

        # Extract model performances
        models = self._extract_model_performances()

        # Analyze ensemble
        ensemble_analysis = self._analyze_ensemble()

        # Walk-forward metrics
        walk_forward = self._extract_walk_forward_metrics()

        # Build report
        report = MLPerformanceReport(
            training_date=datetime.now(),
            training_duration_minutes=self.training_metrics.get('total_time', 0) / 60,
            python_version=self.training_metrics.get('python_version', 'Unknown'),
            data_directory=self.training_metrics.get('data_dir', 'Unknown'),

            models=models,
            ensemble_accuracy=self.training_metrics.get('ensemble', {}).get('accuracy', 0),
            ensemble_f1=self.training_metrics.get('ensemble', {}).get('f1', 0),
            ensemble_auc=self.training_metrics.get('ensemble', {}).get('auc', 0),

            ensemble_analysis=ensemble_analysis,
            data_quality=self.data_quality_report or self._default_data_quality(),
            walk_forward=walk_forward,

            confusion_matrix=self.training_metrics.get('ensemble', {}).get('confusion_matrix', []),
            top_features=self._get_top_features(),
            model_weights=self.ensemble.get_model_weights() if self.ensemble else {},
        )

        return report

    def _extract_model_performances(self) -> List[ModelPerformance]:
        """Extract performance metrics for each model."""
        models = []

        model_names = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'neural_network']

        for name in model_names:
            if name in self.training_metrics:
                m = self.training_metrics[name]

                # Get per-class metrics if available
                per_class = m.get('per_class', {})

                models.append(ModelPerformance(
                    name=name,
                    train_accuracy=m.get('train_accuracy', 0),
                    val_accuracy=m.get('val_accuracy', 0),
                    train_f1=m.get('train_f1', 0),
                    val_f1=m.get('val_f1', 0),
                    auc=m.get('val_auc', m.get('auc', 0)),
                    weight=self.ensemble.get_model_weights().get(name, 0.2) if self.ensemble else 0.2,
                    training_time=m.get('training_time', 0),
                    best_iteration=m.get('best_iteration', 0),

                    sell_precision=per_class.get('SELL', {}).get('precision', 0),
                    sell_recall=per_class.get('SELL', {}).get('recall', 0),
                    sell_f1=per_class.get('SELL', {}).get('f1', 0),
                    hold_precision=per_class.get('HOLD', {}).get('precision', 0),
                    hold_recall=per_class.get('HOLD', {}).get('recall', 0),
                    hold_f1=per_class.get('HOLD', {}).get('f1', 0),
                    buy_precision=per_class.get('BUY', {}).get('precision', 0),
                    buy_recall=per_class.get('BUY', {}).get('recall', 0),
                    buy_f1=per_class.get('BUY', {}).get('f1', 0),
                ))

        return models

    def _analyze_ensemble(self) -> EnsembleAnalysis:
        """Analyze ensemble voting patterns."""
        ensemble_metrics = self.training_metrics.get('ensemble', {})
        pred_dist = ensemble_metrics.get('prediction_distribution', {})

        # Get actual distribution from per_class metrics
        per_class = ensemble_metrics.get('per_class', {})

        return EnsembleAnalysis(
            total_predictions=ensemble_metrics.get('total_predictions', 0),
            unanimous_predictions=0,  # Would need raw predictions
            majority_predictions=0,
            split_predictions=0,

            avg_confidence=ensemble_metrics.get('avg_confidence', 0),
            high_conf_predictions=0,
            low_conf_predictions=0,

            sell_predictions=pred_dist.get('SELL', 0),
            hold_predictions=pred_dist.get('HOLD', 0),
            buy_predictions=pred_dist.get('BUY', 0),

            sell_actual=per_class.get('SELL', {}).get('support', 0),
            hold_actual=per_class.get('HOLD', {}).get('support', 0),
            buy_actual=per_class.get('BUY', {}).get('support', 0),
        )

    def _extract_walk_forward_metrics(self) -> WalkForwardMetrics:
        """Extract walk-forward validation metrics."""
        wf = self.validation_results

        # Get per-window metrics
        window_results = wf.get('window_results', [])
        accuracies = [w.get('accuracy', 0) for w in window_results]
        f1s = [w.get('f1', 0) for w in window_results]

        return WalkForwardMetrics(
            n_windows=wf.get('n_windows', len(window_results)),
            overall_accuracy=wf.get('overall_accuracy', np.mean(accuracies) if accuracies else 0),
            overall_f1=wf.get('overall_f1', np.mean(f1s) if f1s else 0),
            accuracy_std=wf.get('accuracy_std', np.std(accuracies) if accuracies else 0),
            f1_std=np.std(f1s) if f1s else 0,

            window_accuracies=accuracies,
            window_f1s=f1s,

            performance_degradation=wf.get('performance_degradation', 0),
            consistency_score=wf.get('consistency_score', 0),

            min_accuracy=min(accuracies) if accuracies else 0,
            max_accuracy=max(accuracies) if accuracies else 0,
        )

    def _get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if self.ensemble and hasattr(self.ensemble, 'get_feature_importance'):
            importance_df = self.ensemble.get_feature_importance()
            if importance_df is not None and not importance_df.empty:
                top = importance_df.head(n)
                return list(zip(top['feature'], top['importance']))
        return []

    def _default_data_quality(self) -> DataQualityReport:
        """Return default data quality report if none provided."""
        return DataQualityReport(
            total_samples=0,
            total_features=0,
            missing_values_pct=0,
            zero_volume_pct=0,
            outlier_pct=0,
            ohlcv_violations=0,
            time_gaps=0,
            constant_features=0,
            near_constant_features=0,
            high_correlation_pairs=0,
            sell_pct=33.0,
            hold_pct=34.0,
            buy_pct=33.0,
            is_synthetic=False,
            real_data_days=0,
            synthetic_data_days=0,
        )

    def print_report(self, report: MLPerformanceReport) -> str:
        """Generate formatted comprehensive report."""
        lines = []

        # Header
        lines.append("\n" + "=" * 120)
        lines.append("        HIGH-FREQUENCY CRYPTO TRADING - ML ENSEMBLE PERFORMANCE REPORT")
        lines.append("=" * 120)
        lines.append(f"  Training Date:      {report.training_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Training Duration:  {report.training_duration_minutes:.1f} minutes")
        lines.append(f"  Data Directory:     {report.data_directory}")

        # Data Quality Section
        dq = report.data_quality
        lines.append(f"\n{'DATA QUALITY ASSESSMENT':=^120}")
        lines.append(f"  Total Samples:      {dq.total_samples:,}")
        lines.append(f"  Total Features:     {dq.total_features}")
        lines.append(f"  ")
        lines.append(f"  Missing Values:     {dq.missing_values_pct:.2f}%")
        lines.append(f"  Zero Volume:        {dq.zero_volume_pct:.2f}%")
        lines.append(f"  Outliers:           {dq.outlier_pct:.3f}%")
        lines.append(f"  OHLCV Violations:   {dq.ohlcv_violations}")
        lines.append(f"  Time Gaps:          {dq.time_gaps}")
        lines.append(f"  ")
        lines.append(f"  Constant Features:  {dq.constant_features}")
        lines.append(f"  High Correlations:  {dq.high_correlation_pairs}")
        lines.append(f"  ")
        lines.append(f"  Target Distribution:")
        lines.append(f"    SELL: {dq.sell_pct:.1f}%  |  HOLD: {dq.hold_pct:.1f}%  |  BUY: {dq.buy_pct:.1f}%")

        if dq.is_synthetic:
            lines.append(f"  ")
            lines.append(f"  Data Source: SYNTHETIC + REAL")
            lines.append(f"    Real Data:      {dq.real_data_days} days")
            lines.append(f"    Synthetic Data: {dq.synthetic_data_days} days")

        # Ensemble Performance Summary
        lines.append(f"\n{'ENSEMBLE PERFORMANCE SUMMARY':=^120}")
        lines.append(f"  Overall Accuracy:   {report.ensemble_accuracy:.4f} ({report.ensemble_accuracy*100:.2f}%)")
        lines.append(f"  Overall F1 Score:   {report.ensemble_f1:.4f}")
        lines.append(f"  Overall AUC:        {report.ensemble_auc:.4f}")

        # Per-Model Performance Table
        lines.append(f"\n{'INDIVIDUAL MODEL PERFORMANCE':=^120}")
        lines.append("-" * 120)
        lines.append(f"  {'Model':<15} | {'Train Acc':>10} | {'Val Acc':>10} | {'Val F1':>10} | {'AUC':>8} | {'Weight':>8} | {'Time':>8} | {'Overfit':>8}")
        lines.append("-" * 120)

        for m in report.models:
            overfit_flag = "!!!" if m.overfit_ratio > 1.15 else ("!" if m.overfit_ratio > 1.1 else "")
            lines.append(
                f"  {m.name:<15} | {m.train_accuracy:>10.4f} | {m.val_accuracy:>10.4f} | "
                f"{m.val_f1:>10.4f} | {m.auc:>8.4f} | {m.weight:>8.2%} | {m.training_time:>7.1f}s | "
                f"{m.overfit_ratio:>6.2f}x {overfit_flag}"
            )
        lines.append("-" * 120)

        # Per-Class Performance
        lines.append(f"\n{'PER-CLASS METRICS (BY MODEL)':=^120}")
        lines.append("-" * 120)
        lines.append(f"  {'Model':<15} | {'SELL Prec':>10} {'SELL Rec':>10} {'SELL F1':>10} | {'HOLD Prec':>10} {'HOLD Rec':>10} {'HOLD F1':>10} | {'BUY Prec':>10} {'BUY Rec':>10} {'BUY F1':>10}")
        lines.append("-" * 120)

        for m in report.models:
            lines.append(
                f"  {m.name:<15} | {m.sell_precision:>10.4f} {m.sell_recall:>10.4f} {m.sell_f1:>10.4f} | "
                f"{m.hold_precision:>10.4f} {m.hold_recall:>10.4f} {m.hold_f1:>10.4f} | "
                f"{m.buy_precision:>10.4f} {m.buy_recall:>10.4f} {m.buy_f1:>10.4f}"
            )
        lines.append("-" * 120)

        # Ensemble Analysis
        ea = report.ensemble_analysis
        lines.append(f"\n{'ENSEMBLE VOTING ANALYSIS':=^120}")
        lines.append(f"  Prediction Distribution:")
        total_preds = ea.sell_predictions + ea.hold_predictions + ea.buy_predictions
        if total_preds > 0:
            lines.append(f"    SELL: {ea.sell_predictions:>8,} ({ea.sell_predictions/total_preds*100:>5.1f}%)  |  "
                        f"HOLD: {ea.hold_predictions:>8,} ({ea.hold_predictions/total_preds*100:>5.1f}%)  |  "
                        f"BUY: {ea.buy_predictions:>8,} ({ea.buy_predictions/total_preds*100:>5.1f}%)")

        lines.append(f"  ")
        lines.append(f"  Actual Distribution:")
        total_actual = ea.sell_actual + ea.hold_actual + ea.buy_actual
        if total_actual > 0:
            lines.append(f"    SELL: {ea.sell_actual:>8,} ({ea.sell_actual/total_actual*100:>5.1f}%)  |  "
                        f"HOLD: {ea.hold_actual:>8,} ({ea.hold_actual/total_actual*100:>5.1f}%)  |  "
                        f"BUY: {ea.buy_actual:>8,} ({ea.buy_actual/total_actual*100:>5.1f}%)")

        # Confusion Matrix
        if report.confusion_matrix:
            lines.append(f"\n{'CONFUSION MATRIX':=^120}")
            cm = report.confusion_matrix
            n_classes = len(cm)
            labels = ['SELL', 'HOLD', 'BUY'] if n_classes == 3 else [f'C{i}' for i in range(n_classes)]

            lines.append("               Predicted")
            lines.append("           " + "  ".join([f"{l:>8}" for l in labels]))
            for i, row in enumerate(cm):
                actual_label = labels[i] if i < len(labels) else f'C{i}'
                lines.append(f"  Actual {actual_label}: " + "  ".join([f"{v:>8,}" for v in row]))

        # Walk-Forward Validation
        wf = report.walk_forward
        lines.append(f"\n{'WALK-FORWARD VALIDATION':=^120}")
        lines.append(f"  Windows:            {wf.n_windows}")
        lines.append(f"  Overall Accuracy:   {wf.overall_accuracy:.4f} +/- {wf.accuracy_std:.4f}")
        lines.append(f"  Overall F1:         {wf.overall_f1:.4f} +/- {wf.f1_std:.4f}")
        lines.append(f"  Consistency Score:  {wf.consistency_score:.4f}")
        lines.append(f"  Degradation:        {wf.performance_degradation:+.6f}")

        if wf.window_accuracies:
            lines.append(f"  ")
            lines.append(f"  Per-Window Accuracy: {' -> '.join([f'{a:.3f}' for a in wf.window_accuracies])}")

        # Top Features
        if report.top_features:
            lines.append(f"\n{'TOP 20 FEATURES BY IMPORTANCE':=^120}")
            lines.append("-" * 60)
            for i, (feat, imp) in enumerate(report.top_features[:20], 1):
                bar = "|" * int(imp * 50)
                lines.append(f"  {i:>2}. {feat:<40} {imp:.4f} {bar}")
            lines.append("-" * 60)

        # Model Weights
        if report.model_weights:
            lines.append(f"\n{'ENSEMBLE MODEL WEIGHTS':=^120}")
            for name, weight in sorted(report.model_weights.items(), key=lambda x: -x[1]):
                bar = "|" * int(weight * 50)
                lines.append(f"  {name:<20} {weight:.4f} ({weight*100:.1f}%) {bar}")

        # Recommendations
        lines.append(f"\n{'RECOMMENDATIONS':=^120}")

        # Check for issues
        issues = []
        recommendations = []

        # Overfitting check
        overfit_models = [m for m in report.models if m.overfit_ratio > 1.15]
        if overfit_models:
            issues.append(f"  [!] OVERFITTING DETECTED in: {', '.join([m.name for m in overfit_models])}")
            recommendations.append("  --> Increase regularization or reduce model complexity")

        # Class imbalance in predictions
        if ea.sell_predictions == 0 or ea.buy_predictions == 0:
            issues.append(f"  [!] PREDICTION BIAS: Model not predicting all classes")
            recommendations.append("  --> Adjust classification thresholds or use class weights")

        # Low accuracy
        if report.ensemble_accuracy < 0.35:
            issues.append(f"  [!] LOW ACCURACY: {report.ensemble_accuracy:.2%} (expected > 35% for 3-class)")
            recommendations.append("  --> Consider feature engineering or hyperparameter tuning")

        # Data quality issues
        if dq.zero_volume_pct > 1:
            issues.append(f"  [!] HIGH ZERO VOLUME: {dq.zero_volume_pct:.1f}%")
            recommendations.append("  --> Verify data source quality")

        if issues:
            for issue in issues:
                lines.append(issue)
            lines.append("")
            for rec in recommendations:
                lines.append(rec)
        else:
            lines.append("  [OK] No critical issues detected")
            lines.append("  [OK] Model training appears healthy")

        # Footer
        lines.append("\n" + "=" * 120)
        lines.append("                    END OF ML ENSEMBLE PERFORMANCE REPORT")
        lines.append("=" * 120 + "\n")

        return "\n".join(lines)

    def save_report(self, report: MLPerformanceReport, filepath: str):
        """Save report to JSON file."""
        data = {
            'training_date': report.training_date.isoformat(),
            'training_duration_minutes': report.training_duration_minutes,
            'ensemble_accuracy': report.ensemble_accuracy,
            'ensemble_f1': report.ensemble_f1,
            'ensemble_auc': report.ensemble_auc,
            'models': [
                {
                    'name': m.name,
                    'train_accuracy': m.train_accuracy,
                    'val_accuracy': m.val_accuracy,
                    'val_f1': m.val_f1,
                    'auc': m.auc,
                    'weight': m.weight,
                    'overfit_ratio': m.overfit_ratio,
                }
                for m in report.models
            ],
            'walk_forward': {
                'overall_accuracy': report.walk_forward.overall_accuracy,
                'overall_f1': report.walk_forward.overall_f1,
                'consistency_score': report.walk_forward.consistency_score,
            },
            'data_quality': {
                'total_samples': report.data_quality.total_samples,
                'zero_volume_pct': report.data_quality.zero_volume_pct,
                'is_synthetic': report.data_quality.is_synthetic,
            },
            'top_features': report.top_features[:10],
            'model_weights': report.model_weights,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def analyze_training_results(training_log_file: str = None) -> MLPerformanceReport:
    """
    Convenience function to analyze training results from log file.

    Parameters
    ----------
    training_log_file : str
        Path to training log file

    Returns
    -------
    MLPerformanceReport
        Analyzed performance report
    """
    # Parse log file and extract metrics
    # This is a placeholder - actual implementation would parse the log

    analyzer = MLPerformanceAnalyzer()
    return analyzer.analyze()


if __name__ == "__main__":
    # Demo with sample data
    sample_metrics = {
        'random_forest': {
            'train_accuracy': 0.4546,
            'val_accuracy': 0.3822,
            'val_f1': 0.3770,
            'val_auc': 0.5591,
            'training_time': 110.6,
        },
        'xgboost': {
            'train_accuracy': 0.4305,
            'val_accuracy': 0.3926,
            'val_f1': 0.3816,
            'val_auc': 0.5946,
            'training_time': 40.0,
            'best_iteration': 36,
        },
        'ensemble': {
            'accuracy': 0.39,
            'f1': 0.38,
            'auc': 0.59,
            'prediction_distribution': {'SELL': 0, 'HOLD': 35, 'BUY': 65},
            'per_class': {
                'SELL': {'support': 33, 'precision': 0, 'recall': 0, 'f1': 0},
                'HOLD': {'support': 34, 'precision': 0.17, 'recall': 0.15, 'f1': 0.16},
                'BUY': {'support': 33, 'precision': 0.32, 'recall': 0.38, 'f1': 0.35},
            }
        }
    }

    sample_dq = DataQualityReport(
        total_samples=250000,
        total_features=230,
        missing_values_pct=0.0,
        zero_volume_pct=0.0,
        outlier_pct=0.09,
        ohlcv_violations=199,
        time_gaps=0,
        constant_features=12,
        near_constant_features=60,
        high_correlation_pairs=88,
        sell_pct=33.0,
        hold_pct=34.0,
        buy_pct=33.0,
        is_synthetic=True,
        real_data_days=7,
        synthetic_data_days=173,
    )

    sample_wf = {
        'n_windows': 5,
        'overall_accuracy': 0.2121,
        'overall_f1': 0.2148,
        'accuracy_std': 0.037,
        'consistency_score': 0.8254,
        'performance_degradation': -0.00035,
        'window_results': [
            {'accuracy': 0.243, 'f1': 0.186},
            {'accuracy': 0.198, 'f1': 0.200},
            {'accuracy': 0.152, 'f1': 0.174},
            {'accuracy': 0.257, 'f1': 0.233},
            {'accuracy': 0.212, 'f1': 0.196},
        ]
    }

    analyzer = MLPerformanceAnalyzer(
        training_metrics=sample_metrics,
        validation_results=sample_wf,
        data_quality_report=sample_dq,
    )

    report = analyzer.analyze()
    print(analyzer.print_report(report))
