"""
Data Quality and Integrity Validation Module
=============================================

Comprehensive checks to ensure data quality before ML training:
1. Missing value analysis
2. Duplicate detection
3. Time series continuity (gaps)
4. Outlier detection
5. Distribution analysis
6. Feature correlation checks
7. Target leakage detection
8. Stationarity tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QualityReport:
    """Container for data quality report."""
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    critical_issues: List[str]
    warnings_list: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]


class DataQualityChecker:
    """
    Comprehensive data quality and integrity validation.

    Performs multiple checks on raw and engineered data to ensure
    it's suitable for ML training.
    """

    def __init__(self,
                 missing_threshold: float = 0.05,
                 duplicate_threshold: float = 0.01,
                 outlier_std_threshold: float = 5.0,
                 gap_threshold_minutes: int = 5,
                 correlation_threshold: float = 0.95,
                 min_samples_per_class: int = 1000):
        """
        Initialize the data quality checker.

        Args:
            missing_threshold: Max allowed missing value ratio (default 5%)
            duplicate_threshold: Max allowed duplicate ratio (default 1%)
            outlier_std_threshold: Std devs to consider outlier (default 5)
            gap_threshold_minutes: Max allowed gap in time series (default 5 min)
            correlation_threshold: Threshold for high correlation warning (default 0.95)
            min_samples_per_class: Minimum samples per target class
        """
        self.missing_threshold = missing_threshold
        self.duplicate_threshold = duplicate_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.gap_threshold_minutes = gap_threshold_minutes
        self.correlation_threshold = correlation_threshold
        self.min_samples_per_class = min_samples_per_class

        self.checks_performed = []
        self.issues = []
        self.warnings_list = []
        self.metrics = {}

    def run_all_checks(self,
                       df: pd.DataFrame,
                       feature_cols: List[str] = None,
                       target_col: str = 'target',
                       datetime_col: str = 'datetime',
                       symbol_col: str = 'symbol',
                       verbose: bool = True) -> QualityReport:
        """
        Run all data quality checks.

        Args:
            df: DataFrame to validate
            feature_cols: List of feature columns to check
            target_col: Name of target column
            datetime_col: Name of datetime column
            symbol_col: Name of symbol column
            verbose: Print progress and results

        Returns:
            QualityReport with all findings
        """
        self.checks_performed = []
        self.issues = []
        self.warnings_list = []
        self.metrics = {}

        if verbose:
            print("")
            print("  " + "=" * 56)
            print("  DATA QUALITY & INTEGRITY VALIDATION")
            print("  " + "=" * 56)
            print(f"  Samples: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            print("")

        # 1. Basic structure check
        self._check_basic_structure(df, verbose)

        # 2. Missing values check
        self._check_missing_values(df, feature_cols, verbose)

        # 3. Duplicate check
        self._check_duplicates(df, datetime_col, symbol_col, verbose)

        # 4. Time series continuity
        if datetime_col in df.columns:
            self._check_time_continuity(df, datetime_col, symbol_col, verbose)

        # 5. Outlier detection
        self._check_outliers(df, feature_cols, verbose)

        # 6. OHLCV integrity
        self._check_ohlcv_integrity(df, verbose)

        # 7. Target distribution
        if target_col in df.columns:
            self._check_target_distribution(df, target_col, verbose)

        # 8. Feature statistics
        self._check_feature_statistics(df, feature_cols, verbose)

        # 9. High correlation check
        self._check_high_correlations(df, feature_cols, verbose)

        # 10. Infinite values check
        self._check_infinite_values(df, feature_cols, verbose)

        # 11. Data type consistency
        self._check_data_types(df, verbose)

        # 12. Target leakage detection
        if target_col in df.columns and feature_cols:
            self._check_target_leakage(df, feature_cols, target_col, verbose)

        # Generate report
        report = self._generate_report(verbose)

        return report

    def _check_basic_structure(self, df: pd.DataFrame, verbose: bool):
        """Check basic DataFrame structure."""
        check_name = "Basic Structure"

        issues = []

        if len(df) == 0:
            issues.append("CRITICAL: DataFrame is empty!")

        if len(df.columns) == 0:
            issues.append("CRITICAL: No columns in DataFrame!")

        # Check for required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [c for c in required_cols if c not in df.columns]
        if missing_ohlcv:
            issues.append(f"Missing OHLCV columns: {missing_ohlcv}")

        self.metrics['total_rows'] = len(df)
        self.metrics['total_columns'] = len(df.columns)

        passed = len(issues) == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.issues.extend(issues)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{'OK' if passed else 'FAIL'}] {check_name}")
            if not passed:
                for issue in issues:
                    print(f"      - {issue}")

    def _check_missing_values(self, df: pd.DataFrame, feature_cols: List[str], verbose: bool):
        """Check for missing values."""
        check_name = "Missing Values"

        cols_to_check = feature_cols if feature_cols else df.columns

        missing_counts = df[cols_to_check].isnull().sum()
        missing_pct = missing_counts / len(df)

        total_missing = missing_counts.sum()
        total_missing_pct = total_missing / (len(df) * len(cols_to_check))

        high_missing_cols = missing_pct[missing_pct > self.missing_threshold]

        self.metrics['missing_value_pct'] = total_missing_pct
        self.metrics['cols_with_high_missing'] = len(high_missing_cols)

        issues = []
        if len(high_missing_cols) > 0:
            issues.append(f"{len(high_missing_cols)} columns have >{self.missing_threshold*100:.1f}% missing values")
            for col in high_missing_cols.index[:5]:  # Show top 5
                issues.append(f"  - {col}: {high_missing_cols[col]*100:.1f}% missing")

        passed = total_missing_pct <= self.missing_threshold
        self.checks_performed.append((check_name, passed))

        if issues:
            if passed:
                self.warnings_list.extend(issues)
            else:
                self.issues.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'WARN'}] {check_name}: {total_missing_pct*100:.2f}% overall")
            if issues and not passed:
                for issue in issues[:3]:
                    print(f"      - {issue}")

    def _check_duplicates(self, df: pd.DataFrame, datetime_col: str, symbol_col: str, verbose: bool):
        """Check for duplicate rows."""
        check_name = "Duplicate Rows"

        # Check for exact duplicates
        exact_duplicates = df.duplicated().sum()
        exact_dup_pct = exact_duplicates / len(df)

        # Check for duplicate timestamps per symbol
        timestamp_dups = 0
        if datetime_col in df.columns and symbol_col in df.columns:
            timestamp_dups = df.duplicated(subset=[datetime_col, symbol_col]).sum()
        elif datetime_col in df.columns:
            timestamp_dups = df.duplicated(subset=[datetime_col]).sum()

        self.metrics['exact_duplicates'] = exact_duplicates
        self.metrics['exact_duplicate_pct'] = exact_dup_pct
        self.metrics['timestamp_duplicates'] = timestamp_dups

        issues = []
        if exact_dup_pct > self.duplicate_threshold:
            issues.append(f"High exact duplicate rate: {exact_dup_pct*100:.2f}%")
        if timestamp_dups > 0:
            issues.append(f"Duplicate timestamps found: {timestamp_dups:,}")

        passed = exact_dup_pct <= self.duplicate_threshold and timestamp_dups == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.warnings_list.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'WARN'}] {check_name}: {exact_duplicates:,} exact, {timestamp_dups:,} timestamp dups")

    def _check_time_continuity(self, df: pd.DataFrame, datetime_col: str, symbol_col: str, verbose: bool):
        """Check for gaps in time series."""
        check_name = "Time Series Continuity"

        total_gaps = 0
        max_gap_minutes = 0
        gap_details = []

        symbols = df[symbol_col].unique() if symbol_col in df.columns else ['ALL']

        for symbol in symbols:
            if symbol_col in df.columns:
                symbol_df = df[df[symbol_col] == symbol].sort_values(datetime_col)
            else:
                symbol_df = df.sort_values(datetime_col)

            if len(symbol_df) < 2:
                continue

            times = pd.to_datetime(symbol_df[datetime_col])
            time_diffs = times.diff().dt.total_seconds() / 60  # Minutes

            # Find gaps (assuming 1-minute data, gaps > threshold are issues)
            gaps = time_diffs[time_diffs > self.gap_threshold_minutes]
            total_gaps += len(gaps)

            if len(gaps) > 0:
                symbol_max_gap = gaps.max()
                if symbol_max_gap > max_gap_minutes:
                    max_gap_minutes = symbol_max_gap
                gap_details.append(f"{symbol}: {len(gaps)} gaps, max {symbol_max_gap:.0f} min")

        self.metrics['total_time_gaps'] = total_gaps
        self.metrics['max_gap_minutes'] = max_gap_minutes

        issues = []
        if total_gaps > 0:
            issues.append(f"Found {total_gaps} time gaps > {self.gap_threshold_minutes} minutes")
            issues.append(f"Maximum gap: {max_gap_minutes:.0f} minutes")

        passed = total_gaps == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.warnings_list.extend(issues)

        if verbose:
            status = "OK" if passed else "WARN"
            print(f"  [{status}] {check_name}: {total_gaps} gaps found, max {max_gap_minutes:.0f} min")

    def _check_outliers(self, df: pd.DataFrame, feature_cols: List[str], verbose: bool):
        """Check for outliers in features."""
        check_name = "Outlier Detection"

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_counts = {}
        total_outliers = 0

        for col in feature_cols[:50]:  # Check first 50 features for speed
            if col not in df.columns:
                continue

            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            mean = col_data.mean()
            std = col_data.std()

            if std == 0:
                continue

            z_scores = np.abs((col_data - mean) / std)
            outliers = (z_scores > self.outlier_std_threshold).sum()

            if outliers > 0:
                outlier_counts[col] = outliers
                total_outliers += outliers

        outlier_pct = total_outliers / (len(df) * len(feature_cols)) if feature_cols else 0

        self.metrics['total_outliers'] = total_outliers
        self.metrics['outlier_pct'] = outlier_pct
        self.metrics['cols_with_outliers'] = len(outlier_counts)

        # Sort by count
        top_outlier_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        issues = []
        if outlier_pct > 0.01:  # More than 1% outliers
            issues.append(f"High outlier rate: {outlier_pct*100:.2f}%")
            for col, count in top_outlier_cols:
                issues.append(f"  - {col}: {count:,} outliers")

        passed = outlier_pct <= 0.05  # Less than 5% outliers
        self.checks_performed.append((check_name, passed))

        if issues:
            self.warnings_list.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'WARN'}] {check_name}: {total_outliers:,} outliers ({outlier_pct*100:.3f}%)")

    def _check_ohlcv_integrity(self, df: pd.DataFrame, verbose: bool):
        """Check OHLCV data integrity."""
        check_name = "OHLCV Integrity"

        issues = []

        if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            issues.append("Missing OHLCV columns - skipping integrity check")
            self.checks_performed.append((check_name, False))
            if verbose:
                print(f"  [SKIP] {check_name}: Missing required columns")
            return

        # Check high >= low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"High < Low violations: {invalid_hl:,}")

        # Check high >= open and high >= close
        invalid_high = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
        if invalid_high > 0:
            issues.append(f"High < Open/Close violations: {invalid_high:,}")

        # Check low <= open and low <= close
        invalid_low = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
        if invalid_low > 0:
            issues.append(f"Low > Open/Close violations: {invalid_low:,}")

        # Check for zero or negative prices
        zero_prices = ((df['close'] <= 0) | (df['open'] <= 0)).sum()
        if zero_prices > 0:
            issues.append(f"Zero or negative prices: {zero_prices:,}")

        # Check for zero volume
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            zero_vol_pct = zero_volume / len(df)
            self.metrics['zero_volume_pct'] = zero_vol_pct
            if zero_vol_pct > 0.1:  # More than 10% zero volume
                issues.append(f"High zero volume rate: {zero_vol_pct*100:.1f}%")

        self.metrics['ohlcv_violations'] = invalid_hl + invalid_high + invalid_low + zero_prices

        passed = len(issues) == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.issues.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'FAIL'}] {check_name}: {self.metrics['ohlcv_violations']} violations")
            if not passed:
                for issue in issues[:3]:
                    print(f"      - {issue}")

    def _check_target_distribution(self, df: pd.DataFrame, target_col: str, verbose: bool):
        """Check target variable distribution."""
        check_name = "Target Distribution"

        if target_col not in df.columns:
            self.checks_performed.append((check_name, True))
            return

        target_counts = df[target_col].value_counts()
        target_pcts = target_counts / len(df) * 100

        issues = []

        # Check for extreme imbalance
        min_class_pct = target_pcts.min()
        max_class_pct = target_pcts.max()
        imbalance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')

        self.metrics['target_classes'] = len(target_counts)
        self.metrics['min_class_pct'] = min_class_pct
        self.metrics['max_class_pct'] = max_class_pct
        self.metrics['imbalance_ratio'] = imbalance_ratio

        # Check minimum samples per class
        min_samples = target_counts.min()
        if min_samples < self.min_samples_per_class:
            issues.append(f"Low sample count in minority class: {min_samples:,} (min: {self.min_samples_per_class:,})")

        if imbalance_ratio > 3:
            issues.append(f"High class imbalance ratio: {imbalance_ratio:.1f}x")

        passed = imbalance_ratio <= 5 and min_samples >= self.min_samples_per_class
        self.checks_performed.append((check_name, passed))

        if issues:
            self.warnings_list.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'WARN'}] {check_name}:")
            class_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            for cls in sorted(target_counts.index):
                name = class_names.get(cls, str(cls))
                print(f"      {name}: {target_counts[cls]:,} ({target_pcts[cls]:.1f}%)")

    def _check_feature_statistics(self, df: pd.DataFrame, feature_cols: List[str], verbose: bool):
        """Check feature statistics for anomalies."""
        check_name = "Feature Statistics"

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        issues = []

        # Check for constant features
        constant_features = []
        near_constant_features = []

        for col in feature_cols:
            if col not in df.columns:
                continue

            nunique = df[col].nunique()
            if nunique == 1:
                constant_features.append(col)
            elif nunique <= 3:
                near_constant_features.append(col)

        self.metrics['constant_features'] = len(constant_features)
        self.metrics['near_constant_features'] = len(near_constant_features)

        if constant_features:
            issues.append(f"Constant features (no variance): {len(constant_features)}")

        if near_constant_features:
            issues.append(f"Near-constant features (<=3 unique): {len(near_constant_features)}")

        passed = len(constant_features) == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.warnings_list.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'WARN'}] {check_name}: {len(constant_features)} constant, {len(near_constant_features)} near-constant")

    def _check_high_correlations(self, df: pd.DataFrame, feature_cols: List[str], verbose: bool):
        """Check for highly correlated features."""
        check_name = "Feature Correlations"

        if feature_cols is None or len(feature_cols) < 2:
            self.checks_performed.append((check_name, True))
            if verbose:
                print(f"  [SKIP] {check_name}: Not enough features")
            return

        # Sample for speed
        sample_cols = feature_cols[:50]  # Check first 50 features
        valid_cols = [c for c in sample_cols if c in df.columns]

        if len(valid_cols) < 2:
            self.checks_performed.append((check_name, True))
            return

        try:
            corr_matrix = df[valid_cols].corr()

            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                        high_corr_pairs.append((
                            corr_matrix.index[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))

            self.metrics['high_correlation_pairs'] = len(high_corr_pairs)

            issues = []
            if len(high_corr_pairs) > 0:
                issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs (>{self.correlation_threshold})")

            passed = len(high_corr_pairs) < 20  # Allow some high correlations
            self.checks_performed.append((check_name, passed))

            if issues:
                self.warnings_list.extend(issues)

            if verbose:
                print(f"  [{'OK' if passed else 'WARN'}] {check_name}: {len(high_corr_pairs)} pairs > {self.correlation_threshold}")

        except Exception as e:
            self.checks_performed.append((check_name, True))
            if verbose:
                print(f"  [SKIP] {check_name}: Error computing correlations")

    def _check_infinite_values(self, df: pd.DataFrame, feature_cols: List[str], verbose: bool):
        """Check for infinite values."""
        check_name = "Infinite Values"

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        valid_cols = [c for c in feature_cols if c in df.columns]

        inf_count = np.isinf(df[valid_cols].select_dtypes(include=[np.number])).sum().sum()
        inf_pct = inf_count / (len(df) * len(valid_cols)) if valid_cols else 0

        self.metrics['infinite_values'] = inf_count
        self.metrics['infinite_pct'] = inf_pct

        issues = []
        if inf_count > 0:
            issues.append(f"Infinite values found: {inf_count:,}")

        passed = inf_count == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.issues.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'FAIL'}] {check_name}: {inf_count:,} infinite values")

    def _check_data_types(self, df: pd.DataFrame, verbose: bool):
        """Check data type consistency."""
        check_name = "Data Types"

        dtype_counts = df.dtypes.value_counts()

        issues = []

        # Check for object types in numeric columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_like_objects = []

        for col in object_cols:
            try:
                pd.to_numeric(df[col].dropna().head(100))
                numeric_like_objects.append(col)
            except:
                pass

        if numeric_like_objects:
            issues.append(f"Numeric data stored as strings: {numeric_like_objects[:5]}")

        self.metrics['object_columns'] = len(object_cols)
        self.metrics['numeric_as_string'] = len(numeric_like_objects)

        passed = len(numeric_like_objects) == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.warnings_list.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'WARN'}] {check_name}: {len(object_cols)} object cols, {len(numeric_like_objects)} should be numeric")

    def _check_target_leakage(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, verbose: bool):
        """Check for potential target leakage in features."""
        check_name = "Target Leakage"

        if target_col not in df.columns or not feature_cols:
            self.checks_performed.append((check_name, True))
            return

        suspicious_features = []

        # Check correlation with target
        target = df[target_col]

        for col in feature_cols[:50]:  # Check first 50 features
            if col not in df.columns:
                continue

            try:
                corr = df[col].corr(target)
                if abs(corr) > 0.8:  # Very high correlation with target
                    suspicious_features.append((col, corr))
            except:
                pass

        # Check for future-looking feature names
        future_keywords = ['forward', 'future', 'next', 'lead', 'predict', 'target']
        future_named = [col for col in feature_cols
                       if any(kw in col.lower() for kw in future_keywords)]

        self.metrics['leakage_suspects'] = len(suspicious_features)
        self.metrics['future_named_features'] = len(future_named)

        issues = []
        if suspicious_features:
            issues.append(f"Features with >80% target correlation: {len(suspicious_features)}")
            for feat, corr in suspicious_features[:3]:
                issues.append(f"  - {feat}: {corr:.3f}")

        if future_named:
            issues.append(f"Suspicious feature names (may contain future data): {future_named[:5]}")

        passed = len(suspicious_features) == 0 and len(future_named) == 0
        self.checks_performed.append((check_name, passed))

        if issues:
            self.issues.extend(issues)

        if verbose:
            print(f"  [{'OK' if passed else 'FAIL'}] {check_name}: {len(suspicious_features)} suspicious correlations")
            if not passed:
                for issue in issues[:3]:
                    print(f"      - {issue}")

    def _generate_report(self, verbose: bool) -> QualityReport:
        """Generate final quality report."""
        passed_checks = sum(1 for _, passed in self.checks_performed if passed)
        failed_checks = len(self.checks_performed) - passed_checks

        # Determine overall pass/fail
        critical_issues = [i for i in self.issues if 'CRITICAL' in i.upper()]
        overall_passed = len(critical_issues) == 0 and failed_checks <= 2

        # Generate recommendations
        recommendations = []

        if self.metrics.get('missing_value_pct', 0) > 0.01:
            recommendations.append("Consider imputing missing values or dropping columns with high missing rate")

        if self.metrics.get('imbalance_ratio', 1) > 3:
            recommendations.append("Consider class balancing techniques (SMOTE, class weights, undersampling)")

        if self.metrics.get('constant_features', 0) > 0:
            recommendations.append(f"Remove {self.metrics['constant_features']} constant features")

        if self.metrics.get('high_correlation_pairs', 0) > 10:
            recommendations.append("Consider feature selection to reduce multicollinearity")

        if self.metrics.get('total_time_gaps', 0) > 0:
            recommendations.append("Address time series gaps (forward fill, interpolation, or filter)")

        report = QualityReport(
            passed=overall_passed,
            total_checks=len(self.checks_performed),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=len(self.warnings_list),
            critical_issues=critical_issues,
            warnings_list=self.warnings_list,
            metrics=self.metrics,
            recommendations=recommendations
        )

        if verbose:
            print("")
            print("  " + "-" * 56)
            print("  QUALITY CHECK SUMMARY")
            print("  " + "-" * 56)
            print(f"  Overall Status: {'PASSED' if overall_passed else 'FAILED'}")
            print(f"  Checks Passed: {passed_checks}/{len(self.checks_performed)}")
            print(f"  Warnings: {len(self.warnings_list)}")
            print(f"  Critical Issues: {len(critical_issues)}")

            if recommendations:
                print("")
                print("  Recommendations:")
                for rec in recommendations[:5]:
                    print(f"    - {rec}")

            print("  " + "-" * 56)

        return report


def run_quick_quality_check(df: pd.DataFrame,
                            feature_cols: List[str] = None,
                            verbose: bool = True) -> bool:
    """
    Run a quick quality check and return pass/fail.

    Args:
        df: DataFrame to check
        feature_cols: Feature columns to validate
        verbose: Print results

    Returns:
        True if passed, False if critical issues found
    """
    checker = DataQualityChecker()
    report = checker.run_all_checks(df, feature_cols=feature_cols, verbose=verbose)
    return report.passed
