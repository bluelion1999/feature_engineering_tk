"""
Core infrastructure for TargetAnalyzer.

This module contains the base class with initialization, task detection,
and distribution analysis methods.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from scipy import stats

from ..base import FeatureEngineeringBase

# Configure logging
logger = logging.getLogger(__name__)


class TargetAnalyzerCore(FeatureEngineeringBase):
    """
    Core infrastructure for target-aware analysis.

    Provides initialization, task detection, and distribution analysis
    for both classification and regression tasks.
    """

    # Class constants for thresholds
    NONLINEAR_IMPROVEMENT_THRESHOLD = 1.2  # Polynomial must be 20% better than linear

    def __init__(self, df: pd.DataFrame, target_column: str, task: str = 'auto'):
        """
        Initialize TargetAnalyzer with a dataframe and target column.

        Args:
            df: Input pandas DataFrame
            target_column: Name of the target variable column
            task: Task type ('classification', 'regression', or 'auto').
                  Default 'auto' will detect based on target characteristics.

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If target_column not in dataframe or invalid task specified
        """
        super().__init__(df)

        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        if task not in ['auto', 'classification', 'regression']:
            raise ValueError("task must be 'auto', 'classification', or 'regression'")

        self.target_column = target_column
        self._analysis_cache: Dict[str, Any] = {}

        # Detect or set task type
        if task == 'auto':
            self.task = self._detect_task()
            logger.info(f"Auto-detected task type: {self.task}")
        else:
            self.task = task
            logger.info(f"Task type set to: {self.task}")

    def _detect_task(self) -> str:
        """
        Automatically detect whether this is a classification or regression task.

        Returns:
            str: 'classification' or 'regression'
        """
        target = self.df[self.target_column].dropna()

        if len(target) == 0:
            logger.warning("Target column is empty, defaulting to classification")
            return 'classification'

        # Check if numeric
        if pd.api.types.is_numeric_dtype(target):
            unique_count = target.nunique()
            unique_ratio = unique_count / len(target)

            # Heuristics for classification vs regression
            if unique_count == 2:
                return 'classification'
            elif unique_count <= 20 or unique_ratio < 0.05:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get information about the detected/specified task.

        Returns:
            Dict containing task type, target column info, and class information
            for classification tasks
        """
        target = self.df[self.target_column]

        info = {
            'task': self.task,
            'target_column': self.target_column,
            'target_dtype': str(target.dtype),
            'unique_values': target.nunique(),
            'missing_count': target.isnull().sum(),
            'missing_percent': (target.isnull().sum() / len(target) * 100)
        }

        if self.task == 'classification':
            info['classes'] = sorted(target.dropna().unique().tolist())
            info['class_count'] = len(info['classes'])

        return info

    def analyze_class_distribution(self) -> pd.DataFrame:
        """
        Analyze class distribution for classification tasks.

        Returns:
            DataFrame with columns: class, count, percentage, imbalance_ratio
            Empty DataFrame if task is not classification
        """
        if self.task != 'classification':
            logger.warning("analyze_class_distribution() is only available for classification tasks")
            return pd.DataFrame()

        if 'class_distribution' in self._analysis_cache:
            return self._analysis_cache['class_distribution']

        target = self.df[self.target_column].dropna()
        value_counts = target.value_counts()

        distribution = pd.DataFrame({
            'class': value_counts.index,
            'count': value_counts.values,
            'percentage': (value_counts.values / len(target) * 100)
        }).reset_index(drop=True)

        majority_count = distribution['count'].max()
        distribution['imbalance_ratio'] = majority_count / distribution['count']

        self._analysis_cache['class_distribution'] = distribution
        return distribution

    def get_class_imbalance_info(self) -> Dict[str, Any]:
        """
        Get detailed class imbalance information for classification tasks.

        Returns:
            Dict containing imbalance metrics, severity, and recommendations
            Empty dict if task is not classification
        """
        if self.task != 'classification':
            logger.warning("get_class_imbalance_info() is only available for classification tasks")
            return {}

        dist = self.analyze_class_distribution()

        if dist.empty:
            return {}

        majority_class = dist.loc[dist['count'].idxmax(), 'class']
        minority_class = dist.loc[dist['count'].idxmin(), 'class']

        # Check for division by zero
        min_count = dist['count'].min()
        if min_count == 0:
            logger.warning("One or more classes have 0 samples, cannot compute imbalance ratio")
            imbalance_ratio = float('inf')
        else:
            imbalance_ratio = dist['count'].max() / min_count

        info = {
            'is_balanced': imbalance_ratio <= 1.5,
            'imbalance_ratio': imbalance_ratio,
            'majority_class': majority_class,
            'majority_count': int(dist['count'].max()),
            'minority_class': minority_class,
            'minority_count': int(min_count),
        }

        # Severity and recommendations
        if imbalance_ratio > 3:
            info['severity'] = 'severe'
            info['recommendation'] = 'Consider using SMOTE, class weights, or stratified sampling'
        elif imbalance_ratio > 1.5:
            info['severity'] = 'moderate'
            info['recommendation'] = 'Consider using class weights or stratified sampling'
        else:
            info['severity'] = 'none'
            info['recommendation'] = 'Classes are well balanced'

        return info

    def analyze_target_distribution(self) -> Dict[str, Any]:
        """
        Analyze target distribution for regression tasks.

        Returns:
            Dict containing comprehensive statistics including mean, median, std,
            skewness, kurtosis, and normality test results
            Empty dict if task is not regression
        """
        if self.task != 'regression':
            logger.warning("analyze_target_distribution() is only available for regression tasks")
            return {}

        if 'target_distribution' in self._analysis_cache:
            return self._analysis_cache['target_distribution']

        target = self.df[self.target_column].dropna()

        distribution = {
            'count': len(target),
            'mean': target.mean(),
            'median': target.median(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'range': target.max() - target.min(),
            'q25': target.quantile(0.25),
            'q75': target.quantile(0.75),
            'iqr': target.quantile(0.75) - target.quantile(0.25),
            'skewness': target.skew(),
            'kurtosis': target.kurtosis()
        }

        # Normality test (sample if dataset is large)
        if len(target) >= 3:
            try:
                sample_size = min(5000, len(target))
                sample = target.sample(sample_size, random_state=42)
                shapiro_stat, shapiro_p = stats.shapiro(sample)
                distribution['shapiro_stat'] = shapiro_stat
                distribution['shapiro_pvalue'] = shapiro_p
                distribution['is_normal'] = shapiro_p > 0.05
            except Exception as e:
                logger.warning(f"Could not compute normality test: {e}")

        self._analysis_cache['target_distribution'] = distribution
        return distribution

    def _make_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """Create a simple ASCII bar chart."""
        filled = int((value / max_value) * width) if max_value > 0 else 0
        return "#" * filled + "-" * (width - filled)

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive text summary report with executive summary.

        Returns:
            str: Formatted text report with task-specific statistics
        """
        lines = []
        info = self.get_task_info()

        # Header
        lines.append("=" * 80)
        lines.append("                         TARGET ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        task_type = info['task'].capitalize()
        n_classes = info.get('class_count', info['unique_values'])

        if self.task == 'classification':
            class_desc = "binary" if n_classes == 2 else f"{n_classes}-class"
            lines.append(f"  Task:      {task_type} ({class_desc})")
            lines.append(f"  Target:    '{info['target_column']}' ({n_classes} classes)")

            imbalance = self.get_class_imbalance_info()
            if imbalance:
                severity = imbalance['severity']
                if severity == 'severe':
                    status = f"Severe class imbalance (ratio: {imbalance['imbalance_ratio']:.1f}:1)"
                    action = "Use SMOTE, class weights, or undersampling"
                elif severity == 'moderate':
                    status = f"Moderate class imbalance (ratio: {imbalance['imbalance_ratio']:.1f}:1)"
                    action = "Consider class weights or stratified sampling"
                else:
                    status = "Classes are well balanced"
                    action = "No rebalancing needed"
                lines.append(f"  Status:    {status}")
                lines.append(f"  Action:    {action}")
        else:
            lines.append(f"  Task:      {task_type} (continuous)")
            lines.append(f"  Target:    '{info['target_column']}'")

            dist = self.analyze_target_distribution()
            if dist:
                skew = dist.get('skewness', 0)
                if abs(skew) > 2:
                    status = f"Highly skewed distribution (skewness: {skew:.2f})"
                    action = "Consider log transformation"
                elif abs(skew) > 1:
                    status = f"Moderately skewed distribution (skewness: {skew:.2f})"
                    action = "May benefit from transformation"
                else:
                    status = "Distribution is approximately symmetric"
                    action = "No transformation needed"
                lines.append(f"  Status:    {status}")
                lines.append(f"  Action:    {action}")

        lines.append("")
        lines.append("-" * 80)

        # Task Information
        lines.append("")
        lines.append(">>> TASK INFORMATION")
        lines.append(f"  Target Column:  {info['target_column']}")
        lines.append(f"  Data Type:      {info['target_dtype']}")
        if self.task == 'classification':
            lines.append(f"  Classes:        {n_classes} ({', '.join(str(c) for c in info.get('classes', [])[:5])}{'...' if n_classes > 5 else ''})")
        else:
            lines.append(f"  Unique Values:  {info['unique_values']}")
        lines.append(f"  Missing:        {info['missing_count']} ({info['missing_percent']:.2f}%)")
        lines.append("")

        if self.task == 'classification':
            lines.append(">>> CLASS DISTRIBUTION")
            dist = self.analyze_class_distribution()
            max_count = dist['count'].max()

            for _, row in dist.iterrows():
                bar = self._make_bar(row['count'], max_count, 25)
                lines.append(f"  {str(row['class']):>10}:  {int(row['count']):>6} ({row['percentage']:>5.1f}%)  {bar}")

            imbalance = self.get_class_imbalance_info()
            if imbalance:
                lines.append("")
                lines.append(f"  Imbalance Ratio: {imbalance['imbalance_ratio']:.2f}:1")
                lines.append(f"  Severity: {imbalance['severity'].upper()}")

        elif self.task == 'regression':
            lines.append(">>> TARGET STATISTICS")
            dist = self.analyze_target_distribution()
            if dist:
                lines.append(f"  Count:     {dist['count']:,}")
                lines.append(f"  Mean:      {dist['mean']:.4f}")
                lines.append(f"  Median:    {dist['median']:.4f}")
                lines.append(f"  Std Dev:   {dist['std']:.4f}")
                lines.append(f"  Range:     [{dist['min']:.4f}, {dist['max']:.4f}]")
                lines.append(f"  IQR:       {dist['iqr']:.4f}")
                lines.append("")
                lines.append(f"  Skewness:  {dist['skewness']:.4f}")
                lines.append(f"  Kurtosis:  {dist['kurtosis']:.4f}")

                if 'is_normal' in dist:
                    normality = "Yes" if dist['is_normal'] else "No"
                    lines.append(f"  Normal:    {normality} (p={dist['shapiro_pvalue']:.4f})")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)
