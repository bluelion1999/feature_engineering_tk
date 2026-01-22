"""
Data quality and recommendations mixin for TargetAnalyzer.

Contains methods for data quality analysis, VIF calculation, and recommendations.
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Optional

from ..utils import get_feature_columns
from ..data_analysis import DataAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class QualityMixin:
    """
    Mixin providing data quality analysis and recommendations for TargetAnalyzer.

    Requires: self.df, self.target_column, self.task,
              self.analyze_feature_target_relationship, self.analyze_feature_correlations,
              self.get_class_imbalance_info, self.analyze_target_distribution,
              self.analyze_mutual_information
    """

    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Analyze data quality issues including missing values, outliers, and potential leakage.

        Returns:
            Dict containing data quality metrics
        """
        results = {}

        # Missing values analysis
        feature_cols = get_feature_columns(self.df, exclude_columns=[self.target_column], numeric_only=False)
        missing_by_feature = {}
        for col in feature_cols:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_by_feature[col] = {
                    'count': missing_count,
                    'percent': missing_count / len(self.df) * 100
                }

        results['missing_values'] = missing_by_feature
        results['target_missing'] = {
            'count': self.df[self.target_column].isnull().sum(),
            'percent': self.df[self.target_column].isnull().sum() / len(self.df) * 100
        }

        # Potential data leakage detection
        leakage_suspects = []

        if self.task == 'regression':
            # Check for perfect or near-perfect correlations
            corr_df = self.analyze_feature_correlations()
            if not corr_df.empty:
                perfect_corr = corr_df[corr_df['abs_correlation'] > 0.99]
                leakage_suspects.extend([
                    {
                        'feature': row['feature'],
                        'reason': f'Near-perfect correlation ({row["correlation"]:.4f})',
                        'severity': 'high'
                    }
                    for row in perfect_corr.to_dict('records')
                ])

        elif self.task == 'classification':
            # Check for features with very low p-values and high test statistics
            rel_df = self.analyze_feature_target_relationship()
            if not rel_df.empty:
                suspicious = rel_df[rel_df['pvalue'] < 1e-10]
                leakage_suspects.extend([
                    {
                        'feature': row['feature'],
                        'reason': f'Extremely significant relationship (p={row["pvalue"]:.2e})',
                        'severity': 'medium'
                    }
                    for row in suspicious.to_dict('records')
                ])

        results['leakage_suspects'] = leakage_suspects

        # Constant features
        constant_features = []
        for col in feature_cols:
            if self.df[col].nunique() == 1:
                constant_features.append(col)

        results['constant_features'] = constant_features

        return results

    def calculate_vif(self, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for multicollinearity detection.

        Wrapper around DataAnalyzer.calculate_vif() that automatically excludes the target column.

        Args:
            feature_columns: List of numeric features. If None, uses all numeric columns
                           (excluding target).

        Returns:
            DataFrame with columns: feature, VIF (sorted by VIF descending)

        Note:
            This delegates to DataAnalyzer.calculate_vif() but excludes the target column.
            For general VIF calculation without a target, use DataAnalyzer directly.
        """
        if feature_columns is None:
            feature_columns = get_feature_columns(self.df, exclude_columns=[self.target_column], numeric_only=True)

        # Delegate to DataAnalyzer implementation
        analyzer = DataAnalyzer(self.df)
        return analyzer.calculate_vif(columns=feature_columns)

    def generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Data quality recommendations
        quality = self.analyze_data_quality()

        if quality['missing_values']:
            high_missing = [k for k, v in quality['missing_values'].items() if v['percent'] > 50]
            if high_missing:
                recommendations.append(
                    f"⚠ {len(high_missing)} features have >50% missing values: "
                    f"{', '.join(high_missing[:3])}{'...' if len(high_missing) > 3 else ''}. "
                    "Consider dropping or imputing."
                )

        if quality['target_missing']['percent'] > 0:
            recommendations.append(
                f"⚠ Target column has {quality['target_missing']['percent']:.1f}% missing values. "
                "These rows cannot be used for supervised learning."
            )

        if quality['constant_features']:
            recommendations.append(
                f"⚠ {len(quality['constant_features'])} constant features provide no information. "
                f"Consider dropping: {', '.join(quality['constant_features'][:3])}"
            )

        if quality['leakage_suspects']:
            high_severity = [s for s in quality['leakage_suspects'] if s['severity'] == 'high']
            if high_severity:
                recommendations.append(
                    f"🚨 {len(high_severity)} features show signs of potential data leakage. "
                    "Review these features carefully!"
                )

        # Task-specific recommendations
        if self.task == 'classification':
            imbalance = self.get_class_imbalance_info()
            if imbalance and imbalance['severity'] != 'none':
                recommendations.append(f"⚙ {imbalance['recommendation']}")

        elif self.task == 'regression':
            dist = self.analyze_target_distribution()
            if 'is_normal' in dist and not dist['is_normal']:
                if abs(dist['skewness']) > 1:
                    recommendations.append(
                        "⚙ Target is highly skewed. Consider log transformation or robust regression methods."
                    )

        # Feature selection recommendations
        try:
            mi_df = self.analyze_mutual_information()
            if not mi_df.empty:
                low_mi = mi_df[mi_df['normalized_mi'] < 0.01]
                if len(low_mi) > 0:
                    recommendations.append(
                        f"📊 {len(low_mi)} features have very low mutual information with target. "
                        "Consider feature selection."
                    )
        except:
            pass

        # Multicollinearity check
        try:
            vif_df = self.calculate_vif()
            if not vif_df.empty:
                high_vif = vif_df[vif_df['VIF'] > 10]
                if len(high_vif) > 0:
                    recommendations.append(
                        f"📉 {len(high_vif)} features have high multicollinearity (VIF>10). "
                        f"Consider removing: {', '.join(high_vif.head(3)['feature'].tolist())}"
                    )
        except:
            pass

        if not recommendations:
            recommendations.append("✓ No major issues detected. Data quality looks good!")

        return recommendations
