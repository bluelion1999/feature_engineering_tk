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
            # Check for features with very low p-values AND a large effect size.
            #
            # P-value alone is not a valid leakage signal: with enough rows, even a
            # trivially small true effect produces a vanishingly small p-value
            # (p shrinks toward 0 as N grows, independent of effect magnitude). At
            # large N, "p < 1e-10" is common for features with no practical
            # relationship to the target and must not be treated as evidence of
            # leakage on its own. This mirrors why the regression branch above
            # gates on correlation magnitude (> 0.99) rather than p-value.
            #
            # We reuse analyze_feature_target_relationship()'s effect-size support
            # (eta-squared for ANOVA F-test / numeric features, Cramer's V for
            # chi-square / categorical features, both from statistical_utils.py)
            # instead of recomputing the underlying tests here.
            #
            # Threshold choice: statistical_utils's own "large" cutoffs (eta^2 >=
            # 0.14 per Cohen's convention; Cramer's V >= ~0.29-0.5 depending on
            # table size) describe a strong *real* relationship, which is exactly
            # the kind of legitimate strong predictor this detector must not flag.
            # Leakage detection specifically wants "the feature basically IS the
            # target" (e.g. a proxy/derived column) -- a much narrower, near-
            # deterministic case. We use a stricter, leakage-specific threshold of
            # 0.8 (80%+ of variance/association explained) for both eta-squared
            # and Cramer's V, chosen to be in the same "near-total" spirit as the
            # regression branch's correlation > 0.99 (r^2 > 0.98) leakage gate,
            # while remaining reachable on eta^2/Cramer's V's bounded [0, 1] scale.
            LEAKAGE_EFFECT_SIZE_THRESHOLD = 0.8

            rel_df = self.analyze_feature_target_relationship(report_effect_sizes=True)
            if not rel_df.empty and 'effect_size' in rel_df.columns:
                suspicious = rel_df[
                    (rel_df['pvalue'] < 1e-10) &
                    (rel_df['effect_size'] >= LEAKAGE_EFFECT_SIZE_THRESHOLD)
                ]
                leakage_suspects.extend([
                    {
                        'feature': row['feature'],
                        'reason': (
                            f'Extremely significant relationship (p={row["pvalue"]:.2e}) AND '
                            f'large effect size ('
                            f'{"eta²" if "ANOVA" in row.get("test_type", "") else "Cramers V"}'
                            f'={row["effect_size"]:.3f}, {row.get("effect_interpretation", "large")})'
                        ),
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

    def generate_recommendations(self,
                                  quality: Optional[Dict[str, Any]] = None,
                                  mi_df: Optional[pd.DataFrame] = None,
                                  vif_df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Args:
            quality: Pre-computed result of analyze_data_quality(). If None, computed
                internally. Callers that already have this (e.g. generate_full_report())
                can pass it in to avoid recomputing it.
            mi_df: Pre-computed result of analyze_mutual_information(). If None, computed
                internally.
            vif_df: Pre-computed result of calculate_vif(). If None, computed internally.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Data quality recommendations
        if quality is None:
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
            if mi_df is None:
                mi_df = self.analyze_mutual_information()
            if not mi_df.empty:
                if self.task == 'classification':
                    # normalized_mi is a fixed absolute bound (see analyze_mutual_information
                    # docstring), so an absolute threshold on it is meaningful here.
                    low_mi = mi_df[mi_df['normalized_mi'] < 0.01]
                    if len(low_mi) > 0:
                        recommendations.append(
                            f"📊 {len(low_mi)} features have very low mutual information with target. "
                            "Consider feature selection."
                        )
                else:
                    # regression: 'relative_mi' is only a within-analysis ranking aid, not an
                    # absolute [0, 1] score (see analyze_mutual_information docstring) -- a
                    # feature can score relative_mi == 1.0 purely for being the "best of the
                    # noise" when every feature is weak.
                    #
                    # We deliberately do NOT threshold the raw 'mutual_info' score against a
                    # fixed absolute epsilon here either: mutual_info_regression's KSG-based
                    # estimator has finite-sample bias whose magnitude depends strongly on
                    # sample size (empirically ~0.07 nats of noise-floor bias at n=200,
                    # dropping to ~0.01 nats at n=5000 for pure noise), so any single fixed
                    # cutoff would be too strict for small datasets and too lax for large
                    # ones -- a false precision this recommendation should not manufacture.
                    #
                    # Instead we only ever make a *relative* claim (features ranking far below
                    # the strongest one here), with an explicit caveat that this is not proof
                    # of absolute predictive value -- so we never mis-describe a noise feature
                    # that merely tops the relative ranking as "well correlated" or similar.
                    low_relative_mi = mi_df[mi_df['relative_mi'] < 0.1]
                    if len(low_relative_mi) > 0:
                        recommendations.append(
                            f"📊 {len(low_relative_mi)} feature(s) rank far below the strongest "
                            "feature in this analysis by mutual information. Note this ranking "
                            "is relative only -- check the raw 'mutual_info' column directly, "
                            "since even the top-ranked feature may carry little real predictive "
                            "signal if the target is weakly related to all analyzed features."
                        )
        except Exception as e:
            logger.debug(f"Could not compute mutual information recommendations: {e}")

        # Multicollinearity check
        try:
            if vif_df is None:
                vif_df = self.calculate_vif()
            if not vif_df.empty:
                high_vif = vif_df[vif_df['VIF'] > 10]
                if len(high_vif) > 0:
                    recommendations.append(
                        f"📉 {len(high_vif)} features have high multicollinearity (VIF>10). "
                        f"Consider removing: {', '.join(high_vif.head(3)['feature'].tolist())}"
                    )
        except Exception as e:
            logger.debug(f"Could not compute VIF recommendations: {e}")

        if not recommendations:
            recommendations.append("✓ No major issues detected. Data quality looks good!")

        return recommendations
