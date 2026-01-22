"""
Statistical analysis mixin for TargetAnalyzer.

Contains methods for feature-target relationship analysis,
class-wise statistics, correlations, mutual information, and residual analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import r2_score

from ..utils import get_feature_columns, get_numeric_columns
from .. import statistical_utils

# Configure logging
logger = logging.getLogger(__name__)


class StatisticalMixin:
    """
    Mixin providing statistical analysis methods for TargetAnalyzer.

    Requires: self.df, self.target_column, self.task
    """

    def analyze_feature_target_relationship(self,
                                           feature_columns: Optional[List[str]] = None,
                                           correct_multiple_tests: bool = False,
                                           alpha: float = 0.05,
                                           report_effect_sizes: bool = False,
                                           check_assumptions: bool = False) -> pd.DataFrame:
        """
        Analyze relationship between features and target using appropriate statistical tests.

        For classification:
        - Chi-square test for categorical features
        - ANOVA F-test for numeric features (with optional assumption checking)

        For regression:
        - Pearson correlation for numeric features
        - ANOVA F-test for categorical features

        Args:
            feature_columns: List of feature columns to analyze. If None, uses all columns except target.
            correct_multiple_tests: Apply Benjamini-Hochberg FDR correction (default False for backward compatibility)
            alpha: Significance level (default 0.05)
            report_effect_sizes: Include effect sizes in results (default False for backward compatibility)
            check_assumptions: Validate test assumptions and use robust alternatives if violated (default False for backward compatibility)

        Returns:
            DataFrame with columns:
            - feature, test_type, statistic, pvalue
            - significant (if correct_multiple_tests=False) OR significant_raw, significant_corrected, pvalue_corrected (if True)
            - effect_size, effect_interpretation (if report_effect_sizes=True)
            - assumptions_met, warnings (if check_assumptions=True)
        """
        if feature_columns is None:
            feature_columns = get_feature_columns(self.df, exclude_columns=[self.target_column], numeric_only=False)

        results = []
        target = self.df[self.target_column].dropna()

        for feature in feature_columns:
            if feature == self.target_column:
                continue

            feature_data = self.df[feature].dropna()

            # Skip if too many missing values
            if len(feature_data) < 10:
                logger.warning(f"Skipping feature '{feature}' due to insufficient data")
                continue

            try:
                if self.task == 'classification':
                    # Check for minimum number of groups before processing
                    if self.df[self.target_column].nunique() < 2:
                        logger.warning(f"Target column '{self.target_column}' has less than 2 unique values, skipping statistical tests")
                        return pd.DataFrame()

                    if pd.api.types.is_numeric_dtype(self.df[feature]):
                        # ANOVA F-test for numeric feature vs categorical target
                        groups = [group.dropna() for _, group in self.df.groupby(self.target_column)[feature]]
                        groups = [g for g in groups if len(g) > 0]

                        if len(groups) < 2:
                            continue

                        result_dict = {'feature': feature}
                        warnings_list = []
                        assumptions_dict = {}

                        # Check assumptions if requested
                        if check_assumptions:
                            sample_check = statistical_utils.validate_sample_size(groups, test_type='anova', min_size=30)
                            assumptions_dict['sufficient_sample'] = sample_check['sufficient']
                            if not sample_check['sufficient']:
                                warnings_list.append(f"Small sample size: {sample_check['actual_sizes']}")

                            normality_results = [statistical_utils.check_normality(g) for g in groups if len(g) >= 3]
                            assumptions_dict['all_normal'] = all(r['is_normal'] for r in normality_results) if normality_results else False

                            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                                variance_check = statistical_utils.check_homogeneity_of_variance(groups)
                                assumptions_dict['equal_variances'] = variance_check['equal_variances']
                                if not variance_check['equal_variances']:
                                    warnings_list.append("Unequal variances detected")
                            else:
                                assumptions_dict['equal_variances'] = None

                            if not assumptions_dict.get('all_normal', True):
                                statistic, pvalue = stats.kruskal(*groups)
                                test_name = 'Kruskal-Wallis H-test'
                                warnings_list.append("Non-normal distribution; using Kruskal-Wallis")
                            else:
                                statistic, pvalue = f_oneway(*groups)
                                test_name = 'ANOVA F-test'
                        else:
                            statistic, pvalue = f_oneway(*groups)
                            test_name = 'ANOVA F-test'

                        result_dict.update({
                            'test_type': test_name,
                            'statistic': statistic,
                            'pvalue': pvalue
                        })

                        if report_effect_sizes and 'ANOVA' in test_name:
                            effect_size_result = statistical_utils.eta_squared(groups)
                            result_dict['effect_size'] = effect_size_result['eta_squared']
                            result_dict['effect_interpretation'] = effect_size_result['interpretation']

                        if check_assumptions:
                            result_dict['assumptions_met'] = all(assumptions_dict.values()) if assumptions_dict else None
                            result_dict['warnings'] = '; '.join(warnings_list) if warnings_list else None

                        results.append(result_dict)

                    else:
                        # Chi-square test for categorical feature vs categorical target
                        contingency_table = pd.crosstab(self.df[feature], self.df[self.target_column])

                        result_dict = {'feature': feature}
                        warnings_list = []

                        if check_assumptions:
                            chi2_check = statistical_utils.check_chi2_expected_frequencies(contingency_table)
                            if not chi2_check['valid']:
                                warnings_list.append(f"{chi2_check['percent_cells_below_threshold']:.1f}% cells below expected frequency threshold")

                        chi2, pvalue, dof, expected = chi2_contingency(contingency_table)
                        result_dict.update({
                            'test_type': 'Chi-square test',
                            'statistic': chi2,
                            'pvalue': pvalue
                        })

                        if report_effect_sizes:
                            cramers_result = statistical_utils.cramers_v(contingency_table)
                            result_dict['effect_size'] = cramers_result['cramers_v']
                            result_dict['effect_interpretation'] = cramers_result['interpretation']

                        if check_assumptions and warnings_list:
                            result_dict['warnings'] = '; '.join(warnings_list)

                        results.append(result_dict)

                elif self.task == 'regression':
                    if pd.api.types.is_numeric_dtype(self.df[feature]):
                        valid_idx = self.df[[feature, self.target_column]].dropna().index
                        if len(valid_idx) <= 2:
                            continue

                        corr, pvalue = pearsonr(self.df.loc[valid_idx, feature],
                                                self.df.loc[valid_idx, self.target_column])

                        result_dict = {
                            'feature': feature,
                            'test_type': 'Pearson correlation',
                            'statistic': corr,
                            'pvalue': pvalue
                        }

                        if report_effect_sizes:
                            abs_corr = abs(corr)
                            if abs_corr < 0.3:
                                interpretation = 'small'
                            elif abs_corr < 0.5:
                                interpretation = 'medium'
                            else:
                                interpretation = 'large'
                            result_dict['effect_size'] = corr
                            result_dict['effect_interpretation'] = interpretation

                        results.append(result_dict)
                    else:
                        groups = [group.dropna() for _, group in self.df.groupby(feature)[self.target_column]]
                        groups = [g for g in groups if len(g) > 0]

                        if len(groups) < 2:
                            continue

                        statistic, pvalue = f_oneway(*groups)
                        result_dict = {
                            'feature': feature,
                            'test_type': 'ANOVA F-test',
                            'statistic': statistic,
                            'pvalue': pvalue
                        }

                        if report_effect_sizes:
                            effect_size_result = statistical_utils.eta_squared(groups)
                            result_dict['effect_size'] = effect_size_result['eta_squared']
                            result_dict['effect_interpretation'] = effect_size_result['interpretation']

                        results.append(result_dict)

            except Exception as e:
                logger.warning(f"Could not analyze feature '{feature}': {e}")
                continue

        df_results = pd.DataFrame(results)

        if df_results.empty:
            return df_results

        if correct_multiple_tests and len(df_results) > 1:
            correction_result = statistical_utils.apply_multiple_testing_correction(
                df_results['pvalue'].values,
                method='fdr_bh',
                alpha=alpha
            )

            df_results['pvalue_corrected'] = correction_result['corrected_pvalues']
            df_results['significant_raw'] = df_results['pvalue'] < alpha
            df_results['significant_corrected'] = correction_result['reject']

            logger.info(f"Multiple testing correction: {correction_result['num_significant_raw']} "
                       f"→ {correction_result['num_significant_corrected']} significant features")
        else:
            df_results['significant'] = df_results['pvalue'] < alpha

        df_results = df_results.sort_values('pvalue')

        return df_results

    def analyze_class_wise_statistics(self,
                                      feature_columns: Optional[List[str]] = None,
                                      confidence_level: float = 0.95,
                                      include_ci: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Compute statistics for each feature broken down by target class (classification only).

        Args:
            feature_columns: List of numeric feature columns. If None, uses all numeric columns.
            confidence_level: Confidence level for CIs (default 0.95)
            include_ci: Include confidence intervals for mean and median (default False for backward compatibility)

        Returns:
            Dict mapping feature names to DataFrames with class-wise statistics.
        """
        if self.task != 'classification':
            logger.warning("analyze_class_wise_statistics() is only available for classification tasks")
            return {}

        if feature_columns is None:
            feature_columns = get_feature_columns(self.df, exclude_columns=[self.target_column], numeric_only=True)

        results = {}
        grouped = self.df.groupby(self.target_column)

        for feature in feature_columns:
            if feature == self.target_column:
                continue

            stats_df = grouped[feature].agg([
                ('count', 'count'),
                ('mean', 'mean'),
                ('median', lambda x: x.quantile(0.5)),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max')
            ])

            if include_ci:
                mean_ci_lower = []
                mean_ci_upper = []
                median_ci_lower = []
                median_ci_upper = []

                for cls in stats_df.index:
                    class_data = self.df[self.df[self.target_column] == cls][feature].dropna()

                    mean_ci = statistical_utils.calculate_mean_ci(class_data, confidence=confidence_level)
                    mean_ci_lower.append(mean_ci['ci_lower'])
                    mean_ci_upper.append(mean_ci['ci_upper'])

                    median_ci = statistical_utils.bootstrap_ci(
                        class_data.values,
                        statistic_func=np.median,
                        n_bootstrap=1000,
                        confidence=confidence_level,
                        random_state=42
                    )
                    median_ci_lower.append(median_ci['ci_lower'])
                    median_ci_upper.append(median_ci['ci_upper'])

                stats_df['mean_ci_lower'] = mean_ci_lower
                stats_df['mean_ci_upper'] = mean_ci_upper
                stats_df['median_ci_lower'] = median_ci_lower
                stats_df['median_ci_upper'] = median_ci_upper

            class_stats = []
            for cls, row in stats_df.iterrows():
                if row['count'] > 0:
                    stat_dict = {
                        'class': cls,
                        'count': int(row['count']),
                        'mean': row['mean'],
                        'median': row['median'],
                        'std': row['std'],
                        'min': row['min'],
                        'max': row['max']
                    }

                    if include_ci:
                        stat_dict['mean_ci_lower'] = row['mean_ci_lower']
                        stat_dict['mean_ci_upper'] = row['mean_ci_upper']
                        stat_dict['median_ci_lower'] = row['median_ci_lower']
                        stat_dict['median_ci_upper'] = row['median_ci_upper']

                    class_stats.append(stat_dict)

            if class_stats:
                results[feature] = pd.DataFrame(class_stats)

        return results

    def analyze_feature_correlations(self,
                                     feature_columns: Optional[List[str]] = None,
                                     method: str = 'pearson',
                                     include_ci: bool = False,
                                     check_linearity: bool = False,
                                     confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Analyze correlations between numeric features and target (regression only).

        Args:
            feature_columns: List of numeric features. If None, uses all numeric columns.
            method: Correlation method ('pearson' or 'spearman')
            include_ci: Include confidence intervals for correlations
            check_linearity: Compare Pearson vs Spearman to detect non-linearity
            confidence_level: Confidence level for CIs (default 0.95)

        Returns:
            DataFrame with correlation results sorted by absolute correlation.
        """
        if self.task != 'regression':
            logger.warning("analyze_feature_correlations() is only available for regression tasks")
            return pd.DataFrame()

        if feature_columns is None:
            feature_columns = get_feature_columns(self.df, exclude_columns=[self.target_column], numeric_only=True)

        results = []
        for feature in feature_columns:
            if feature == self.target_column:
                continue

            valid_idx = self.df[[feature, self.target_column]].dropna().index
            if len(valid_idx) < 3:
                continue

            try:
                if method == 'pearson':
                    corr, pvalue = pearsonr(self.df.loc[valid_idx, feature],
                                           self.df.loc[valid_idx, self.target_column])
                elif method == 'spearman':
                    corr, pvalue = spearmanr(self.df.loc[valid_idx, feature],
                                            self.df.loc[valid_idx, self.target_column])
                else:
                    logger.warning(f"Unknown correlation method: {method}")
                    continue

                result_dict = {
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'pvalue': pvalue,
                    'significant': pvalue < 0.05
                }

                if include_ci:
                    ci = statistical_utils.calculate_correlation_ci(
                        corr,
                        n=len(valid_idx),
                        confidence=confidence_level
                    )
                    result_dict['ci_lower'] = ci['ci_lower']
                    result_dict['ci_upper'] = ci['ci_upper']

                if check_linearity and method == 'pearson':
                    spearman_corr, _ = spearmanr(self.df.loc[valid_idx, feature],
                                                 self.df.loc[valid_idx, self.target_column])
                    diff = abs(corr - spearman_corr)

                    if diff > 0.2:
                        result_dict['linearity_warning'] = f"Non-linear relationship detected (Spearman={spearman_corr:.3f}, diff={diff:.3f})"
                    else:
                        result_dict['linearity_warning'] = None

                results.append(result_dict)

            except Exception as e:
                logger.warning(f"Could not compute correlation for '{feature}': {e}")
                continue

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('abs_correlation', ascending=False)

        return df_results

    def analyze_mutual_information(self, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate mutual information between features and target.

        Args:
            feature_columns: List of features. If None, uses all columns except target.

        Returns:
            DataFrame with columns: feature, mutual_info, normalized_mi
        """
        if feature_columns is None:
            feature_columns = get_feature_columns(self.df, exclude_columns=[self.target_column], numeric_only=False)

        numeric_features = get_numeric_columns(self.df, columns=feature_columns)

        if not numeric_features:
            logger.warning("No numeric features found for mutual information analysis")
            return pd.DataFrame()

        X = self.df[numeric_features].fillna(0)
        y = self.df[self.target_column].dropna()

        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(y) < 10:
            logger.warning("Insufficient data for mutual information analysis")
            return pd.DataFrame()

        try:
            if self.task == 'classification':
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)

            max_mi = np.log(len(np.unique(y))) if self.task == 'classification' else np.max(mi_scores)
            if max_mi > 0:
                normalized_mi = mi_scores / max_mi
            else:
                normalized_mi = mi_scores

            results = pd.DataFrame({
                'feature': numeric_features,
                'mutual_info': mi_scores,
                'normalized_mi': normalized_mi
            })

            results = results.sort_values('mutual_info', ascending=False)
            return results

        except Exception as e:
            logger.warning(f"Could not compute mutual information: {e}")
            return pd.DataFrame()

    def analyze_residuals(self, predictions: pd.Series) -> Dict[str, Any]:
        """
        Analyze residuals for regression tasks.

        Args:
            predictions: Predicted target values (must have same index as self.df)

        Returns:
            Dict containing residual statistics and test results
        """
        if self.task != 'regression':
            logger.warning("analyze_residuals() is only available for regression tasks")
            return {}

        common_idx = self.df[self.target_column].dropna().index.intersection(predictions.index)
        actual = self.df.loc[common_idx, self.target_column]
        pred = predictions.loc[common_idx]

        if len(actual) == 0:
            logger.warning("No valid data for residual analysis")
            return {}

        residuals = actual - pred

        results = {
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_min': residuals.min(),
            'residual_max': residuals.max(),
            'mae': np.abs(residuals).mean(),
            'rmse': np.sqrt((residuals ** 2).mean()),
            'r2_score': r2_score(actual, pred)
        }

        if len(residuals) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals.sample(min(5000, len(residuals))))
                results['shapiro_stat'] = shapiro_stat
                results['shapiro_pvalue'] = shapiro_p
                results['residuals_normal'] = shapiro_p > 0.05
            except Exception as e:
                logger.warning(f"Could not compute normality test on residuals: {e}")

        return results
