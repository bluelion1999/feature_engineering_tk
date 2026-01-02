"""
Tests for statistical_utils module.

Tests assumption validation, effect size calculations, multiple testing corrections,
and confidence interval utilities.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from feature_engineering_tk import statistical_utils


class TestNormalityChecks:
    """Test normality assumption validation."""

    def test_check_normality_normal_data(self):
        """Test normality check with normally distributed data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = statistical_utils.check_normality(data)

        assert result['test_name'] == 'Shapiro-Wilk'
        assert result['is_normal'] == True
        assert result['pvalue'] > 0.05
        assert 'parametric' in result['recommendation'].lower()

    def test_check_normality_non_normal_data(self):
        """Test normality check with non-normal (uniform) data."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 1000)

        result = statistical_utils.check_normality(data)

        assert result['test_name'] == 'Shapiro-Wilk'
        assert result['is_normal'] == False
        assert result['pvalue'] < 0.05
        assert 'non-parametric' in result['recommendation'].lower()

    def test_check_normality_small_sample(self):
        """Test normality check with insufficient data."""
        data = np.array([1, 2])

        result = statistical_utils.check_normality(data)

        assert result['is_normal'] == False
        assert result['statistic'] is None
        assert 'insufficient data' in result['recommendation'].lower()

    def test_check_normality_large_sample(self):
        """Test normality check with large sample (uses subsample)."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)

        result = statistical_utils.check_normality(data, method='shapiro')

        # Should use subsample of 5000
        assert result['sample_size'] == 5000
        assert result['is_normal'] == True


class TestHomogeneityOfVariance:
    """Test homogeneity of variance assumption."""

    def test_check_homogeneity_equal_variances(self):
        """Test Levene's test with equal variances."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0, 1, 100)

        result = statistical_utils.check_homogeneity_of_variance([group1, group2])

        assert result['test_name'] == "Levene's test"
        assert result['equal_variances'] == True
        assert 'standard ANOVA' in result['recommendation']

    def test_check_homogeneity_unequal_variances(self):
        """Test Levene's test with unequal variances."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0, 5, 100)  # Different variance

        result = statistical_utils.check_homogeneity_of_variance([group1, group2])

        assert result['test_name'] == "Levene's test"
        assert result['equal_variances'] == False
        assert 'Welch' in result['recommendation'] or 'non-parametric' in result['recommendation']

    def test_check_homogeneity_insufficient_data(self):
        """Test with insufficient data in some groups."""
        group1 = np.array([1])
        group2 = np.array([2, 3])

        result = statistical_utils.check_homogeneity_of_variance([group1, group2])

        assert result['equal_variances'] == False
        assert 'insufficient data' in result['recommendation'].lower()


class TestSampleSizeValidation:
    """Test sample size validation."""

    def test_validate_sample_size_sufficient_anova(self):
        """Test sample size validation for ANOVA with sufficient data."""
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0, 1, 50)

        result = statistical_utils.validate_sample_size([group1, group2], test_type='anova', min_size=30)

        assert result['sufficient'] == True
        assert result['warning'] is None

    def test_validate_sample_size_insufficient_anova(self):
        """Test sample size validation for ANOVA with insufficient data."""
        group1 = np.random.normal(0, 1, 15)
        group2 = np.random.normal(0, 1, 20)

        result = statistical_utils.validate_sample_size([group1, group2], test_type='anova', min_size=30)

        assert result['sufficient'] == False
        assert result['warning'] is not None


class TestChi2ExpectedFrequencies:
    """Test chi-square expected frequency validation."""

    def test_check_chi2_valid_table(self):
        """Test chi-square validation with valid table."""
        table = np.array([[20, 30], [40, 50]])

        result = statistical_utils.check_chi2_expected_frequencies(table)

        assert result['valid'] == True
        assert 'appropriate' in result['recommendation'].lower()

    def test_check_chi2_invalid_table(self):
        """Test chi-square validation with sparse table."""
        table = np.array([[2, 3], [1, 2]])

        result = statistical_utils.check_chi2_expected_frequencies(table)

        assert result['valid'] == False
        assert 'Fisher' in result['recommendation'] or 'invalid' in result['recommendation']


class TestEffectSizes:
    """Test effect size calculations."""

    def test_cohens_d_medium_effect(self):
        """Test Cohen's d calculation with medium effect."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0.5, 1, 100)  # Half SD difference

        result = statistical_utils.cohens_d(group1, group2)

        assert 'cohens_d' in result
        assert abs(result['cohens_d'] - (-0.5)) < 0.2  # Approximately -0.5
        assert result['interpretation'] in ['small', 'medium']

    def test_cohens_d_large_effect(self):
        """Test Cohen's d with large effect."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(1, 1, 100)  # One SD difference

        result = statistical_utils.cohens_d(group1, group2)

        assert abs(result['cohens_d']) > 0.7
        assert result['interpretation'] in ['medium', 'large']

    def test_eta_squared_from_groups(self):
        """Test eta-squared calculation from group data."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0.5, 1, 100)
        group3 = np.random.normal(1, 1, 100)

        result = statistical_utils.eta_squared([group1, group2, group3])

        assert 0 <= result['eta_squared'] <= 1
        assert result['interpretation'] in ['negligible', 'small', 'medium', 'large']
        assert result['percent_variance_explained'] >= 0

    def test_cramers_v_2x2_table(self):
        """Test Cramér's V for 2x2 contingency table."""
        table = np.array([[20, 30], [40, 50]])

        result = statistical_utils.cramers_v(table)

        assert 0 <= result['cramers_v'] <= 1
        assert result['interpretation'] in ['negligible', 'small', 'medium', 'large']
        assert 'chi2_statistic' in result
        assert 'pvalue' in result

    def test_pearson_r_to_d_conversion(self):
        """Test conversion from Pearson r to Cohen's d."""
        r = 0.5
        d = statistical_utils.pearson_r_to_d(r)

        # r=0.5 should give d ≈ 1.15
        assert abs(d - 1.15) < 0.1


class TestMultipleTestingCorrection:
    """Test multiple testing correction."""

    def test_fdr_bh_correction(self):
        """Test Benjamini-Hochberg FDR correction."""
        pvalues = [0.001, 0.01, 0.03, 0.05, 0.10, 0.50]

        result = statistical_utils.apply_multiple_testing_correction(pvalues, method='fdr_bh', alpha=0.05)

        assert len(result['corrected_pvalues']) == len(pvalues)
        assert result['method'] == 'fdr_bh'
        assert result['num_significant_raw'] > result['num_significant_corrected']
        assert all(result['corrected_pvalues'] >= pvalues)

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        pvalues = [0.001, 0.01, 0.03, 0.05]

        result = statistical_utils.apply_multiple_testing_correction(pvalues, method='bonferroni', alpha=0.05)

        assert result['method'] == 'bonferroni'
        # Bonferroni is most conservative
        assert result['num_significant_corrected'] <= result['num_significant_raw']


class TestConfidenceIntervals:
    """Test confidence interval calculations."""

    def test_calculate_mean_ci_normal_data(self):
        """Test mean CI calculation with normal data."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 50)

        result = statistical_utils.calculate_mean_ci(data, confidence=0.95)

        assert 'mean' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] < result['mean'] < result['ci_upper']
        assert result['confidence_level'] == 0.95

    def test_calculate_mean_ci_small_sample(self):
        """Test mean CI with small sample."""
        data = np.array([10])

        result = statistical_utils.calculate_mean_ci(data)

        assert result['mean'] == 10
        assert np.isnan(result['ci_lower'])
        assert np.isnan(result['ci_upper'])

    def test_calculate_correlation_ci(self):
        """Test correlation CI using Fisher Z-transformation."""
        r = 0.5
        n = 100

        result = statistical_utils.calculate_correlation_ci(r, n, confidence=0.95)

        assert result['r'] == 0.5
        assert result['ci_lower'] < r < result['ci_upper']
        assert -1 <= result['ci_lower'] <= 1
        assert -1 <= result['ci_upper'] <= 1

    def test_bootstrap_ci_median(self):
        """Test bootstrap CI for median."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 100)

        result = statistical_utils.bootstrap_ci(
            data,
            statistic_func=np.median,
            n_bootstrap=1000,
            confidence=0.95,
            random_state=42
        )

        assert 'statistic' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] < result['statistic'] < result['ci_upper']
        assert len(result['bootstrap_distribution']) == 1000

    def test_bootstrap_ci_custom_statistic(self):
        """Test bootstrap CI with custom statistic function."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        # Use 90th percentile as custom statistic
        def percentile_90(x):
            return np.percentile(x, 90)

        result = statistical_utils.bootstrap_ci(
            data,
            statistic_func=percentile_90,
            n_bootstrap=500,
            confidence=0.90,
            random_state=42
        )

        assert result['confidence_level'] == 0.90
        assert result['ci_lower'] < result['statistic'] < result['ci_upper']


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_check_normality_with_nans(self):
        """Test normality check handles NaN values."""
        data = np.array([1, 2, 3, np.nan, 5, 6, 7])

        result = statistical_utils.check_normality(data)

        # Should remove NaN and still work
        assert result['sample_size'] == 6
        assert result['is_normal'] in [True, False]

    def test_cohens_d_zero_variance(self):
        """Test Cohen's d with zero variance."""
        group1 = np.array([5, 5, 5, 5])
        group2 = np.array([5, 5, 5, 5])

        result = statistical_utils.cohens_d(group1, group2)

        # Should handle gracefully
        assert np.isnan(result['cohens_d']) or result['cohens_d'] == 0

    def test_bootstrap_ci_insufficient_data(self):
        """Test bootstrap CI with insufficient data."""
        data = np.array([10])

        result = statistical_utils.bootstrap_ci(
            data,
            statistic_func=np.median,
            n_bootstrap=100
        )

        assert result['statistic'] == 10
        assert np.isnan(result['ci_lower'])


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_complete_anova_workflow(self):
        """Test complete ANOVA workflow with all checks."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)
        group3 = np.random.normal(1, 1, 50)
        groups = [group1, group2, group3]

        # Check assumptions
        sample_check = statistical_utils.validate_sample_size(groups, test_type='anova')
        assert sample_check['sufficient'] == True

        normality_results = [statistical_utils.check_normality(g) for g in groups]
        assert all(r['is_normal'] for r in normality_results)

        variance_check = statistical_utils.check_homogeneity_of_variance(groups)
        assert variance_check['equal_variances'] == True

        # Calculate effect size
        effect_size = statistical_utils.eta_squared(groups)
        assert effect_size['eta_squared'] > 0

    def test_chi_square_workflow(self):
        """Test chi-square workflow with validation."""
        table = np.array([[30, 20], [40, 10]])

        # Check assumptions
        chi2_check = statistical_utils.check_chi2_expected_frequencies(table)
        assert chi2_check['valid'] == True

        # Calculate effect size
        cramers = statistical_utils.cramers_v(table)
        assert 0 <= cramers['cramers_v'] <= 1

    def test_correlation_workflow_with_ci(self):
        """Test correlation workflow with confidence intervals."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = x + np.random.normal(0, 0.5, 100)  # Correlated

        # Calculate correlation
        r, p = stats.pearsonr(x, y)

        # Get CI
        ci = statistical_utils.calculate_correlation_ci(r, n=100, confidence=0.95)

        assert ci['ci_lower'] < r < ci['ci_upper']
        assert r > 0.5  # Should be moderately correlated
