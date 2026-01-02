"""
Statistical utilities for robust statistical analysis.

This module provides utility functions for validating statistical assumptions,
calculating effect sizes, applying multiple testing corrections, and computing
confidence intervals. Designed to enhance the statistical rigor of the
Feature Engineering Toolkit.

Key Functions:
    - check_normality: Test normality assumption
    - check_homogeneity_of_variance: Test equal variance assumption
    - validate_sample_size: Validate sample size requirements
    - check_chi2_expected_frequencies: Validate chi-square assumptions
    - cohens_d: Calculate Cohen's d effect size
    - eta_squared: Calculate eta-squared effect size for ANOVA
    - cramers_v: Calculate Cramér's V effect size for chi-square
    - apply_multiple_testing_correction: Apply FDR or Bonferroni correction
    - calculate_mean_ci: Calculate confidence interval for mean
    - calculate_correlation_ci: Calculate CI for correlation
    - bootstrap_ci: Bootstrap confidence interval for any statistic
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Union, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


# =====================================================
# ASSUMPTION VALIDATION FUNCTIONS
# =====================================================

def check_normality(data: Union[np.ndarray, pd.Series],
                    method: str = 'shapiro',
                    alpha: float = 0.05) -> Dict[str, Any]:
    """
    Test normality assumption using Shapiro-Wilk or other tests.

    For large samples (n > 5000), uses a random subsample to avoid
    computational issues and overly sensitive results.

    Args:
        data: Data to test for normality
        method: Test to use ('shapiro', 'normaltest', 'anderson')
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing:
        - test_name: Name of test performed
        - statistic: Test statistic
        - pvalue: P-value (if applicable)
        - is_normal: Boolean indicating if data appears normal
        - recommendation: Suggested course of action
        - sample_size: Actual sample size tested

    Example:
        >>> data = np.random.normal(0, 1, 1000)
        >>> result = check_normality(data)
        >>> print(result['is_normal'])
        True
    """
    # Convert to numpy array and remove NaN
    data_clean = np.asarray(data)
    data_clean = data_clean[~np.isnan(data_clean)]

    n = len(data_clean)

    if n < 3:
        logger.warning(f"Sample size too small for normality test (n={n})")
        return {
            'test_name': method,
            'statistic': None,
            'pvalue': None,
            'is_normal': False,
            'recommendation': 'Use non-parametric methods (insufficient data)',
            'sample_size': n
        }

    # For large samples, use subsample to avoid computational issues
    if n > 5000 and method == 'shapiro':
        logger.info(f"Large sample (n={n}), using random subsample of 5000 for Shapiro-Wilk test")
        np.random.seed(42)
        data_clean = np.random.choice(data_clean, size=5000, replace=False)
        n = 5000

    # Perform normality test
    if method == 'shapiro':
        statistic, pvalue = stats.shapiro(data_clean)
        test_name = 'Shapiro-Wilk'
    elif method == 'normaltest':
        statistic, pvalue = stats.normaltest(data_clean)
        test_name = "D'Agostino-Pearson"
    elif method == 'anderson':
        result = stats.anderson(data_clean)
        statistic = result.statistic
        # Anderson-Darling uses critical values, not p-values
        # Use 5% significance level (index 2 typically corresponds to 5%)
        critical_value = result.critical_values[2]
        pvalue = None
        is_normal = statistic < critical_value
        return {
            'test_name': 'Anderson-Darling',
            'statistic': statistic,
            'pvalue': None,
            'critical_value': critical_value,
            'is_normal': is_normal,
            'recommendation': 'Use parametric methods' if is_normal else 'Use non-parametric methods',
            'sample_size': n
        }
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'shapiro', 'normaltest', or 'anderson'")

    is_normal = pvalue > alpha

    return {
        'test_name': test_name,
        'statistic': statistic,
        'pvalue': pvalue,
        'is_normal': is_normal,
        'recommendation': 'Use parametric methods' if is_normal else 'Use non-parametric methods',
        'sample_size': n
    }


def check_homogeneity_of_variance(groups: List[Union[np.ndarray, pd.Series]],
                                   method: str = 'levene',
                                   alpha: float = 0.05) -> Dict[str, Any]:
    """
    Test equal variance assumption using Levene's test or Bartlett's test.

    Levene's test is more robust to non-normality than Bartlett's test.

    Args:
        groups: List of arrays/series representing different groups
        method: Test to use ('levene' or 'bartlett')
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing:
        - test_name: Name of test performed
        - statistic: Test statistic
        - pvalue: P-value
        - equal_variances: Boolean indicating if variances are equal
        - recommendation: Suggested course of action
        - group_sizes: Sample sizes per group
        - group_variances: Variance of each group

    Example:
        >>> group1 = np.random.normal(0, 1, 100)
        >>> group2 = np.random.normal(0, 1, 100)
        >>> result = check_homogeneity_of_variance([group1, group2])
        >>> print(result['equal_variances'])
        True
    """
    # Clean groups (remove NaN)
    groups_clean = [np.asarray(g)[~np.isnan(np.asarray(g))] for g in groups]

    # Check if we have at least 2 groups
    if len(groups_clean) < 2:
        raise ValueError("Need at least 2 groups to test homogeneity of variance")

    # Check if all groups have sufficient data
    group_sizes = [len(g) for g in groups_clean]
    if any(size < 2 for size in group_sizes):
        logger.warning(f"Some groups have insufficient data: {group_sizes}")
        return {
            'test_name': method,
            'statistic': None,
            'pvalue': None,
            'equal_variances': False,
            'recommendation': 'Use robust methods (insufficient data in some groups)',
            'group_sizes': group_sizes,
            'group_variances': None
        }

    # Perform homogeneity test
    if method == 'levene':
        statistic, pvalue = stats.levene(*groups_clean)
        test_name = "Levene's test"
    elif method == 'bartlett':
        statistic, pvalue = stats.bartlett(*groups_clean)
        test_name = "Bartlett's test"
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'levene' or 'bartlett'")

    equal_variances = pvalue > alpha
    group_variances = [np.var(g, ddof=1) for g in groups_clean]

    return {
        'test_name': test_name,
        'statistic': statistic,
        'pvalue': pvalue,
        'equal_variances': equal_variances,
        'recommendation': 'Use standard ANOVA' if equal_variances else "Use Welch's ANOVA or non-parametric methods",
        'group_sizes': group_sizes,
        'group_variances': group_variances
    }


def validate_sample_size(groups: Union[List[Union[np.ndarray, pd.Series]], np.ndarray, pd.Series],
                        test_type: str = 'anova',
                        min_size: int = 30) -> Dict[str, Any]:
    """
    Validate sample size requirements for statistical tests.

    Recommended minimum sizes:
    - ANOVA: 30 per group for CLT to apply
    - t-test: 30 per group (or 20 if normality holds)
    - Chi-square: 5 expected counts per cell
    - Correlation: 20+ for reliable estimates

    Args:
        groups: Single array or list of arrays representing groups
        test_type: Type of test ('anova', 'ttest', 'chi2', 'correlation')
        min_size: Minimum recommended sample size

    Returns:
        Dictionary containing:
        - sufficient: Boolean indicating if sample size is adequate
        - actual_sizes: List of actual sample sizes
        - recommended_min: Recommended minimum size
        - warning: Warning message if insufficient (None otherwise)

    Example:
        >>> group1 = np.random.normal(0, 1, 50)
        >>> group2 = np.random.normal(0, 1, 50)
        >>> result = validate_sample_size([group1, group2], test_type='anova')
        >>> print(result['sufficient'])
        True
    """
    # Handle single array case
    if not isinstance(groups, list):
        groups = [groups]

    # Get sizes (after removing NaN)
    actual_sizes = [len(np.asarray(g)[~np.isnan(np.asarray(g))]) for g in groups]

    # Determine recommended minimum based on test type
    if test_type in ['anova', 'ttest']:
        recommended_min = min_size
        sufficient = all(size >= recommended_min for size in actual_sizes)
        warning_msg = f"Some groups below recommended n={recommended_min}: {actual_sizes}" if not sufficient else None
    elif test_type == 'chi2':
        # For chi-square, min_size refers to expected cell frequency
        recommended_min = min_size
        total_n = sum(actual_sizes)
        sufficient = total_n >= recommended_min * len(groups)
        warning_msg = f"Total sample size {total_n} may be insufficient for chi-square" if not sufficient else None
    elif test_type == 'correlation':
        recommended_min = min_size
        n = actual_sizes[0] if actual_sizes else 0
        sufficient = n >= recommended_min
        warning_msg = f"Sample size {n} below recommended n={recommended_min}" if not sufficient else None
    else:
        recommended_min = min_size
        sufficient = all(size >= recommended_min for size in actual_sizes)
        warning_msg = f"Some groups below recommended n={recommended_min}" if not sufficient else None

    return {
        'sufficient': sufficient,
        'actual_sizes': actual_sizes,
        'recommended_min': recommended_min,
        'warning': warning_msg
    }


def check_chi2_expected_frequencies(contingency_table: Union[np.ndarray, pd.DataFrame],
                                    min_expected: float = 5.0) -> Dict[str, Any]:
    """
    Check if chi-square test is valid based on expected cell frequencies.

    Chi-square test requires expected frequencies ≥5 in at least 80% of cells.
    For 2x2 tables with small samples, Fisher's exact test is recommended.

    Args:
        contingency_table: Contingency table (2D array or DataFrame)
        min_expected: Minimum expected frequency (default 5.0)

    Returns:
        Dictionary containing:
        - valid: Boolean indicating if chi-square is valid
        - min_expected_freq: Minimum expected frequency found
        - percent_cells_below_threshold: Percentage of cells below min_expected
        - recommendation: Suggested test to use
        - expected_frequencies: Full table of expected frequencies

    Example:
        >>> table = np.array([[10, 15], [20, 25]])
        >>> result = check_chi2_expected_frequencies(table)
        >>> print(result['valid'])
        True
    """
    # Convert to numpy array
    table = np.asarray(contingency_table)

    if table.ndim != 2:
        raise ValueError("Contingency table must be 2-dimensional")

    # Calculate expected frequencies
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    total = table.sum()

    expected = (row_totals @ col_totals) / total

    # Check validity
    below_threshold = expected < min_expected
    percent_below = 100 * below_threshold.sum() / expected.size
    min_expected_freq = expected.min()

    # Chi-square is valid if <20% of cells have expected frequency <5
    valid = percent_below < 20

    # Special case: 2x2 table with small samples
    is_2x2 = table.shape == (2, 2)
    if is_2x2 and min_expected_freq < 5:
        recommendation = "Use Fisher's exact test (2x2 table with low expected frequencies)"
    elif valid:
        recommendation = "Chi-square test is appropriate"
    else:
        recommendation = f"Chi-square may be invalid ({percent_below:.1f}% of cells below {min_expected}); consider Fisher's exact test or combining categories"

    return {
        'valid': valid,
        'min_expected_freq': min_expected_freq,
        'percent_cells_below_threshold': percent_below,
        'recommendation': recommendation,
        'expected_frequencies': expected
    }


# =====================================================
# EFFECT SIZE CALCULATIONS
# =====================================================

def cohens_d(group1: Union[np.ndarray, pd.Series],
             group2: Union[np.ndarray, pd.Series],
             pooled: bool = True) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size for difference between two groups.

    Cohen's d interpretation:
    - Small: d = 0.2
    - Medium: d = 0.5
    - Large: d = 0.8

    Args:
        group1: First group data
        group2: Second group data
        pooled: Use pooled standard deviation (default True)

    Returns:
        Dictionary containing:
        - cohens_d: Cohen's d value
        - interpretation: Effect size interpretation ('negligible', 'small', 'medium', 'large')
        - description: Detailed interpretation
        - mean_diff: Mean difference between groups
        - pooled_std: Pooled standard deviation (if pooled=True)

    Example:
        >>> group1 = np.random.normal(0, 1, 100)
        >>> group2 = np.random.normal(0.5, 1, 100)
        >>> result = cohens_d(group1, group2)
        >>> print(result['interpretation'])
        'medium'
    """
    # Clean data
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)
    g1_clean = g1[~np.isnan(g1)]
    g2_clean = g2[~np.isnan(g2)]

    # Calculate means
    mean1 = np.mean(g1_clean)
    mean2 = np.mean(g2_clean)
    mean_diff = mean1 - mean2

    # Calculate standard deviation
    if pooled:
        n1 = len(g1_clean)
        n2 = len(g2_clean)
        var1 = np.var(g1_clean, ddof=1)
        var2 = np.var(g2_clean, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        denominator = pooled_std
    else:
        # Use SD of group 1 (control group)
        denominator = np.std(g1_clean, ddof=1)

    # Calculate Cohen's d
    if denominator == 0:
        logger.warning("Zero standard deviation, cannot calculate Cohen's d")
        d = np.nan
    else:
        d = mean_diff / denominator

    # Interpret effect size
    abs_d = abs(d)
    if np.isnan(abs_d):
        interpretation = 'undefined'
        description = "Cannot interpret (zero variance)"
    elif abs_d < 0.2:
        interpretation = 'negligible'
        description = "Negligible effect size (d < 0.2)"
    elif abs_d < 0.5:
        interpretation = 'small'
        description = "Small effect size (0.2 ≤ d < 0.5)"
    elif abs_d < 0.8:
        interpretation = 'medium'
        description = "Medium effect size (0.5 ≤ d < 0.8)"
    else:
        interpretation = 'large'
        description = "Large effect size (d ≥ 0.8)"

    return {
        'cohens_d': d,
        'interpretation': interpretation,
        'description': description,
        'mean_diff': mean_diff,
        'pooled_std': denominator if pooled else None
    }


def eta_squared(groups: List[Union[np.ndarray, pd.Series]],
                f_statistic: Optional[float] = None,
                df_between: Optional[int] = None,
                df_within: Optional[int] = None) -> Dict[str, Any]:
    """
    Calculate eta-squared (η²) effect size for ANOVA.

    Eta-squared represents the proportion of variance explained by group membership.

    Interpretation:
    - Small: η² = 0.01 (1% of variance)
    - Medium: η² = 0.06 (6% of variance)
    - Large: η² = 0.14 (14% of variance)

    Args:
        groups: List of group data (alternative to providing F-statistic)
        f_statistic: F-statistic from ANOVA (optional)
        df_between: Between-groups degrees of freedom (optional)
        df_within: Within-groups degrees of freedom (optional)

    Returns:
        Dictionary containing:
        - eta_squared: η² value
        - interpretation: Effect size interpretation
        - percent_variance_explained: Percentage of variance explained
        - description: Detailed interpretation

    Example:
        >>> group1 = np.random.normal(0, 1, 100)
        >>> group2 = np.random.normal(0.5, 1, 100)
        >>> result = eta_squared([group1, group2])
        >>> print(result['percent_variance_explained'])
        6.2
    """
    # Calculate from raw data if groups provided
    if groups is not None:
        groups_clean = [np.asarray(g)[~np.isnan(np.asarray(g))] for g in groups]
        all_data = np.concatenate(groups_clean)

        # Calculate SS_between and SS_total
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_clean)
        ss_total = np.sum((all_data - grand_mean) ** 2)

        # Calculate eta-squared
        if ss_total == 0:
            eta_sq = np.nan
        else:
            eta_sq = ss_between / ss_total
    elif f_statistic is not None and df_between is not None and df_within is not None:
        # Calculate from F-statistic
        eta_sq = (df_between * f_statistic) / (df_between * f_statistic + df_within)
    else:
        raise ValueError("Must provide either 'groups' or all of (f_statistic, df_between, df_within)")

    # Interpret effect size
    if np.isnan(eta_sq):
        interpretation = 'undefined'
        description = "Cannot interpret (zero variance)"
        percent_var = 0.0
    elif eta_sq < 0.01:
        interpretation = 'negligible'
        description = "Negligible effect size (η² < 0.01)"
        percent_var = eta_sq * 100
    elif eta_sq < 0.06:
        interpretation = 'small'
        description = "Small effect size (0.01 ≤ η² < 0.06)"
        percent_var = eta_sq * 100
    elif eta_sq < 0.14:
        interpretation = 'medium'
        description = "Medium effect size (0.06 ≤ η² < 0.14)"
        percent_var = eta_sq * 100
    else:
        interpretation = 'large'
        description = "Large effect size (η² ≥ 0.14)"
        percent_var = eta_sq * 100

    return {
        'eta_squared': eta_sq,
        'interpretation': interpretation,
        'percent_variance_explained': percent_var,
        'description': description
    }


def cramers_v(contingency_table: Union[np.ndarray, pd.DataFrame],
              correction: bool = True) -> Dict[str, Any]:
    """
    Calculate Cramér's V effect size for chi-square test.

    Cramér's V is a measure of association between two categorical variables.

    Interpretation (for df* = min(rows-1, cols-1)):
    - df* = 1: Small (0.10), Medium (0.30), Large (0.50)
    - df* = 2: Small (0.07), Medium (0.21), Large (0.35)
    - df* = 3+: Small (0.06), Medium (0.17), Large (0.29)

    Args:
        contingency_table: Contingency table (2D array or DataFrame)
        correction: Apply bias correction for small samples (default True)

    Returns:
        Dictionary containing:
        - cramers_v: Cramér's V value
        - interpretation: Effect size interpretation
        - description: Detailed interpretation
        - chi2_statistic: Chi-square statistic
        - pvalue: P-value from chi-square test

    Example:
        >>> table = np.array([[10, 15], [20, 25]])
        >>> result = cramers_v(table)
        >>> print(result['interpretation'])
        'small'
    """
    # Convert to numpy array
    table = np.asarray(contingency_table)

    if table.ndim != 2:
        raise ValueError("Contingency table must be 2-dimensional")

    # Perform chi-square test
    chi2_stat, pvalue, dof, expected = stats.chi2_contingency(table, correction=correction)

    # Calculate Cramér's V
    n = table.sum()
    min_dim = min(table.shape[0], table.shape[1]) - 1

    if n == 0 or min_dim == 0:
        v = np.nan
    else:
        v = np.sqrt(chi2_stat / (n * min_dim))

    # Interpret based on degrees of freedom
    if np.isnan(v):
        interpretation = 'undefined'
        description = "Cannot interpret (invalid table)"
    elif min_dim == 1:
        if v < 0.10:
            interpretation = 'negligible'
        elif v < 0.30:
            interpretation = 'small'
        elif v < 0.50:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        description = f"{interpretation.capitalize()} effect size for 2x2 table (V = {v:.3f})"
    elif min_dim == 2:
        if v < 0.07:
            interpretation = 'negligible'
        elif v < 0.21:
            interpretation = 'small'
        elif v < 0.35:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        description = f"{interpretation.capitalize()} effect size for 3xK table (V = {v:.3f})"
    else:
        if v < 0.06:
            interpretation = 'negligible'
        elif v < 0.17:
            interpretation = 'small'
        elif v < 0.29:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        description = f"{interpretation.capitalize()} effect size for large table (V = {v:.3f})"

    return {
        'cramers_v': v,
        'interpretation': interpretation,
        'description': description,
        'chi2_statistic': chi2_stat,
        'pvalue': pvalue
    }


def pearson_r_to_d(r: float) -> float:
    """
    Convert Pearson correlation r to Cohen's d for effect size comparison.

    Args:
        r: Pearson correlation coefficient

    Returns:
        Cohen's d equivalent

    Example:
        >>> r = 0.5
        >>> d = pearson_r_to_d(r)
        >>> print(f"r={r} is approximately d={d:.2f}")
        r=0.5 is approximately d=1.15
    """
    if abs(r) >= 1:
        return np.inf if r > 0 else -np.inf
    return 2 * r / np.sqrt(1 - r ** 2)


# =====================================================
# MULTIPLE TESTING CORRECTION
# =====================================================

def apply_multiple_testing_correction(pvalues: Union[List[float], np.ndarray],
                                      method: str = 'fdr_bh',
                                      alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply multiple testing correction to p-values.

    Available methods:
    - 'bonferroni': Bonferroni correction (most conservative)
    - 'holm': Holm-Bonferroni (less conservative)
    - 'fdr_bh': Benjamini-Hochberg FDR control (recommended for exploratory analysis)
    - 'fdr_by': Benjamini-Yekutieli FDR control (for dependent tests)

    Args:
        pvalues: Array of p-values to correct
        method: Correction method (default 'fdr_bh')
        alpha: Family-wise error rate (default 0.05)

    Returns:
        Dictionary containing:
        - corrected_pvalues: Corrected p-values
        - reject: Boolean array indicating significance after correction
        - method: Correction method used
        - alpha: Significance level used
        - num_significant_raw: Number significant before correction
        - num_significant_corrected: Number significant after correction

    Example:
        >>> pvalues = [0.01, 0.04, 0.03, 0.50]
        >>> result = apply_multiple_testing_correction(pvalues)
        >>> print(result['reject'])
        [True True True False]
    """
    pvalues_array = np.asarray(pvalues)

    # Validate p-values
    if np.any((pvalues_array < 0) | (pvalues_array > 1)):
        raise ValueError("All p-values must be between 0 and 1")

    # Apply correction using statsmodels
    reject, corrected_pvals, alpha_sidak, alpha_bonf = multipletests(
        pvalues_array,
        alpha=alpha,
        method=method,
        is_sorted=False,
        returnsorted=False
    )

    # Count significant tests
    num_sig_raw = np.sum(pvalues_array < alpha)
    num_sig_corrected = np.sum(reject)

    return {
        'corrected_pvalues': corrected_pvals,
        'reject': reject,
        'method': method,
        'alpha': alpha,
        'num_significant_raw': int(num_sig_raw),
        'num_significant_corrected': int(num_sig_corrected)
    }


# =====================================================
# CONFIDENCE INTERVAL UTILITIES
# =====================================================

def calculate_mean_ci(data: Union[np.ndarray, pd.Series],
                     confidence: float = 0.95) -> Dict[str, Any]:
    """
    Calculate confidence interval for the mean using t-distribution.

    Args:
        data: Data array
        confidence: Confidence level (default 0.95)

    Returns:
        Dictionary containing:
        - mean: Sample mean
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - confidence_level: Confidence level used
        - margin_of_error: Margin of error
        - standard_error: Standard error of the mean

    Example:
        >>> data = np.random.normal(100, 15, 50)
        >>> result = calculate_mean_ci(data, confidence=0.95)
        >>> print(f"Mean: {result['mean']:.2f} [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
        Mean: 99.23 [95.12, 103.34]
    """
    # Clean data
    data_clean = np.asarray(data)
    data_clean = data_clean[~np.isnan(data_clean)]

    n = len(data_clean)
    if n < 2:
        return {
            'mean': np.mean(data_clean) if n == 1 else np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'confidence_level': confidence,
            'margin_of_error': np.nan,
            'standard_error': np.nan
        }

    # Calculate statistics
    mean = np.mean(data_clean)
    se = stats.sem(data_clean)  # Standard error
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    ci_lower = mean - margin
    ci_upper = mean + margin

    return {
        'mean': mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence,
        'margin_of_error': margin,
        'standard_error': se
    }


def calculate_correlation_ci(r: float,
                             n: int,
                             confidence: float = 0.95) -> Dict[str, Any]:
    """
    Calculate confidence interval for Pearson correlation using Fisher Z-transformation.

    Args:
        r: Pearson correlation coefficient
        n: Sample size
        confidence: Confidence level (default 0.95)

    Returns:
        Dictionary containing:
        - r: Correlation coefficient
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - confidence_level: Confidence level used

    Example:
        >>> result = calculate_correlation_ci(r=0.5, n=100)
        >>> print(f"r = {result['r']:.2f} [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
        r = 0.50 [0.33, 0.64]
    """
    if abs(r) >= 1:
        logger.warning(f"Correlation |r| >= 1 (r={r}), CI may be unreliable")

    if n < 4:
        return {
            'r': r,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'confidence_level': confidence
        }

    # Fisher Z-transformation
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + confidence) / 2)

    # CI in Z-space
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    # Transform back to r-space
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)

    return {
        'r': r,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence
    }


def bootstrap_ci(data: Union[np.ndarray, pd.Series],
                statistic_func: Callable[[np.ndarray], float],
                n_bootstrap: int = 1000,
                confidence: float = 0.95,
                random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence interval for any statistic.

    Uses the percentile method: CI is the (alpha/2, 1-alpha/2) percentiles
    of the bootstrap distribution.

    Args:
        data: Data array
        statistic_func: Function that takes an array and returns a scalar statistic
                       (e.g., np.median, np.std, custom function)
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence: Confidence level (default 0.95)
        random_state: Random seed for reproducibility (optional)

    Returns:
        Dictionary containing:
        - statistic: Observed statistic value
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - confidence_level: Confidence level used
        - bootstrap_distribution: Array of bootstrap statistics

    Example:
        >>> data = np.random.normal(100, 15, 50)
        >>> result = bootstrap_ci(data, statistic_func=np.median, n_bootstrap=1000)
        >>> print(f"Median: {result['statistic']:.2f} [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
        Median: 99.50 [95.20, 103.80]
    """
    # Clean data
    data_clean = np.asarray(data)
    data_clean = data_clean[~np.isnan(data_clean)]

    if len(data_clean) < 2:
        stat_value = statistic_func(data_clean) if len(data_clean) == 1 else np.nan
        return {
            'statistic': stat_value,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'confidence_level': confidence,
            'bootstrap_distribution': np.array([])
        }

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate observed statistic
    observed_stat = statistic_func(data_clean)

    # Bootstrap
    n = len(data_clean)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data_clean, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(sample)

    # Calculate percentile CI
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'statistic': observed_stat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence,
        'bootstrap_distribution': bootstrap_stats
    }
