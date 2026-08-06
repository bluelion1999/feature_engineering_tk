"""
Tests for data_analysis module.

Tests focus on division by zero fix in z-score outlier detection.
"""

import pytest
import pandas as pd
import numpy as np
from feature_engineering_tk import DataAnalyzer


class TestDataAnalyzer:
    """Test suite for DataAnalyzer class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'constant': [100, 100, 100, 100, 100]  # Zero std dev
        })

    def test_initialization(self, sample_df):
        """Test that DataAnalyzer initializes correctly."""
        analyzer = DataAnalyzer(sample_df)
        assert analyzer.df.equals(sample_df)
        assert analyzer.df is not sample_df

    def test_zscore_outliers_with_constant_column(self, sample_df):
        """Test that z-score outlier detection handles zero std dev (CRITICAL FIX)."""
        analyzer = DataAnalyzer(sample_df)

        # This would crash before with division by zero
        outliers = analyzer.detect_outliers_zscore(
            columns=['constant', 'numeric1'],
            threshold=2.0
        )

        # Constant column should be skipped (not in results)
        assert 'constant' not in outliers
        # But normal column should work
        # (numeric1 has no outliers with threshold=2.0, so might be empty)

    def test_zscore_outliers_all_constant(self):
        """Test z-score with all constant columns."""
        df = pd.DataFrame({
            'const1': [5, 5, 5, 5],
            'const2': [10, 10, 10, 10]
        })

        analyzer = DataAnalyzer(df)

        # Should not crash
        outliers = analyzer.detect_outliers_zscore()

        # Should return empty dict
        assert len(outliers) == 0

    def test_basic_info(self, sample_df):
        """Test get_basic_info method."""
        analyzer = DataAnalyzer(sample_df)
        info = analyzer.get_basic_info()

        assert info['shape'][0] == (5, 4)
        assert len(info['columns'][0]) == 4
        assert 'duplicates' in info
        assert 'memory_usage_mb' in info

    def test_correlation_matrix(self, sample_df):
        """Test correlation matrix generation."""
        analyzer = DataAnalyzer(sample_df)
        corr = analyzer.get_correlation_matrix()

        assert not corr.empty
        assert corr.shape[0] == corr.shape[1]  # Should be square

    def test_high_correlations(self, sample_df):
        """Test high correlation detection."""
        analyzer = DataAnalyzer(sample_df)
        high_corr = analyzer.get_high_correlations(threshold=0.7)

        # Should return DataFrame
        assert isinstance(high_corr, pd.DataFrame)

    def test_calculate_vif(self):
        """Test VIF calculation in DataAnalyzer.

        Uses non-zero-mean data (unlike plain np.random.randn(), which is
        mean ~0 and happens to mask a missing-intercept bug in the VIF
        design matrix). feature3 is deliberately built as a near-perfect
        linear combination of feature1, so it should show very high VIF,
        while feature1/feature2 are independent and should show VIF close
        to 1.
        """
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'feature1': np.random.uniform(50, 150, n),
            'feature2': np.random.uniform(500, 1500, n),
        })
        # Create highly correlated feature (near-perfect linear combination)
        df['feature3'] = df['feature1'] * 0.9 + np.random.randn(n) * 0.1

        analyzer = DataAnalyzer(df)
        vif_df = analyzer.calculate_vif()

        # Should return DataFrame with VIF values
        assert not vif_df.empty
        assert 'feature' in vif_df.columns
        assert 'VIF' in vif_df.columns
        assert len(vif_df) == 3  # All three features

        vif_by_feature = vif_df.set_index('feature')['VIF']

        # feature3 is (almost) collinear with feature1 -> very high VIF
        assert vif_by_feature['feature1'] > 50
        assert vif_by_feature['feature3'] > 50

        # feature2 is independent of the others -> VIF should be close to 1,
        # not inflated by the missing-intercept bug (which produced values
        # in the hundreds/thousands for non-zero-mean data).
        assert vif_by_feature['feature2'] == pytest.approx(1.0, abs=1.0)

    def test_calculate_vif_matches_reference_with_intercept(self):
        """Regression test for the missing-intercept VIF bug.

        variance_inflation_factor() assumes the design matrix already
        includes a constant/intercept column; VIF is only statistically
        meaningful relative to a regression that includes one. This test
        builds a pure-noise column with a large, non-zero mean that is
        uncorrelated with the other features (true VIF ~1.0) and compares
        DataAnalyzer.calculate_vif() directly against a reference value
        computed the correct way (sm.add_constant + variance_inflation_factor).
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        np.random.seed(0)
        n = 500
        x1 = np.random.uniform(50, 150, n)
        x2 = x1 * 2 + np.random.normal(0, 5, n) + 1000  # correlated w/ x1, large offset
        x3 = np.random.normal(500, 50, n)  # pure noise, uncorrelated, large mean

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

        # Reference VIF computed the statistically correct way
        df_const = add_constant(df, has_constant='add')
        reference_vif = {
            col: variance_inflation_factor(df_const.values, i + 1)
            for i, col in enumerate(df.columns)
        }

        analyzer = DataAnalyzer(df)
        vif_df = analyzer.calculate_vif()
        vif_by_feature = vif_df.set_index('feature')['VIF']

        for col in df.columns:
            assert vif_by_feature[col] == pytest.approx(reference_vif[col], rel=1e-6)

        # Sanity check on the statistical meaning: x3 is pure, uncorrelated
        # noise, so its true VIF should be close to 1 (not inflated into the
        # hundreds, as the missing-intercept bug produced).
        assert vif_by_feature['x3'] == pytest.approx(1.0, abs=1.0)

    def test_calculate_vif_insufficient_columns(self):
        """Test VIF with insufficient columns."""
        df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        analyzer = DataAnalyzer(df)
        vif_df = analyzer.calculate_vif()

        # Should return empty DataFrame
        assert vif_df.empty

    def test_detect_misclassified_categorical_binary(self):
        """Test detection of binary/flag columns."""
        np.random.seed(42)
        df = pd.DataFrame({
            'binary_flag': [0, 1, 0, 1, 0, 1] * 5,
            'low_cardinality': [1, 2, 3, 1, 2, 3] * 5,
            'normal_numeric': np.random.randn(30) * 100  # Many unique values
        })
        analyzer = DataAnalyzer(df)
        misclassified = analyzer.detect_misclassified_categorical()

        # Should detect binary_flag and low_cardinality
        assert not misclassified.empty
        assert 'binary_flag' in misclassified['column'].values
        assert 'low_cardinality' in misclassified['column'].values
        assert 'normal_numeric' not in misclassified['column'].values

        # Check binary flag has 2 unique values
        binary_row = misclassified[misclassified['column'] == 'binary_flag'].iloc[0]
        assert binary_row['unique_count'] == 2
        assert 'Binary flag' in binary_row['suggestion']

    def test_detect_misclassified_categorical_integer_column(self):
        """Test detection of integer columns with low cardinality."""
        df = pd.DataFrame({
            'rating': [1, 2, 3, 4, 5] * 10,  # 5 unique integers
            'continuous': np.linspace(0, 100, 50)  # 50 unique values
        })
        analyzer = DataAnalyzer(df)
        misclassified = analyzer.detect_misclassified_categorical()

        # Should detect rating as likely categorical
        assert 'rating' in misclassified['column'].values
        assert 'continuous' not in misclassified['column'].values

    def test_detect_misclassified_categorical_nullable_integer_dtype(self):
        """Regression test: pandas nullable Int64 columns should be detected too.

        The integer-column branch used to check
        `col_data.dtype in ['int64', 'int32', ...]`, a string comparison
        that never matches pandas' nullable IntegerDtype objects (e.g.
        pd.Int64Dtype()), silently skipping this detection path for
        nullable-typed columns. Fixed to use pd.api.types.is_integer_dtype().
        """
        # 15 unique values over 150 rows: unique_count (15) is above the
        # max_unique (10) branch and unique_ratio (0.1) is above
        # min_unique_ratio (0.05), so only the nullable-integer branch
        # can flag this column.
        df = pd.DataFrame({
            'nullable_rating': pd.array(list(range(15)) * 10, dtype='Int64')
        })
        analyzer = DataAnalyzer(df)
        misclassified = analyzer.detect_misclassified_categorical()

        assert 'nullable_rating' in misclassified['column'].values
        row = misclassified[misclassified['column'] == 'nullable_rating'].iloc[0]
        assert 'Integer column' in row['suggestion']

    def test_detect_misclassified_categorical_low_unique_ratio(self):
        """Test detection based on low unique ratio."""
        # Create column with many repeated values
        df = pd.DataFrame({
            'repeated': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3] * 20  # Only 3 unique in 200 rows
        })
        analyzer = DataAnalyzer(df)
        misclassified = analyzer.detect_misclassified_categorical(min_unique_ratio=0.05)

        # Should detect due to low unique ratio (3/200 = 1.5%)
        assert 'repeated' in misclassified['column'].values

    def test_detect_misclassified_categorical_no_numeric_columns(self):
        """Test with no numeric columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        analyzer = DataAnalyzer(df)
        misclassified = analyzer.detect_misclassified_categorical()

        # Should return empty DataFrame
        assert misclassified.empty

    def test_suggest_binning_skewed_distribution(self):
        """Test binning suggestion for skewed distribution."""
        np.random.seed(42)
        # Create right-skewed data
        df = pd.DataFrame({
            'skewed': np.random.exponential(scale=2.0, size=100)
        })
        analyzer = DataAnalyzer(df)
        binning = analyzer.suggest_binning()

        # Should suggest quantile binning for skewed data
        assert not binning.empty
        assert 'skewed' in binning['column'].values
        skewed_row = binning[binning['column'] == 'skewed'].iloc[0]
        assert skewed_row['strategy'] == 'quantile'
        assert 'skewed' in skewed_row['reason'].lower()

    def test_suggest_binning_uniform_distribution(self):
        """Test binning suggestion for uniform distribution."""
        np.random.seed(42)
        df = pd.DataFrame({
            'uniform': np.random.uniform(0, 100, size=100)
        })
        analyzer = DataAnalyzer(df)
        binning = analyzer.suggest_binning()

        # Should suggest uniform binning for uniform data
        assert not binning.empty
        assert 'uniform' in binning['column'].values
        uniform_row = binning[binning['column'] == 'uniform'].iloc[0]
        assert uniform_row['strategy'] == 'uniform'

    def test_suggest_binning_with_outliers(self):
        """Test binning suggestion for data with outliers."""
        np.random.seed(42)
        # Create data with enough unique values (>20) and outliers
        normal_data = np.random.normal(50, 10, 100).tolist()
        outliers = [200, 250]
        df = pd.DataFrame({
            'with_outliers': normal_data + outliers
        })
        analyzer = DataAnalyzer(df)
        binning = analyzer.suggest_binning()

        # Should suggest quantile binning (outliers create skewness, both handled by quantile)
        assert not binning.empty
        assert 'with_outliers' in binning['column'].values
        outlier_row = binning[binning['column'] == 'with_outliers'].iloc[0]
        assert outlier_row['strategy'] == 'quantile'
        # Outliers create skewness, so either reason is correct
        assert ('outlier' in outlier_row['reason'].lower() or
                'skew' in outlier_row['reason'].lower())

    def test_suggest_binning_min_unique_threshold(self):
        """Test that columns with few unique values are not suggested."""
        df = pd.DataFrame({
            'few_unique': [1, 2, 3, 4, 5] * 20,  # Only 5 unique values, 100 rows
            'many_unique': list(range(100))  # 100 unique values, 100 rows
        })
        analyzer = DataAnalyzer(df)
        binning = analyzer.suggest_binning(min_unique=20)

        # Should only suggest binning for many_unique
        assert 'many_unique' in binning['column'].values
        assert 'few_unique' not in binning['column'].values

    def test_suggest_binning_no_numeric_columns(self):
        """Test binning suggestion with no numeric columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 10,
            'cat2': ['X', 'Y', 'Z'] * 10
        })
        analyzer = DataAnalyzer(df)
        binning = analyzer.suggest_binning()

        # Should return empty DataFrame
        assert binning.empty

    def test_get_categorical_summary_with_all_nan_column(self):
        """Test categorical summary with column containing only NaN values (Bug #3).

        Bug #3: Lines 85-86 used unique_count > 0 check instead of checking
        if value_counts() is empty, which could cause IndexError in edge cases.
        """
        # Use dtype=object to ensure column is treated as categorical
        df = pd.DataFrame({
            'all_nan': pd.Series([np.nan, np.nan, np.nan], dtype=object),
            'normal': ['A', 'B', 'C']
        })
        analyzer = DataAnalyzer(df)

        # Should not crash with IndexError on empty value_counts
        result = analyzer.get_categorical_summary(max_unique=10)

        # All-NaN column should be handled gracefully (included with 0 values)
        all_nan_row = result[result['column'] == 'all_nan']
        assert len(all_nan_row) == 1, "All-NaN column should be included in summary"
        all_nan_info = all_nan_row.iloc[0]
        assert all_nan_info['unique_count'] == 0
        assert all_nan_info['top_value'] is None or pd.isna(all_nan_info['top_value'])
        assert all_nan_info['top_value_freq'] == 0
        assert all_nan_info['top_value_percent'] == 0

    def test_get_cardinality_info(self, sample_df):
        """Test get_cardinality_info with a normal, non-empty DataFrame (regression coverage)."""
        analyzer = DataAnalyzer(sample_df)
        cardinality = analyzer.get_cardinality_info()

        assert list(cardinality.columns) == ['column', 'unique_count', 'cardinality_ratio', 'dtype']
        assert len(cardinality) == len(sample_df.columns)

        # 'constant' column has 1 unique value out of 5 rows -> ratio 0.2
        constant_row = cardinality[cardinality['column'] == 'constant'].iloc[0]
        assert constant_row['unique_count'] == 1
        assert constant_row['cardinality_ratio'] == pytest.approx(0.2)

        # 'numeric1' is fully unique (5 unique values out of 5 rows) -> ratio 1.0
        numeric1_row = cardinality[cardinality['column'] == 'numeric1'].iloc[0]
        assert numeric1_row['unique_count'] == 5
        assert numeric1_row['cardinality_ratio'] == pytest.approx(1.0)

        # Sorted by unique_count descending
        assert cardinality['unique_count'].is_monotonic_decreasing

    def test_get_cardinality_info_empty_dataframe(self):
        """Test get_cardinality_info with 0 rows (CRITICAL FIX).

        Previously, unique_count / len(self.df) raised ZeroDivisionError
        when the DataFrame had columns but 0 rows. It should now return
        a correctly-shaped DataFrame with NaN cardinality_ratio instead
        of crashing.
        """
        df = pd.DataFrame({'a': pd.Series([], dtype='float64'),
                            'b': pd.Series([], dtype='object')})
        analyzer = DataAnalyzer(df)

        # Should not raise ZeroDivisionError
        result = analyzer.get_cardinality_info()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['column', 'unique_count', 'cardinality_ratio', 'dtype']
        assert len(result) == 2
        assert set(result['column']) == {'a', 'b'}
        assert (result['unique_count'] == 0).all()
        assert result['cardinality_ratio'].isna().all()

    def test_get_cardinality_info_empty_dataframe_no_columns(self):
        """Test get_cardinality_info with 0 rows AND 0 columns."""
        df = pd.DataFrame()
        analyzer = DataAnalyzer(df)

        result = analyzer.get_cardinality_info()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['column', 'unique_count', 'cardinality_ratio', 'dtype']
        assert len(result) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
