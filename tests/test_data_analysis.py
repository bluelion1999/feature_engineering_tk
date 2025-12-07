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

        assert info['shape'] == (5, 4)
        assert len(info['columns']) == 4
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
        """Test VIF calculation in DataAnalyzer."""
        # Create dataset with known multicollinearity
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
        # Create highly correlated feature
        df['feature3'] = df['feature1'] * 0.9 + np.random.randn(100) * 0.1

        analyzer = DataAnalyzer(df)
        vif_df = analyzer.calculate_vif()

        # Should return DataFrame with VIF values
        assert not vif_df.empty
        assert 'feature' in vif_df.columns
        assert 'VIF' in vif_df.columns
        assert len(vif_df) == 3  # All three features

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
