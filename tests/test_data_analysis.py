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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
