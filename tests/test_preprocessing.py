"""
Tests for preprocessing module.

Tests focus on critical bug fixes:
- Inplace operation correctness
- Division by zero handling
- Deprecated method replacements
- Input validation
"""

import pytest
import pandas as pd
import numpy as np
from feature_engineering_tk import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'with_nulls': [1.0, None, 3.0, None, 5.0]
        })

    def test_initialization(self, sample_df):
        """Test that DataPreprocessor initializes correctly."""
        preprocessor = DataPreprocessor(sample_df)
        assert preprocessor.df.equals(sample_df)
        assert preprocessor.df is not sample_df  # Should be a copy

    def test_initialization_with_empty_df(self):
        """Test initialization with empty DataFrame."""
        empty_df = pd.DataFrame()
        preprocessor = DataPreprocessor(empty_df)
        assert preprocessor.df.empty

    def test_initialization_with_invalid_input(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError):
            DataPreprocessor([1, 2, 3])

    def test_inplace_false_doesnt_modify_original(self, sample_df):
        """Test that inplace=False doesn't modify internal dataframe."""
        preprocessor = DataPreprocessor(sample_df)
        original_shape = preprocessor.df.shape

        # This was broken before - inplace=False would still modify self.df
        result = preprocessor.convert_dtypes({'categorical': 'category'}, inplace=False)

        # Original should be unchanged
        assert preprocessor.df['categorical'].dtype == 'object'
        # Result should be modified
        assert result['categorical'].dtype.name == 'category'

    def test_inplace_true_modifies_internal(self, sample_df):
        """Test that inplace=True correctly modifies internal dataframe."""
        preprocessor = DataPreprocessor(sample_df)

        # This was broken before - self.df wasn't updated
        preprocessor.convert_dtypes({'categorical': 'category'}, inplace=True)

        assert preprocessor.df['categorical'].dtype.name == 'category'

    def test_clip_values_inplace(self, sample_df):
        """Test clip_values inplace operation (was broken)."""
        preprocessor = DataPreprocessor(sample_df)

        # Inplace=True should modify self.df
        preprocessor.clip_values('numeric1', lower=2, upper=4, inplace=True)
        assert preprocessor.df['numeric1'].min() == 2
        assert preprocessor.df['numeric1'].max() == 4

    def test_clip_values_not_inplace(self, sample_df):
        """Test clip_values non-inplace operation."""
        preprocessor = DataPreprocessor(sample_df)

        result = preprocessor.clip_values('numeric1', lower=2, upper=4, inplace=False)
        # Original unchanged
        assert preprocessor.df['numeric1'].min() == 1
        # Result modified
        assert result['numeric1'].min() == 2

    def test_apply_custom_function_inplace(self, sample_df):
        """Test apply_custom_function inplace operation (was broken)."""
        preprocessor = DataPreprocessor(sample_df)

        preprocessor.apply_custom_function('numeric1', lambda x: x * 2, inplace=True)
        assert preprocessor.df['numeric1'].tolist() == [2, 4, 6, 8, 10]

    def test_deprecated_fillna_methods_replaced(self, sample_df):
        """Test that deprecated fillna methods are replaced."""
        preprocessor = DataPreprocessor(sample_df)

        # These should use ffill() and bfill() instead of deprecated fillna(method=...)
        result_forward = preprocessor.handle_missing_values(
            strategy='forward_fill',
            columns=['with_nulls'],
            inplace=False
        )
        result_backward = preprocessor.handle_missing_values(
            strategy='backward_fill',
            columns=['with_nulls'],
            inplace=False
        )

        assert result_forward['with_nulls'].isna().sum() == 0
        assert result_backward['with_nulls'].isna().sum() == 0

    def test_handle_outliers_zscore_division_by_zero(self):
        """Test that z-score outlier detection handles zero std dev."""
        # Create dataframe with constant column
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 100, 5]
        })

        preprocessor = DataPreprocessor(df)

        # Should not crash with division by zero
        result = preprocessor.handle_outliers(
            columns=['constant', 'variable'],
            method='zscore',
            action='remove',
            inplace=False
        )

        # Constant column should be skipped, variable column processed
        assert 'constant' in result.columns
        assert len(result) < len(df)  # Some outliers should be removed from variable

    def test_input_validation(self, sample_df):
        """Test input validation for various methods."""
        preprocessor = DataPreprocessor(sample_df)

        # Invalid strategy
        with pytest.raises(ValueError):
            preprocessor.handle_missing_values(strategy='invalid')

        # Invalid keep parameter
        with pytest.raises(ValueError):
            preprocessor.remove_duplicates(keep='invalid')

        # Invalid method for handle_outliers
        with pytest.raises(ValueError):
            preprocessor.handle_outliers(columns=['numeric1'], method='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
