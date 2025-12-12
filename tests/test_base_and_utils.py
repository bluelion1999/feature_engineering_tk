"""
Tests for base classes and utility functions.

Tests the foundational infrastructure introduced in v2.3.0:
- FeatureEngineeringBase class
- Utility functions in utils module
"""

import pytest
import pandas as pd
import numpy as np
from feature_engineering_tk.base import FeatureEngineeringBase
from feature_engineering_tk.utils import (
    validate_and_copy_dataframe,
    validate_columns,
    get_numeric_columns,
    validate_numeric_columns,
    get_string_columns,
    get_feature_columns
)


class TestFeatureEngineeringBase:
    """Test suite for FeatureEngineeringBase class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'text': ['hello', 'world', 'foo', 'bar', 'baz']
        })

    def test_initialization_with_valid_dataframe(self, sample_df):
        """Test initialization with valid DataFrame."""
        base = FeatureEngineeringBase(sample_df)
        assert base.df.shape == sample_df.shape
        assert base.df is not sample_df  # Should be a copy, not same object

    def test_initialization_with_empty_dataframe(self):
        """Test initialization with empty DataFrame."""
        empty_df = pd.DataFrame()
        base = FeatureEngineeringBase(empty_df)
        assert base.df.empty

    def test_initialization_with_invalid_input(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            FeatureEngineeringBase([1, 2, 3])

        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            FeatureEngineeringBase({"a": [1, 2, 3]})

    def test_get_dataframe_returns_copy(self, sample_df):
        """Test that get_dataframe returns a copy, not reference."""
        base = FeatureEngineeringBase(sample_df)
        df_copy = base.get_dataframe()

        # Modify the copy
        df_copy['new_col'] = [99, 99, 99, 99, 99]

        # Original should be unchanged
        assert 'new_col' not in base.df.columns


class TestUtilityFunctions:
    """Test suite for utility functions."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'object_col': ['x', 'y', 'z', 'w', 'v']
        })

    def test_validate_and_copy_dataframe_valid(self, sample_df):
        """Test validation with valid DataFrame."""
        result = validate_and_copy_dataframe(sample_df)
        assert result.shape == sample_df.shape
        assert result is not sample_df  # Should be copy

    def test_validate_and_copy_dataframe_empty(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = validate_and_copy_dataframe(empty_df)
        assert result.empty

    def test_validate_and_copy_dataframe_invalid(self):
        """Test validation with invalid input."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            validate_and_copy_dataframe([1, 2, 3])

    def test_validate_columns_valid_single_string(self, sample_df):
        """Test validation with single column as string."""
        result = validate_columns(sample_df, 'int_col')
        assert result == ['int_col']

    def test_validate_columns_valid_list(self, sample_df):
        """Test validation with list of valid columns."""
        result = validate_columns(sample_df, ['int_col', 'float_col'])
        assert result == ['int_col', 'float_col']

    def test_validate_columns_missing_warning(self, sample_df):
        """Test that missing columns log warning but continue."""
        result = validate_columns(sample_df, ['int_col', 'missing_col'])
        assert result == ['int_col']

    def test_validate_columns_missing_raise(self, sample_df):
        """Test that raise_on_missing raises error."""
        with pytest.raises(ValueError, match="not found"):
            validate_columns(sample_df, ['missing_col'], raise_on_missing=True)

    def test_validate_columns_invalid_type(self, sample_df):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="must be a string or list"):
            validate_columns(sample_df, 123)

    def test_get_numeric_columns_all(self, sample_df):
        """Test getting all numeric columns."""
        result = get_numeric_columns(sample_df)
        assert set(result) == {'int_col', 'float_col'}

    def test_get_numeric_columns_subset(self, sample_df):
        """Test getting numeric columns from subset."""
        result = get_numeric_columns(sample_df, ['int_col', 'string_col'])
        assert result == ['int_col']

    def test_validate_numeric_columns_valid(self, sample_df):
        """Test validation with numeric columns."""
        result = validate_numeric_columns(sample_df, ['int_col', 'float_col'])
        assert set(result) == {'int_col', 'float_col'}

    def test_validate_numeric_columns_mixed(self, sample_df):
        """Test validation filters out non-numeric."""
        result = validate_numeric_columns(sample_df, ['int_col', 'string_col'])
        assert result == ['int_col']

    def test_get_string_columns_all(self, sample_df):
        """Test getting all string columns."""
        result = get_string_columns(sample_df)
        assert set(result) == {'string_col', 'object_col'}

    def test_get_string_columns_subset(self, sample_df):
        """Test getting string columns from subset."""
        result = get_string_columns(sample_df, ['string_col', 'int_col'])
        assert result == ['string_col']

    def test_get_feature_columns_all_numeric(self, sample_df):
        """Test getting all numeric feature columns."""
        result = get_feature_columns(sample_df)
        assert set(result) == {'int_col', 'float_col'}

    def test_get_feature_columns_exclude_target(self, sample_df):
        """Test excluding target column."""
        result = get_feature_columns(sample_df, exclude_columns=['int_col'])
        assert result == ['float_col']

    def test_get_feature_columns_all_types(self, sample_df):
        """Test getting all column types."""
        result = get_feature_columns(sample_df, numeric_only=False)
        assert len(result) == 4

    def test_get_feature_columns_exclude_multiple(self, sample_df):
        """Test excluding multiple columns."""
        result = get_feature_columns(
            sample_df,
            exclude_columns=['int_col', 'string_col'],
            numeric_only=False
        )
        assert set(result) == {'float_col', 'object_col'}
