"""
Tests for custom exceptions.

Tests that custom exceptions are raised appropriately and contain
useful error messages.
"""

import pytest
import pandas as pd
from feature_engineering_tk import (
    DataPreprocessor,
    FeatureEngineer,
    InvalidStrategyError,
    InvalidMethodError,
    TransformerNotFittedError,
)


class TestCustomExceptions:
    """Test suite for custom exceptions."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })

    def test_invalid_strategy_error(self, sample_df):
        """Test that InvalidStrategyError is raised with helpful message."""
        preprocessor = DataPreprocessor(sample_df)

        with pytest.raises(InvalidStrategyError) as exc_info:
            preprocessor.handle_missing_values(strategy='invalid_strategy')

        error = exc_info.value
        assert error.strategy == 'invalid_strategy'
        assert 'invalid_strategy' in str(error)
        assert 'Valid strategies' in str(error)

    def test_invalid_method_error_outliers(self, sample_df):
        """Test that InvalidMethodError is raised for invalid outlier method."""
        preprocessor = DataPreprocessor(sample_df)

        with pytest.raises(InvalidMethodError) as exc_info:
            preprocessor.handle_outliers(columns=['numeric1'], method='invalid_method')

        error = exc_info.value
        assert error.method == 'invalid_method'
        assert 'iqr' in str(error)
        assert 'zscore' in str(error)

    def test_invalid_method_error_scaling(self, sample_df):
        """Test that InvalidMethodError is raised for invalid scaling method."""
        engineer = FeatureEngineer(sample_df)

        with pytest.raises(InvalidMethodError) as exc_info:
            engineer.scale_features(['numeric1'], method='invalid_scaler')

        error = exc_info.value
        assert error.method == 'invalid_scaler'
        assert 'standard' in str(error)
        assert 'minmax' in str(error)

    def test_transformer_not_fitted_error(self, sample_df):
        """Test that TransformerNotFittedError is raised when saving without fitting."""
        engineer = FeatureEngineer(sample_df)

        with pytest.raises(TransformerNotFittedError) as exc_info:
            engineer.save_transformers('transformers.joblib')

        error = exc_info.value
        assert 'encoder or scaler' in str(error)
        assert 'fitted' in str(error).lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
