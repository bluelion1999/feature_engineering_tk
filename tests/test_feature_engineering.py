"""
Tests for feature_engineering module.

Tests focus on critical bug fixes and new features:
- Inplace operation correctness
- Transformer persistence (save/load)
- Input validation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from feature_engineering_tk import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'date': pd.date_range('2024-01-01', periods=5)
        })

    def test_initialization(self, sample_df):
        """Test that FeatureEngineer initializes correctly."""
        engineer = FeatureEngineer(sample_df)
        assert engineer.df.equals(sample_df)
        assert engineer.df is not sample_df
        assert len(engineer.encoders) == 0
        assert len(engineer.scalers) == 0

    def test_inplace_false_label_encoding(self, sample_df):
        """Test label encoding with inplace=False (was broken)."""
        engineer = FeatureEngineer(sample_df)

        result = engineer.encode_categorical_label(['categorical'], inplace=False)

        # Original should be unchanged
        assert engineer.df['categorical'].dtype == 'object'
        # Result should be encoded
        assert pd.api.types.is_integer_dtype(result['categorical'])

    def test_inplace_true_label_encoding(self, sample_df):
        """Test label encoding with inplace=True (was broken)."""
        engineer = FeatureEngineer(sample_df)

        engineer.encode_categorical_label(['categorical'], inplace=True)

        # Internal df should be modified
        assert pd.api.types.is_integer_dtype(engineer.df['categorical'])
        # Encoder should be stored
        assert 'categorical_label' in engineer.encoders

    def test_scale_features_inplace(self, sample_df):
        """Test scaling with inplace=True (was broken)."""
        engineer = FeatureEngineer(sample_df)

        engineer.scale_features(['numeric1', 'numeric2'], method='standard', inplace=True)

        # Should be scaled (mean ~0, std ~1) - using ddof=0 for consistency with sklearn
        assert abs(engineer.df['numeric1'].mean()) < 1e-10
        # sklearn StandardScaler uses ddof=0, pandas uses ddof=1 by default
        assert abs(engineer.df['numeric1'].std(ddof=0) - 1.0) < 1e-10
        assert 'standard_scaler' in engineer.scalers

    def test_scale_features_not_inplace(self, sample_df):
        """Test scaling with inplace=False."""
        engineer = FeatureEngineer(sample_df)

        result = engineer.scale_features(['numeric1'], method='standard', inplace=False)

        # Original unchanged
        assert engineer.df['numeric1'].mean() == 3.0
        # Result scaled
        assert abs(result['numeric1'].mean()) < 1e-10

    def test_create_binning_inplace(self, sample_df):
        """Test binning with inplace=True (was broken)."""
        engineer = FeatureEngineer(sample_df)

        engineer.create_binning('numeric1', bins=3, inplace=True)

        assert 'numeric1_binned' in engineer.df.columns

    def test_create_log_transform_inplace(self, sample_df):
        """Test log transform with inplace=True (was broken)."""
        engineer = FeatureEngineer(sample_df)

        engineer.create_log_transform(['numeric1'], inplace=True)

        assert 'numeric1_log' in engineer.df.columns

    def test_create_sqrt_transform_inplace(self, sample_df):
        """Test sqrt transform with inplace=True (was broken)."""
        engineer = FeatureEngineer(sample_df)

        engineer.create_sqrt_transform(['numeric1'], inplace=True)

        assert 'numeric1_sqrt' in engineer.df.columns

    def test_ratio_features_configurable_epsilon(self, sample_df):
        """Test that ratio features use configurable epsilon."""
        engineer = FeatureEngineer(sample_df)

        # Create ratio with custom epsilon
        engineer.create_ratio_features(
            'numeric1',
            'numeric2',
            epsilon=1e-10,
            inplace=True
        )

        assert 'numeric1_to_numeric2_ratio' in engineer.df.columns

    def test_save_and_load_transformers(self, sample_df):
        """Test saving and loading fitted transformers (NEW FEATURE)."""
        engineer = FeatureEngineer(sample_df)

        # Fit some transformers
        engineer.encode_categorical_label(['categorical'], inplace=True)
        engineer.scale_features(['numeric1'], method='standard', inplace=True)

        assert len(engineer.encoders) > 0
        assert len(engineer.scalers) > 0

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engineer.save_transformers(tmp_path)

            # Create new engineer and load
            new_engineer = FeatureEngineer(sample_df)
            assert len(new_engineer.encoders) == 0  # Should be empty initially

            new_engineer.load_transformers(tmp_path)

            # Should have loaded transformers
            assert len(new_engineer.encoders) > 0
            assert len(new_engineer.scalers) > 0
            assert 'categorical_label' in new_engineer.encoders
            assert 'standard_scaler' in new_engineer.scalers
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_transformers_without_fitting(self, sample_df):
        """Test that saving without fitting raises error."""
        from feature_engineering_tk import TransformerNotFittedError
        engineer = FeatureEngineer(sample_df)

        with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
            with pytest.raises(TransformerNotFittedError):
                engineer.save_transformers(tmp.name)

    def test_input_validation_scale_method(self, sample_df):
        """Test input validation for scaling method."""
        from feature_engineering_tk import InvalidMethodError
        engineer = FeatureEngineer(sample_df)

        with pytest.raises(InvalidMethodError):
            engineer.scale_features(['numeric1'], method='invalid')

    def test_input_validation_polynomial_degree(self, sample_df):
        """Test input validation for polynomial degree."""
        engineer = FeatureEngineer(sample_df)

        with pytest.raises(ValueError, match="degree must be 2 or 3"):
            engineer.create_polynomial_features(['numeric1'], degree=5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
