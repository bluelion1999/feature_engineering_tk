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

        # Original should be unchanged (still string-like: 'object' on
        # pandas < 3.0, native StringDtype by default on pandas >= 3.0 -
        # isinstance check since its .name varies, e.g. 'str' in pandas 3.0)
        original_dtype = engineer.df['categorical'].dtype
        assert original_dtype == 'object' or isinstance(original_dtype, pd.StringDtype)
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

    def test_scale_features_empty_dataframe_raises(self, sample_df):
        """scale_features() on a zero-row DataFrame used to crash with a
        low-level sklearn ValueError ("Found array with 0 sample(s)...");
        it should now raise a clear, catchable EmptyDataFrameError instead."""
        from feature_engineering_tk import EmptyDataFrameError

        empty_df = sample_df.iloc[0:0].copy()
        engineer = FeatureEngineer(empty_df)

        with pytest.raises(EmptyDataFrameError):
            engineer.scale_features(['numeric1'], method='standard')

    def test_scale_features_non_empty_still_works(self, sample_df):
        """Regression guard: the empty-DataFrame check must not affect
        normal, non-empty scaling behavior."""
        engineer = FeatureEngineer(sample_df)
        result = engineer.scale_features(['numeric1'], method='standard')
        assert abs(result['numeric1'].mean()) < 1e-10

    def test_create_binning_quantile_on_constant_column_raises(self, sample_df):
        """create_binning(strategy='quantile') on a constant (zero-variance)
        column used to silently succeed while producing an all-NaN binned
        column (pandas.qcut() collapses every edge to the same value and,
        with duplicates='drop', returns all-NaN with no warning). It should
        now raise a clear ConstantColumnError instead."""
        from feature_engineering_tk import ConstantColumnError

        df = sample_df.copy()
        df['constant'] = 7
        engineer = FeatureEngineer(df)

        with pytest.raises(ConstantColumnError):
            engineer.create_binning('constant', bins=3, strategy='quantile')

    def test_create_binning_uniform_on_constant_column_still_works(self, sample_df):
        """Regression guard: strategy='uniform' produces a valid (if
        degenerate) single-bin result for constant columns and must not be
        affected by the new quantile-only ConstantColumnError guard."""
        df = sample_df.copy()
        df['constant'] = 7
        engineer = FeatureEngineer(df)

        result = engineer.create_binning('constant', bins=3, strategy='uniform')
        assert 'constant_binned' in result.columns
        assert result['constant_binned'].notna().all()

    def test_create_binning_quantile_on_non_constant_column_still_works(self, sample_df):
        """Regression guard: the new constant-column guard must not affect
        normal quantile binning on a column with variance."""
        engineer = FeatureEngineer(sample_df)
        result = engineer.create_binning('numeric1', bins=3, strategy='quantile')
        assert 'numeric1_binned' in result.columns

    # -- encode_categorical_onehot --------------------------------------
    #
    # Pins down the pre-existing pd.get_dummies()-based output contract
    # (column naming/dtype) while also covering the encoder-storage fix:
    # self.encoders should now be populated, matching encode_categorical_label
    # and encode_categorical_ordinal.

    def test_onehot_output_column_naming_unchanged(self, sample_df):
        """Output columns/dtype should still match historical pd.get_dummies() behavior."""
        engineer = FeatureEngineer(sample_df)

        result = engineer.encode_categorical_onehot(['categorical'], inplace=False)

        # Original categorical column replaced by one dummy column per category
        assert 'categorical' not in result.columns
        for expected_col in ['categorical_A', 'categorical_B', 'categorical_C']:
            assert expected_col in result.columns
            assert pd.api.types.is_integer_dtype(result[expected_col])

        # Values match get_dummies semantics
        assert result.loc[0, 'categorical_A'] == 1
        assert result.loc[0, 'categorical_B'] == 0
        assert result.loc[1, 'categorical_B'] == 1

    def test_onehot_inplace_false_does_not_modify_original(self, sample_df):
        """inplace=False must not mutate the internal dataframe."""
        engineer = FeatureEngineer(sample_df)

        result = engineer.encode_categorical_onehot(['categorical'], inplace=False)

        assert 'categorical' in engineer.df.columns
        assert 'categorical_A' not in engineer.df.columns
        assert 'categorical_A' in result.columns

    def test_onehot_inplace_true_returns_self_and_updates_df(self, sample_df):
        """inplace=True should update self.df and return self for chaining."""
        engineer = FeatureEngineer(sample_df)

        result = engineer.encode_categorical_onehot(['categorical'], inplace=True)

        assert result is engineer
        assert 'categorical' not in engineer.df.columns
        assert 'categorical_A' in engineer.df.columns

    def test_onehot_populates_encoders_inplace_false(self, sample_df):
        """Bug fix: self.encoders should be populated even with inplace=False."""
        from sklearn.preprocessing import OneHotEncoder

        engineer = FeatureEngineer(sample_df)
        engineer.encode_categorical_onehot(['categorical'], inplace=False)

        assert 'categorical_onehot' in engineer.encoders
        assert isinstance(engineer.encoders['categorical_onehot'], OneHotEncoder)

    def test_onehot_populates_encoders_inplace_true(self, sample_df):
        """Bug fix: self.encoders should be populated with inplace=True too."""
        from sklearn.preprocessing import OneHotEncoder

        engineer = FeatureEngineer(sample_df)
        engineer.encode_categorical_onehot(['categorical'], inplace=True)

        assert 'categorical_onehot' in engineer.encoders
        assert isinstance(engineer.encoders['categorical_onehot'], OneHotEncoder)

    def test_onehot_save_and_load_transformers_roundtrip(self, sample_df):
        """The stored OneHotEncoder should survive save_transformers()/load_transformers()."""
        engineer = FeatureEngineer(sample_df)
        engineer.encode_categorical_onehot(['categorical'], inplace=True)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engineer.save_transformers(tmp_path)

            new_engineer = FeatureEngineer(sample_df)
            assert 'categorical_onehot' not in new_engineer.encoders

            new_engineer.load_transformers(tmp_path)

            assert 'categorical_onehot' in new_engineer.encoders
            loaded_encoder = new_engineer.encoders['categorical_onehot']
            # Loaded encoder is fitted and usable for transforming new data
            transformed = loaded_encoder.transform([['A']])
            assert transformed.shape[1] == 3  # A, B, C
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_onehot_loaded_encoder_handles_unseen_category(self, sample_df):
        """Stored encoder uses handle_unknown='ignore' so unseen categories at
        inference time don't raise - they encode to all zeros instead."""
        engineer = FeatureEngineer(sample_df)
        engineer.encode_categorical_onehot(['categorical'], inplace=True)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engineer.save_transformers(tmp_path)

            new_engineer = FeatureEngineer(sample_df)
            new_engineer.load_transformers(tmp_path)
            loaded_encoder = new_engineer.encoders['categorical_onehot']

            # 'D' was never seen during fit
            transformed = loaded_encoder.transform([['D']])
            assert transformed.sum() == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_onehot_dummy_na_parameter(self):
        """dummy_na=True should add an explicit indicator column for missing values
        instead of silently encoding NaN rows as all-zero."""
        df = pd.DataFrame({'categorical': ['A', 'B', None, 'A']})
        engineer = FeatureEngineer(df)

        result_default = engineer.encode_categorical_onehot(['categorical'], inplace=False)
        assert 'categorical_nan' not in result_default.columns
        # Missing value row is all-zero across dummy columns (historical behavior)
        assert result_default.loc[2].sum() == 0

        result_dummy_na = engineer.encode_categorical_onehot(
            ['categorical'], dummy_na=True, inplace=False
        )
        assert 'categorical_nan' in result_dummy_na.columns
        assert result_dummy_na.loc[2, 'categorical_nan'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
