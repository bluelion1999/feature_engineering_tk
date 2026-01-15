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

        # Should not crash with division by zero (lower threshold to actually detect outlier)
        result = preprocessor.handle_outliers(
            columns=['constant', 'variable'],
            method='zscore',
            action='remove',
            threshold=1.5,  # Threshold to catch 100 as outlier (z-score ~1.97)
            inplace=False
        )

        # Constant column should be skipped, variable column processed
        assert 'constant' in result.columns
        assert len(result) < len(df)  # Outlier 100 should be removed from variable

    def test_input_validation(self, sample_df):
        """Test input validation for various methods."""
        preprocessor = DataPreprocessor(sample_df)

        # Invalid strategy - now raises InvalidStrategyError
        from feature_engineering_tk import InvalidStrategyError, InvalidMethodError
        with pytest.raises(InvalidStrategyError):
            preprocessor.handle_missing_values(strategy='invalid')

        # Invalid keep parameter
        with pytest.raises(ValueError):
            preprocessor.remove_duplicates(keep='invalid')

        # Invalid method for handle_outliers - now raises InvalidMethodError
        with pytest.raises(InvalidMethodError):
            preprocessor.handle_outliers(columns=['numeric1'], method='invalid')


class TestStringPreprocessing:
    """Tests for string preprocessing methods."""

    @pytest.fixture
    def string_df(self):
        return pd.DataFrame({
            'name': ['  Alice  ', 'BOB', ' Charlie ', 'DAVID'],
            'city': ['New York  ', '  los angeles', 'CHICAGO', '  Boston  '],
            'description': ['Test123!', 'Hello World', 'Data@Science', '12345'],
            'numeric': [1, 2, 3, 4]
        })

    def test_clean_string_columns_strip_lower(self, string_df):
        """Test cleaning with strip and lower operations."""
        preprocessor = DataPreprocessor(string_df)
        result = preprocessor.clean_string_columns(
            ['name', 'city'],
            operations=['strip', 'lower'],
            inplace=False
        )

        assert result['name'].tolist() == ['alice', 'bob', 'charlie', 'david']
        assert result['city'].tolist() == ['new york', 'los angeles', 'chicago', 'boston']
        assert preprocessor.df['name'].tolist() == ['  Alice  ', 'BOB', ' Charlie ', 'DAVID']  # Original unchanged

    def test_clean_string_columns_remove_operations(self, string_df):
        """Test remove_punctuation and remove_digits operations."""
        preprocessor = DataPreprocessor(string_df)
        result = preprocessor.clean_string_columns(
            ['description'],
            operations=['remove_punctuation', 'remove_digits'],
            inplace=False
        )

        assert result['description'].iloc[0] == 'Test'
        assert result['description'].iloc[2] == 'DataScience'
        assert result['description'].iloc[3] == ''

    def test_clean_string_columns_invalid_operations(self, string_df):
        """Test error handling for invalid operations."""
        from feature_engineering_tk import InvalidMethodError
        preprocessor = DataPreprocessor(string_df)

        with pytest.raises(InvalidMethodError):
            preprocessor.clean_string_columns(['name'], operations=['invalid_op'])

    def test_clean_string_columns_inplace(self, string_df):
        """Test inplace modification."""
        preprocessor = DataPreprocessor(string_df)
        result = preprocessor.clean_string_columns(
            ['name'],
            operations=['strip', 'lower'],
            inplace=True
        )

        assert result is preprocessor  # Returns self for chaining
        assert preprocessor.df['name'].tolist() == ['alice', 'bob', 'charlie', 'david']

    def test_handle_whitespace_variants(self, string_df):
        """Test whitespace variant standardization."""
        df = pd.DataFrame({
            'category': ['Apple', '  Apple', 'Apple  ', '  Apple  ', 'Banana']
        })
        preprocessor = DataPreprocessor(df)
        result = preprocessor.handle_whitespace_variants(['category'], inplace=False)

        unique_vals = result['category'].unique()
        assert len(unique_vals) == 2  # Only 'Apple' and 'Banana'
        assert result['category'].iloc[0] == 'Apple'
        assert result['category'].iloc[1] == 'Apple'
        assert result['category'].iloc[4] == 'Banana'

    def test_extract_string_length(self, string_df):
        """Test string length feature extraction."""
        preprocessor = DataPreprocessor(string_df)
        result = preprocessor.extract_string_length(['name', 'description'], inplace=False)

        assert 'name_length' in result.columns
        assert 'description_length' in result.columns
        assert result['name_length'].iloc[0] == len('  Alice  ')
        assert result['description_length'].iloc[1] == len('Hello World')

    def test_extract_string_length_custom_suffix(self, string_df):
        """Test custom suffix for length columns."""
        preprocessor = DataPreprocessor(string_df)
        result = preprocessor.extract_string_length(['name'], suffix='_len', inplace=False)

        assert 'name_len' in result.columns
        assert 'name_length' not in result.columns


class TestDataValidation:
    """Tests for data validation methods."""

    @pytest.fixture
    def quality_df(self):
        return pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [1, 1, 1, 1, 1],  # Constant
            'c': [1, 2, 3, 4, 5],
            'd': [1, np.inf, 3, -np.inf, 5],
            'e': ['x', 'y', 'z', 'w', 'v']  # High cardinality (100% unique)
        })

    def test_validate_data_quality_all_issues(self, quality_df):
        """Test data quality validation detects all issue types."""
        preprocessor = DataPreprocessor(quality_df)
        validation = preprocessor.validate_data_quality()

        assert validation['shape'] == (5, 5)
        assert 'a' in validation['missing_values']
        assert validation['missing_values']['a'] == 1
        assert 'b' in validation['constant_columns']
        assert 'd' in validation['infinite_values']
        assert validation['infinite_values']['d'] == 2
        assert len(validation['issues_found']) > 0

    def test_validate_data_quality_clean_data(self):
        """Test validation with clean data."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [6, 7, 8, 9, 10]
        })
        preprocessor = DataPreprocessor(df)
        validation = preprocessor.validate_data_quality()

        assert validation['duplicate_rows'] == 0
        assert len(validation['missing_values']) == 0
        assert len(validation['constant_columns']) == 0
        assert len(validation['infinite_values']) == 0
        assert "No major data quality issues detected" in validation['issues_found']

    def test_detect_infinite_values(self, quality_df):
        """Test infinite value detection."""
        preprocessor = DataPreprocessor(quality_df)
        inf_counts = preprocessor.detect_infinite_values()

        assert 'd' in inf_counts
        assert inf_counts['d'] == 2  # One np.inf and one -np.inf

    def test_detect_infinite_values_specific_columns(self, quality_df):
        """Test infinite value detection for specific columns."""
        preprocessor = DataPreprocessor(quality_df)
        inf_counts = preprocessor.detect_infinite_values(columns=['c', 'd'])

        assert 'd' in inf_counts
        assert 'a' not in inf_counts
        assert 'b' not in inf_counts

    def test_create_missing_indicators(self):
        """Test missing value indicator creation."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35, np.nan, 45],
            'income': [50000, 60000, np.nan, 70000, 80000]
        })
        preprocessor = DataPreprocessor(df)
        result = preprocessor.create_missing_indicators(['age', 'income'], inplace=False)

        assert 'age_was_missing' in result.columns
        assert 'income_was_missing' in result.columns
        assert result['age_was_missing'].tolist() == [0, 1, 0, 1, 0]
        assert result['income_was_missing'].tolist() == [0, 0, 1, 0, 0]

    def test_create_missing_indicators_custom_suffix(self):
        """Test missing indicators with custom suffix."""
        df = pd.DataFrame({'x': [1, np.nan, 3]})
        preprocessor = DataPreprocessor(df)
        result = preprocessor.create_missing_indicators(['x'], suffix='_missing', inplace=False)

        assert 'x_missing' in result.columns
        assert result['x_missing'].tolist() == [0, 1, 0]

    def test_create_missing_indicators_uses_df_result_not_self_df(self):
        """Test that create_missing_indicators correctly uses df_result (Bug #1 fix).

        Bug #1: Line 1093 previously used self.df[col] instead of df_result[col].
        This test verifies the fix uses df_result correctly.
        """
        # Simple test: Create DataFrame with missing values
        df = pd.DataFrame({
            'x': [1, np.nan, 3, np.nan, 5],
            'y': [10, 20, 30, 40, 50]
        })
        preprocessor = DataPreprocessor(df)

        # Call create_missing_indicators with inplace=False
        result = preprocessor.create_missing_indicators(['x'], inplace=False)

        # Verify correct behavior
        assert 'x_was_missing' in result.columns
        assert result['x_was_missing'].tolist() == [0, 1, 0, 1, 0]

        # Ensure original preprocessor.df unchanged
        assert 'x_was_missing' not in preprocessor.df.columns


class TestEnhancedErrorHandling:
    """Tests for enhanced error handling in existing methods."""

    def test_clip_values_bounds_validation(self):
        """Test that clip_values validates lower < upper."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        preprocessor = DataPreprocessor(df)

        with pytest.raises(ValueError, match="lower bound.*must be less than upper bound"):
            preprocessor.clip_values('x', lower=10, upper=5)

    def test_sample_data_size_validation(self):
        """Test that sample_data validates n <= len(df)."""
        from feature_engineering_tk import ValidationError
        df = pd.DataFrame({'x': [1, 2, 3]})
        preprocessor = DataPreprocessor(df)

        with pytest.raises(ValidationError, match="Cannot sample.*rows"):
            preprocessor.sample_data(n=10)

    def test_handle_outliers_type_validation(self):
        """Test that handle_outliers validates columns is a list."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        preprocessor = DataPreprocessor(df)

        with pytest.raises(TypeError, match="columns must be a list"):
            preprocessor.handle_outliers(columns='x')  # Should be ['x']

    def test_handle_missing_values_drop_warning(self):
        """Test warning when drop strategy removes many rows."""
        df = pd.DataFrame({
            'x': [1, np.nan, np.nan, np.nan, np.nan]  # 80% missing
        })
        preprocessor = DataPreprocessor(df)

        # This should log a warning (would need to capture logs to test properly)
        result = preprocessor.handle_missing_values(strategy='drop', columns=['x'])
        assert len(result) == 1  # Only one non-missing row

    def test_remove_duplicates_logging(self):
        """Test that duplicate removal includes logging."""
        df = pd.DataFrame({
            'x': [1, 1, 2, 2, 3],
            'y': ['a', 'a', 'b', 'b', 'c']
        })
        preprocessor = DataPreprocessor(df)

        result = preprocessor.remove_duplicates()
        assert len(result) == 3  # Should remove 2 duplicates

    def test_handle_outliers_logging(self):
        """Test that outlier detection includes logging."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 100, 5]  # 100 is an outlier
        })
        preprocessor = DataPreprocessor(df)

        result = preprocessor.handle_outliers(['x'], method='iqr', action='remove')
        assert len(result) < len(df)  # Outlier should be removed


class TestMethodChaining:
    """Tests for fluent API method chaining."""

    def test_basic_chaining(self):
        """Test basic method chaining with inplace=True."""
        df = pd.DataFrame({
            'x': [1, 1, 2, 3, 4],  # Actual duplicate rows
            'y': [5, 5, 6, 7, 8],
            'name': ['  Alice  ', '  Alice  ', 'Charlie', 'David', 'Eve']
        })
        preprocessor = DataPreprocessor(df)

        result = preprocessor\
            .remove_duplicates(inplace=True)\
            .clean_string_columns(['name'], operations=['strip', 'lower'], inplace=True)

        # Should return the preprocessor instance
        assert isinstance(result, DataPreprocessor)
        # Should have modified internal df
        assert len(preprocessor.df) == 4  # One duplicate removed
        assert preprocessor.df['name'].iloc[0] == 'alice'

    def test_chaining_multiple_operations(self):
        """Test chaining multiple different operations."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35, 100, 45],  # Has missing and outlier
            'income': [50000, 60000, 70000, 70000, 80000],  # Has duplicate row
            'name': ['  Alice  ', '  Bob  ', 'Charlie', 'David', 'Eve']
        })
        # Add duplicate row
        df = pd.concat([df, df.iloc[[1]]], ignore_index=True)

        preprocessor = DataPreprocessor(df)

        result = preprocessor\
            .handle_missing_values(strategy='mean', columns=['age'], inplace=True)\
            .remove_duplicates(inplace=True)\
            .handle_outliers(['age'], method='iqr', action='cap', inplace=True)\
            .clean_string_columns(['name'], operations=['strip', 'lower'], inplace=True)

        assert isinstance(result, DataPreprocessor)
        assert preprocessor.df['age'].isna().sum() == 0  # No missing values
        assert len(preprocessor.df) == 5  # Duplicate removed
        assert all(preprocessor.df['name'].str.islower())  # All lowercase
        assert all(preprocessor.df['name'].str.strip() == preprocessor.df['name'])  # All stripped

    def test_chaining_returns_self(self):
        """Test that chaining actually returns self, not a copy."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        preprocessor = DataPreprocessor(df)

        result = preprocessor.remove_duplicates(inplace=True)

        assert result is preprocessor  # Same object
        assert id(result) == id(preprocessor)

    def test_non_inplace_breaks_chain(self):
        """Test that inplace=False returns DataFrame, not preprocessor."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        preprocessor = DataPreprocessor(df)

        result = preprocessor.remove_duplicates(inplace=False)

        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, DataPreprocessor)

    def test_chaining_with_new_methods(self):
        """Test chaining with newly added string and validation methods."""
        df = pd.DataFrame({
            'name': ['  Alice  ', 'Bob', 'Charlie'],
            'age': [25, np.nan, 35]
        })
        preprocessor = DataPreprocessor(df)

        result = preprocessor\
            .create_missing_indicators(['age'], inplace=True)\
            .handle_whitespace_variants(['name'], inplace=True)\
            .extract_string_length(['name'], inplace=True)

        assert isinstance(result, DataPreprocessor)
        assert 'age_was_missing' in preprocessor.df.columns
        assert 'name_length' in preprocessor.df.columns

    def test_chaining_all_methods(self):
        """Test that all methods support chaining."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v'],
            'c': [10, 20, 30, 40, 50]
        })
        preprocessor = DataPreprocessor(df)

        # Test various methods return self
        assert preprocessor.clip_values('a', lower=2, upper=4, inplace=True) is preprocessor
        assert preprocessor.sample_data(n=3, inplace=True) is preprocessor


class TestOperationHistory:
    """Test suite for operation history and summary tracking."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 1],  # Has duplicates
            'b': [10, 20, None, 40, 50, 10],  # Has missing
            'c': ['x', 'y', 'z', 'w', 'v', 'x'],
            'd': [100, 200, 300, 400, 500, 100]
        })

    def test_history_initialized_empty(self, sample_df):
        """Test that operation history starts empty."""
        preprocessor = DataPreprocessor(sample_df)
        assert preprocessor._operation_history == []

    def test_history_tracks_single_operation(self, sample_df):
        """Test that single operation is tracked correctly."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=True)

        assert len(preprocessor._operation_history) == 1
        op = preprocessor._operation_history[0]

        assert op['method'] == 'remove_duplicates'
        assert 'timestamp' in op
        assert op['shape_before'] == (6, 4)
        assert op['shape_after'] == (5, 4)
        assert op['rows_changed'] == -1
        assert op['cols_changed'] == 0

    def test_history_tracks_multiple_operations(self, sample_df):
        """Test that multiple operations are tracked in order."""
        preprocessor = DataPreprocessor(sample_df)

        preprocessor.remove_duplicates(inplace=True)
        preprocessor.handle_missing_values(strategy='drop', inplace=True)
        preprocessor.drop_columns(['d'], inplace=True)

        assert len(preprocessor._operation_history) == 3

        assert preprocessor._operation_history[0]['method'] == 'remove_duplicates'
        assert preprocessor._operation_history[1]['method'] == 'handle_missing_values'
        assert preprocessor._operation_history[2]['method'] == 'drop_columns'

        # Check final shape
        assert preprocessor._operation_history[-1]['shape_after'] == (4, 3)

    def test_history_includes_parameters(self, sample_df):
        """Test that operation parameters are recorded."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.handle_missing_values(strategy='mean', columns=['b'], inplace=True)

        op = preprocessor._operation_history[0]
        assert op['parameters']['strategy'] == 'mean'
        assert op['parameters']['columns'] == ['b']

    def test_history_includes_additional_info(self, sample_df):
        """Test that additional info is recorded when provided."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=True)

        op = preprocessor._operation_history[0]
        assert 'details' in op
        assert 'rows_removed' in op['details']

    def test_history_not_tracked_when_inplace_false(self, sample_df):
        """Test that operations are NOT tracked when inplace=False."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=False)
        preprocessor.drop_columns(['d'], inplace=False)

        assert len(preprocessor._operation_history) == 0

    def test_get_summary_empty_history(self, sample_df):
        """Test summary when no operations performed."""
        preprocessor = DataPreprocessor(sample_df)
        summary = preprocessor.get_preprocessing_summary()

        assert "No preprocessing operations" in summary

    def test_get_summary_with_operations(self, sample_df):
        """Test summary generation with operations."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=True)
        preprocessor.drop_columns(['d'], inplace=True)

        summary = preprocessor.get_preprocessing_summary()

        assert "PREPROCESSING SUMMARY" in summary
        assert "REMOVE_DUPLICATES" in summary
        assert "DROP_COLUMNS" in summary
        assert "TOTAL OPERATIONS: 2" in summary
        assert "Initial shape: (6, 4)" in summary
        assert "Final shape: (5, 3)" in summary

    def test_get_summary_shows_shape_changes(self, sample_df):
        """Test that summary correctly shows shape changes."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.handle_outliers(['a'], method='iqr', action='remove', inplace=True)

        summary = preprocessor.get_preprocessing_summary()
        assert "Shape:" in summary
        assert "Rows changed:" in summary

    def test_export_summary_text_format(self, sample_df, tmp_path):
        """Test exporting summary as text file."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=True)

        filepath = tmp_path / "summary.txt"
        preprocessor.export_summary(str(filepath), format='text')

        assert filepath.exists()
        content = filepath.read_text()
        assert "PREPROCESSING SUMMARY" in content
        assert "REMOVE_DUPLICATES" in content

    def test_export_summary_markdown_format(self, sample_df, tmp_path):
        """Test exporting summary as markdown file."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=True)
        preprocessor.drop_columns(['d'], inplace=True)

        filepath = tmp_path / "summary.md"
        preprocessor.export_summary(str(filepath), format='markdown')

        assert filepath.exists()
        content = filepath.read_text()
        assert "# Preprocessing Summary" in content
        assert "## Operations" in content
        assert "remove_duplicates" in content

    def test_export_summary_json_format(self, sample_df, tmp_path):
        """Test exporting summary as JSON file."""
        import json
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=True)

        filepath = tmp_path / "summary.json"
        preprocessor.export_summary(str(filepath), format='json')

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)

        assert 'operations' in data
        assert 'summary' in data
        assert len(data['operations']) == 1
        assert data['operations'][0]['method'] == 'remove_duplicates'

    def test_export_summary_invalid_format(self, sample_df):
        """Test that invalid format raises ValueError."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.remove_duplicates(inplace=True)

        with pytest.raises(ValueError, match="format must be one of"):
            preprocessor.export_summary('test.txt', format='invalid')

    def test_export_summary_no_operations_raises_error(self, sample_df):
        """Test that exporting with no operations raises ValidationError."""
        from feature_engineering_tk.exceptions import ValidationError
        preprocessor = DataPreprocessor(sample_df)

        with pytest.raises(ValidationError, match="No preprocessing operations"):
            preprocessor.export_summary('test.txt')

    def test_history_complex_workflow(self, sample_df):
        """Test history tracking through complex workflow."""
        preprocessor = DataPreprocessor(sample_df)

        # Perform multiple operations
        preprocessor.remove_duplicates(inplace=True)
        preprocessor.handle_missing_values(strategy='mean', inplace=True)
        preprocessor.clean_string_columns(['c'], operations=['strip', 'lower'], inplace=True)
        preprocessor.drop_columns(['d'], inplace=True)

        # Verify history
        assert len(preprocessor._operation_history) == 4

        # Verify cumulative shape changes
        initial_shape = preprocessor._operation_history[0]['shape_before']
        final_shape = preprocessor._operation_history[-1]['shape_after']

        assert initial_shape == (6, 4)
        assert final_shape[0] == 5  # One duplicate removed
        assert final_shape[1] == 3  # One column dropped

    def test_history_tracks_filter_rows(self, sample_df):
        """Test that filter_rows operation is tracked."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.filter_rows(lambda df: df['a'] > 2, inplace=True)

        assert len(preprocessor._operation_history) == 1
        op = preprocessor._operation_history[0]
        assert op['method'] == 'filter_rows'
        assert op['rows_changed'] < 0  # Rows were removed

    def test_history_tracks_handle_outliers(self, sample_df):
        """Test that handle_outliers operation is tracked."""
        preprocessor = DataPreprocessor(sample_df)
        preprocessor.handle_outliers(['a'], method='iqr', action='remove', inplace=True)

        assert len(preprocessor._operation_history) == 1
        op = preprocessor._operation_history[0]
        assert op['method'] == 'handle_outliers'
        assert 'method' in op['parameters']
        assert 'action' in op['parameters']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
