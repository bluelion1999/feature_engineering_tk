"""
Tests for the utils module.

Covers the shared validation and column-selection helpers used across all
toolkit classes, including regression tests for pandas dtype-fragility bugs:

- ``validate_numeric_columns()`` crashing on pandas nullable/extension
  dtypes (Int64, Float64, boolean) because it used
  ``np.issubdtype(dtype, np.number)`` instead of
  ``pd.api.types.is_numeric_dtype()``.
- ``_is_string_like_dtype()`` missing pyarrow-backed ``ArrowDtype`` string
  columns (e.g. from ``pd.read_parquet(dtype_backend="pyarrow")``).
- ``get_string_columns()`` silently dropping Categorical-dtype columns.
"""

import pytest
import pandas as pd
import numpy as np
from feature_engineering_tk.utils import (
    _is_string_like_dtype,
    validate_and_copy_dataframe,
    validate_columns,
    get_numeric_columns,
    validate_numeric_columns,
    get_string_columns,
    get_feature_columns,
)
from feature_engineering_tk.exceptions import ColumnNotFoundError


class TestIsStringLikeDtype:
    """Test suite for the internal _is_string_like_dtype() helper."""

    def test_object_dtype_is_string_like(self):
        """Classic object-dtype columns are treated as string-like."""
        s = pd.Series(['a', 'b', 'c'], dtype='object')
        assert _is_string_like_dtype(s.dtype) is True

    def test_pandas_stringdtype_is_string_like(self):
        """Pandas' native StringDtype (pandas >= 3.0 default) is string-like."""
        s = pd.Series(['a', 'b', 'c'], dtype='string')
        assert _is_string_like_dtype(s.dtype) is True

    def test_category_dtype_is_string_like(self):
        """Regression (bug #3): Categorical dtype should be string-like,
        matching how get_categorical_summary() already treats it."""
        s = pd.Series(['a', 'b', 'a'], dtype='category')
        assert _is_string_like_dtype(s.dtype) is True

    def test_category_dtype_with_numeric_categories_is_string_like(self):
        """Any Categorical is string-like regardless of category value dtype."""
        s = pd.Series([1, 2, 1], dtype='category')
        assert _is_string_like_dtype(s.dtype) is True

    def test_numeric_dtype_is_not_string_like(self):
        """Plain numeric dtypes should not be treated as string-like."""
        s = pd.Series([1, 2, 3])
        assert _is_string_like_dtype(s.dtype) is False

    def test_nullable_integer_dtype_is_not_string_like(self):
        """Pandas nullable Int64 should not be treated as string-like."""
        s = pd.Series([1, 2, 3], dtype='Int64')
        assert _is_string_like_dtype(s.dtype) is False

    def test_arrow_string_dtype_is_string_like(self):
        """Regression (bug #2): pyarrow-backed string columns (ArrowDtype
        wrapping pa.string()) should be treated as string-like."""
        pa = pytest.importorskip("pyarrow")
        s = pd.Series(['a', 'b', 'c'], dtype=pd.ArrowDtype(pa.string()))
        assert _is_string_like_dtype(s.dtype) is True

    def test_arrow_large_string_dtype_is_string_like(self):
        """pa.large_string() backed ArrowDtype should also be string-like."""
        pa = pytest.importorskip("pyarrow")
        s = pd.Series(['a', 'b', 'c'], dtype=pd.ArrowDtype(pa.large_string()))
        assert _is_string_like_dtype(s.dtype) is True

    def test_arrow_numeric_dtype_is_not_string_like(self):
        """ArrowDtype wrapping a non-string pyarrow type is not string-like."""
        pa = pytest.importorskip("pyarrow")
        s = pd.Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int64()))
        assert _is_string_like_dtype(s.dtype) is False


class TestValidateAndCopyDataframe:
    """Test suite for validate_and_copy_dataframe()."""

    def test_returns_copy_not_same_object(self):
        """The returned DataFrame should be a distinct copy."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = validate_and_copy_dataframe(df)
        assert result is not df
        assert result.equals(df)

    def test_mutating_copy_does_not_affect_original(self):
        """Mutating the returned copy must not mutate the caller's DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = validate_and_copy_dataframe(df)
        result.loc[0, 'a'] = 999
        assert df.loc[0, 'a'] == 1

    def test_raises_typeerror_for_non_dataframe(self):
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            validate_and_copy_dataframe([1, 2, 3])

    def test_empty_dataframe_logs_warning_but_succeeds(self, caplog):
        """Empty DataFrame should not raise, just warn."""
        df = pd.DataFrame()
        with caplog.at_level('WARNING'):
            result = validate_and_copy_dataframe(df)
        assert result.empty
        assert any('empty' in msg.lower() for msg in caplog.messages)


class TestValidateColumns:
    """Test suite for validate_columns()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

    def test_normalizes_single_string_to_list(self, sample_df):
        """A single string column name should be normalized to a list."""
        result = validate_columns(sample_df, 'a')
        assert result == ['a']

    def test_valid_columns_returned_unchanged(self, sample_df):
        """All-valid column lists pass through unchanged."""
        result = validate_columns(sample_df, ['a', 'b'])
        assert result == ['a', 'b']

    def test_missing_columns_dropped_by_default(self, sample_df, caplog):
        """Missing columns should be dropped and logged, not raised, by default."""
        with caplog.at_level('WARNING'):
            result = validate_columns(sample_df, ['a', 'missing'])
        assert result == ['a']
        assert any('missing' in msg for msg in caplog.messages)

    def test_missing_columns_raises_when_requested(self, sample_df):
        """raise_on_missing=True should raise ColumnNotFoundError (not a plain
        ValueError - ColumnNotFoundError subclasses ValidationError ->
        MLToolkitError -> Exception, so code that used to catch ValueError
        here must now catch ColumnNotFoundError instead)."""
        with pytest.raises(ColumnNotFoundError):
            validate_columns(sample_df, ['missing'], raise_on_missing=True)

    def test_missing_columns_raises_with_column_and_available_columns(self, sample_df):
        """The raised ColumnNotFoundError should carry the missing column name
        and the DataFrame's available columns for a richer, catchable error."""
        with pytest.raises(ColumnNotFoundError) as exc_info:
            validate_columns(sample_df, ['missing'], raise_on_missing=True)
        assert exc_info.value.column_name == 'missing'
        assert exc_info.value.available_columns == ['a', 'b', 'c']

    def test_missing_columns_raises_reports_first_missing_of_several(self, sample_df):
        """With multiple missing columns, the exception should report the
        first one encountered (deterministic, matches input order)."""
        with pytest.raises(ColumnNotFoundError) as exc_info:
            validate_columns(sample_df, ['missing1', 'missing2'], raise_on_missing=True)
        assert exc_info.value.column_name == 'missing1'

    def test_empty_column_list_returns_empty_list(self, sample_df, caplog):
        """An empty input list should return an empty list without error."""
        with caplog.at_level('WARNING'):
            result = validate_columns(sample_df, [])
        assert result == []

    def test_duplicate_column_names_preserved(self, sample_df):
        """Duplicate names in the input are preserved (not deduped)."""
        result = validate_columns(sample_df, ['a', 'a', 'b'])
        assert result == ['a', 'a', 'b']

    def test_invalid_type_raises_typeerror(self, sample_df):
        """Non-str/list input should raise TypeError."""
        with pytest.raises(TypeError):
            validate_columns(sample_df, 123)


class TestGetNumericColumns:
    """Test suite for get_numeric_columns()."""

    def test_returns_only_numeric_columns(self):
        """Mixed dtype DataFrame should return only numeric columns."""
        df = pd.DataFrame({
            'num': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'flt': [1.1, 2.2, 3.3],
        })
        result = get_numeric_columns(df)
        assert set(result) == {'num', 'flt'}

    def test_respects_columns_subset(self):
        """When a subset of columns is given, only checks that subset."""
        df = pd.DataFrame({'num': [1, 2], 'flt': [1.1, 2.2], 'text': ['a', 'b']})
        result = get_numeric_columns(df, columns=['num', 'text'])
        assert result == ['num']

    def test_nullable_dtypes_recognized_as_numeric(self):
        """get_numeric_columns already handles nullable dtypes via select_dtypes."""
        df = pd.DataFrame({
            'nullable_int': pd.array([1, 2, None], dtype='Int64'),
            'nullable_float': pd.array([1.0, None, 3.0], dtype='Float64'),
            'nullable_bool': pd.array([True, False, None], dtype='boolean'),
            'text': ['a', 'b', 'c'],
        })
        result = get_numeric_columns(df)
        assert set(result) == {'nullable_int', 'nullable_float'}

    def test_no_numeric_columns_returns_empty_list(self):
        """All-string DataFrame should return an empty list."""
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['z', 'w']})
        assert get_numeric_columns(df) == []


class TestValidateNumericColumns:
    """Test suite for validate_numeric_columns()."""

    def test_returns_only_numeric_columns(self):
        """Non-numeric columns among the input should be dropped."""
        df = pd.DataFrame({'num': [1, 2, 3], 'text': ['a', 'b', 'c']})
        result = validate_numeric_columns(df, ['num', 'text'])
        assert result == ['num']

    def test_missing_column_skipped_with_warning(self, caplog):
        """A column not present in the DataFrame should be skipped, not raise."""
        df = pd.DataFrame({'num': [1, 2, 3]})
        with caplog.at_level('WARNING'):
            result = validate_numeric_columns(df, ['num', 'missing'])
        assert result == ['num']

    def test_empty_input_list_returns_empty_list(self):
        """Empty input should return an empty list without error."""
        df = pd.DataFrame({'num': [1, 2, 3]})
        assert validate_numeric_columns(df, []) == []

    def test_nullable_int64_does_not_raise(self):
        """Regression (bug #1): nullable Int64 columns used to raise
        TypeError: Cannot interpret 'Int64Dtype()' as a data type because
        np.issubdtype() can't handle pandas extension dtypes."""
        df = pd.DataFrame({'nullable_int': pd.array([1, 2, None], dtype='Int64')})
        result = validate_numeric_columns(df, ['nullable_int'])
        assert result == ['nullable_int']

    def test_nullable_float64_does_not_raise(self):
        """Regression (bug #1): nullable Float64 columns should validate cleanly."""
        df = pd.DataFrame({'nullable_float': pd.array([1.0, None, 3.0], dtype='Float64')})
        result = validate_numeric_columns(df, ['nullable_float'])
        assert result == ['nullable_float']

    def test_nullable_boolean_does_not_raise(self):
        """Regression (bug #1): nullable boolean dtype should validate cleanly
        (booleans are numeric-like for pandas' is_numeric_dtype)."""
        df = pd.DataFrame({'nullable_bool': pd.array([True, False, None], dtype='boolean')})
        result = validate_numeric_columns(df, ['nullable_bool'])
        assert result == ['nullable_bool']

    def test_non_numeric_column_skipped_with_warning(self, caplog):
        """A string column should be skipped with a warning, not raised."""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        with caplog.at_level('WARNING'):
            result = validate_numeric_columns(df, ['text'])
        assert result == []


class TestGetStringColumns:
    """Test suite for get_string_columns()."""

    def test_returns_object_and_string_dtype_columns(self):
        """Object and native StringDtype columns should both be returned."""
        df = pd.DataFrame({
            'obj_col': pd.Series(['a', 'b'], dtype='object'),
            'string_col': pd.Series(['c', 'd'], dtype='string'),
            'num_col': [1, 2],
        })
        result = get_string_columns(df)
        assert set(result) == {'obj_col', 'string_col'}

    def test_category_columns_included(self):
        """Regression (bug #3): category-dtype columns should no longer be
        silently dropped, even when their categories are strings."""
        df = pd.DataFrame({
            'cat_col': pd.Series(['x', 'y', 'x'], dtype='category'),
            'num_col': [1, 2, 3],
        })
        result = get_string_columns(df)
        assert 'cat_col' in result
        assert 'num_col' not in result

    def test_category_columns_with_numeric_categories_included(self):
        """A Categorical column is treated as string-like unconditionally,
        regardless of what its categories actually contain."""
        df = pd.DataFrame({'cat_numeric': pd.Series([1, 2, 1], dtype='category')})
        result = get_string_columns(df)
        assert result == ['cat_numeric']

    def test_respects_columns_subset(self):
        """When given a subset, only that subset is checked/returned."""
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['z', 'w'], 'c': [1, 2]})
        result = get_string_columns(df, columns=['a', 'c'])
        assert result == ['a']

    def test_missing_columns_in_subset_logged_and_skipped(self, caplog):
        """Columns in the subset that don't exist should be logged and skipped."""
        df = pd.DataFrame({'a': ['x', 'y']})
        with caplog.at_level('WARNING'):
            result = get_string_columns(df, columns=['a', 'missing'])
        assert result == ['a']

    def test_empty_columns_subset_returns_empty_list(self):
        """An empty subset should return an empty list."""
        df = pd.DataFrame({'a': ['x', 'y']})
        assert get_string_columns(df, columns=[]) == []

    def test_no_string_columns_returns_empty_list(self):
        """All-numeric DataFrame should return an empty list."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
        assert get_string_columns(df) == []

    def test_arrow_string_columns_included(self):
        """Regression (bug #2): ArrowDtype string columns should be
        recognized via the columns=None path too."""
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame({
            'arrow_str': pd.Series(['a', 'b'], dtype=pd.ArrowDtype(pa.string())),
            'arrow_int': pd.Series([1, 2], dtype=pd.ArrowDtype(pa.int64())),
        })
        result = get_string_columns(df)
        assert result == ['arrow_str']


class TestGetFeatureColumns:
    """Test suite for get_feature_columns()."""

    def test_numeric_only_excludes_string_columns(self):
        """numeric_only=True (default) should drop string columns."""
        df = pd.DataFrame({'num': [1, 2], 'text': ['a', 'b']})
        result = get_feature_columns(df)
        assert result == ['num']

    def test_numeric_only_false_returns_all_columns(self):
        """numeric_only=False should keep all columns (minus exclusions)."""
        df = pd.DataFrame({'num': [1, 2], 'text': ['a', 'b']})
        result = get_feature_columns(df, numeric_only=False)
        assert set(result) == {'num', 'text'}

    def test_exclude_columns_removed(self):
        """Columns named in exclude_columns should be removed from the result."""
        df = pd.DataFrame({'num': [1, 2], 'target': [0, 1]})
        result = get_feature_columns(df, exclude_columns=['target'], numeric_only=False)
        assert result == ['num']

    def test_default_exclude_columns_is_empty(self):
        """exclude_columns=None should behave like an empty exclusion list."""
        df = pd.DataFrame({'num': [1, 2]})
        result = get_feature_columns(df, exclude_columns=None)
        assert result == ['num']

    def test_exclude_nonexistent_column_is_a_no_op(self):
        """Excluding a column name that isn't present shouldn't raise."""
        df = pd.DataFrame({'num': [1, 2]})
        result = get_feature_columns(df, exclude_columns=['not_present'])
        assert result == ['num']

    def test_typo_exclude_column_logs_warning(self, caplog):
        """
        Regression: a typo'd/nonexistent exclude_columns entry (e.g.
        'taget' instead of 'target') used to be silently ignored with no
        signal to the caller that their exclusion request had no effect.
        It should now log a warning naming the missing column(s), matching
        the style of validate_columns()/get_string_columns() in this same
        module, while still returning the correct feature columns.
        """
        df = pd.DataFrame({'num': [1, 2], 'target': [0, 1]})
        with caplog.at_level('WARNING'):
            result = get_feature_columns(
                df, exclude_columns=['taget'], numeric_only=False
            )
        # The typo'd column was never actually excluded (unchanged, tolerant
        # behavior), but a warning must be logged about it.
        assert set(result) == {'num', 'target'}
        assert any('taget' in msg for msg in caplog.messages)

    def test_mixed_valid_and_typo_exclude_columns_warns_only_for_typo(self, caplog):
        """A mix of a real column and a typo should exclude the real one
        and warn only about the one that doesn't exist."""
        df = pd.DataFrame({'num': [1, 2], 'target': [0, 1]})
        with caplog.at_level('WARNING'):
            result = get_feature_columns(
                df, exclude_columns=['target', 'taget'], numeric_only=False
            )
        assert result == ['num']
        assert len(caplog.messages) == 1
        assert "['taget']" in caplog.messages[0]

    def test_correct_exclude_usage_unchanged_no_warning(self, caplog):
        """Regression check: normal, correct exclude_columns usage (all
        entries exist in the DataFrame) should behave exactly as before -
        no warning logged, and the excluded column removed from the result."""
        df = pd.DataFrame({'num': [1, 2], 'target': [0, 1]})
        with caplog.at_level('WARNING'):
            result = get_feature_columns(
                df, exclude_columns=['target'], numeric_only=False
            )
        assert result == ['num']
        assert caplog.messages == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
