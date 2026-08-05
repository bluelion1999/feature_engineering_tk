"""
Tests for the base module (FeatureEngineeringBase, inplace_transform).

FeatureEngineeringBase is covered thoroughly: it's the shared parent class
for every public toolkit class (DataPreprocessor, FeatureEngineer,
DataAnalyzer, TargetAnalyzer, FeatureSelector).

The inplace_transform decorator is covered against its ACTUAL behavior, not
its documented usage pattern. Investigation during this test pass found the
decorator's own docstring example is broken: it tells callers to reference a
`df_result` variable inside the decorated method body, but the decorator
never injects such a variable into the method's local scope (a Python
function's local names can't be injected from outside like that) - calling a
method written exactly as the docstring shows raises NameError. The decorator
only "works" (in the sense of round-tripping inplace=True/False correctly)
if the decorated method reads/writes `self.df` directly instead - the
opposite of what the docstring instructs. It also appears to be unused
anywhere in the current codebase (grep confirms no callers). Per task scope,
this is documented here and left unfixed; tests below assert the actual,
observed behavior rather than the aspirational documented one.
"""

import pytest
import pandas as pd
from feature_engineering_tk.base import FeatureEngineeringBase, inplace_transform


class TestFeatureEngineeringBase:
    """Test suite for the FeatureEngineeringBase shared base class."""

    def test_init_copies_dataframe(self):
        """The base class should store a copy, not a reference, of the input."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        obj = FeatureEngineeringBase(df)
        assert obj.df is not df
        assert obj.df.equals(df)

    def test_init_mutation_isolated_from_caller(self):
        """Mutating obj.df must not affect the caller's original DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        obj = FeatureEngineeringBase(df)
        obj.df.loc[0, 'a'] = 999
        assert df.loc[0, 'a'] == 1

    def test_init_raises_typeerror_for_non_dataframe(self):
        """Passing a non-DataFrame should raise TypeError."""
        with pytest.raises(TypeError):
            FeatureEngineeringBase([1, 2, 3])

    def test_init_with_empty_dataframe_does_not_raise(self, caplog):
        """An empty DataFrame is a valid (if unusual) input; should just warn."""
        with caplog.at_level('WARNING'):
            obj = FeatureEngineeringBase(pd.DataFrame())
        assert obj.df.empty
        assert any('empty' in msg.lower() for msg in caplog.messages)

    def test_get_dataframe_returns_copy(self):
        """get_dataframe() should return a fresh copy each call."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        obj = FeatureEngineeringBase(df)
        result = obj.get_dataframe()
        assert result is not obj.df
        assert result.equals(obj.df)

    def test_get_dataframe_mutation_does_not_affect_internal_state(self):
        """Mutating the returned copy must not affect obj.df."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        obj = FeatureEngineeringBase(df)
        result = obj.get_dataframe()
        result.loc[0, 'a'] = 999
        assert obj.df.loc[0, 'a'] == 1

    def test_subclass_inherits_base_behavior(self):
        """A subclass with no custom __init__ should get the same behavior."""
        class Dummy(FeatureEngineeringBase):
            pass

        df = pd.DataFrame({'x': [1, 2]})
        dummy = Dummy(df)
        assert dummy.df.equals(df)
        assert dummy.get_dataframe().equals(df)


class TestInplaceTransformDecorator:
    """
    Test suite for the inplace_transform decorator's ACTUAL behavior.

    See module docstring for the discrepancy found between this decorator's
    documented usage pattern and what it actually does. These tests
    intentionally do NOT exercise the documented `df_result`-variable
    pattern as a "should work" case - it doesn't - but they do assert it
    fails the way it currently fails, so a future fix (or removal) has a
    baseline to compare against.
    """

    def test_documented_usage_pattern_raises_nameerror(self):
        """
        The decorator's own docstring shows:

            @inplace_transform
            def some_method(self, columns, inplace=False):
                df_result[columns] = df_result[columns].transform(...)
                return df_result

        `df_result` is never defined in the method's local scope (it is not
        a parameter, and the decorator cannot inject a local variable into
        another function's frame), so following this documented pattern
        raises NameError. This is the decorator's actual, current behavior.
        """
        class Dummy(FeatureEngineeringBase):
            @inplace_transform
            def some_method(self, columns, inplace=False):
                df_result[columns] = df_result[columns] * 2  # noqa: F821
                return df_result  # noqa: F821

        dummy = Dummy(pd.DataFrame({'a': [1, 2, 3]}))
        with pytest.raises(NameError):
            dummy.some_method(['a'], inplace=False)

    def test_self_df_pattern_inplace_false_returns_transformed_copy(self):
        """
        If the decorated method reads/writes `self.df` directly (not the
        documented `df_result` pattern), the decorator does correctly:
        - leave the original object's `self.df` unchanged when inplace=False
        - return the transformed DataFrame
        """
        class Dummy(FeatureEngineeringBase):
            @inplace_transform
            def double(self, columns, inplace=False):
                self.df[columns] = self.df[columns] * 2
                return self.df

        df = pd.DataFrame({'a': [1, 2, 3]})
        dummy = Dummy(df)
        result = dummy.double(['a'], inplace=False)

        assert result['a'].tolist() == [2, 4, 6]
        # self.df should be restored/unchanged since inplace=False
        assert dummy.df['a'].tolist() == [1, 2, 3]

    def test_self_df_pattern_inplace_true_updates_state_and_returns_self(self):
        """With the self.df pattern and inplace=True, the decorator updates
        self.df and returns self (enabling chaining), matching the rest of
        the toolkit's inplace convention."""
        class Dummy(FeatureEngineeringBase):
            @inplace_transform
            def double(self, columns, inplace=False):
                self.df[columns] = self.df[columns] * 2
                return self.df

        dummy = Dummy(pd.DataFrame({'a': [1, 2, 3]}))
        result = dummy.double(['a'], inplace=True)

        assert result is dummy
        assert dummy.df['a'].tolist() == [2, 4, 6]

    def test_self_df_pattern_restores_original_df_on_exception(self):
        """If the decorated method raises, self.df should be restored to its
        pre-call state when inplace=False (per the decorator's except block)."""
        class Dummy(FeatureEngineeringBase):
            @inplace_transform
            def boom(self, inplace=False):
                self.df['a'] = self.df['a'] * 2
                raise ValueError("boom")

        df = pd.DataFrame({'a': [1, 2, 3]})
        dummy = Dummy(df)
        with pytest.raises(ValueError):
            dummy.boom(inplace=False)

        assert dummy.df['a'].tolist() == [1, 2, 3]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
