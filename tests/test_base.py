"""
Tests for the base module (FeatureEngineeringBase).

FeatureEngineeringBase is covered thoroughly: it's the shared parent class
for every public toolkit class (DataPreprocessor, FeatureEngineer,
DataAnalyzer, TargetAnalyzer, FeatureSelector).

Note: this module previously also defined an `inplace_transform` decorator.
It was removed (see CHANGELOG.md "Removed") because it was dead code - grep
confirmed zero usages anywhere in the codebase - and structurally broken
relative to its own documented usage pattern: the docstring told callers to
reference a `df_result` variable inside the decorated method body, but the
decorator never actually injected such a variable into the method's local
scope, so following the documented pattern exactly would have raised
NameError. Since it had no real callers to preserve compatibility for, it
was deleted outright rather than fixed in place.
"""

import pytest
import pandas as pd
from feature_engineering_tk.base import FeatureEngineeringBase


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


def test_inplace_transform_no_longer_exported():
    """
    Regression: inplace_transform was removed from base.py entirely (dead,
    unused, and structurally broken relative to its own docstring - see
    module docstring above). Confirm it's actually gone rather than just
    unused, so this doesn't silently regress if someone re-adds it.
    """
    import feature_engineering_tk.base as base_module
    assert not hasattr(base_module, 'inplace_transform')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
