"""
Tests for feature_selection module.

Covers FeatureSelector's variance/correlation/target-correlation filtering,
statistical tests, tree-based importance, missing-value filtering, and the
select_features_auto() convenience pipeline. Includes regression tests for
bugs found during a dedicated audit of this previously-untested file:

- select_features_auto() type hint/docstring vs. actual tuple return
- select_by_missing_values() mutating the caller's exclude_columns list
- select_by_target_correlation() crashing with a raw pandas ValueError on
  non-numeric (classification) targets instead of raising a project exception
- select_by_statistical_test() silently falling back to f_classif for an
  unrecognized score_func string instead of raising InvalidMethodError
"""

import pytest
import pandas as pd
import numpy as np

from feature_engineering_tk import FeatureSelector, select_features_auto
from feature_engineering_tk.exceptions import ColumnNotFoundError, DataTypeError, InvalidMethodError


@pytest.fixture
def classification_df():
    """Numeric features + a numeric binary target, with some redundancy."""
    np.random.seed(42)
    n = 200
    feature1 = np.random.randn(n)
    df = pd.DataFrame({
        'feature1': feature1,
        # near-duplicate of feature1 -> highly correlated pair
        'feature1_dup': feature1 + np.random.randn(n) * 0.01,
        'feature2': np.random.randn(n),
        'constant': np.full(n, 5.0),  # zero variance
        'target': (feature1 + np.random.randn(n) * 0.1 > 0).astype(int),
    })
    return df


@pytest.fixture
def regression_df():
    np.random.seed(0)
    n = 150
    feature1 = np.random.randn(n)
    feature2 = np.random.randn(n)
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'noise': np.random.randn(n),
        'target': feature1 * 3 + feature2 * 0.5 + np.random.randn(n) * 0.1,
    })
    return df


class TestSelectByVariance:
    """Tests for select_by_variance()."""

    def test_removes_constant_columns(self, classification_df):
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_variance(threshold=0.01)

        assert 'constant' not in selected
        assert 'feature1' in selected
        assert 'target' not in selected  # target always excluded

    def test_target_automatically_excluded(self, classification_df):
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_variance(threshold=0.0)
        assert 'target' not in selected

    def test_updates_selected_features_and_scores(self, classification_df):
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_variance(threshold=0.0)

        assert selector.selected_features == selected
        assert 'variance' in selector.feature_scores

    def test_no_numeric_features_returns_empty(self):
        df = pd.DataFrame({'cat': ['a', 'b', 'c'], 'target': [0, 1, 0]})
        selector = FeatureSelector(df, target_column='target')
        selected = selector.select_by_variance()
        assert selected == []

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        selector = FeatureSelector(df, target_column=None)
        selected = selector.select_by_variance()
        assert selected == []

    def test_no_target_column_still_works(self, classification_df):
        df = classification_df.drop(columns=['target'])
        selector = FeatureSelector(df, target_column=None)
        selected = selector.select_by_variance(threshold=0.01)
        assert 'constant' not in selected
        assert 'feature1' in selected


class TestSelectByCorrelation:
    """Tests for select_by_correlation()."""

    def test_drops_one_of_highly_correlated_pair(self, classification_df):
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_correlation(threshold=0.9)

        # feature1 and feature1_dup are ~perfectly correlated; only one survives
        assert not ({'feature1', 'feature1_dup'} <= set(selected))
        assert 'feature2' in selected

    def test_order_dependent_keeps_earlier_indexed_column(self, classification_df):
        """
        Documented limitation: elimination always keeps the earlier-indexed
        column of a correlated pair, regardless of which is more predictive
        of the target. This test pins down that behavior explicitly.
        """
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_correlation(threshold=0.9)

        # feature1 comes before feature1_dup in column order -> feature1 kept
        assert 'feature1' in selected
        assert 'feature1_dup' not in selected

    def test_target_excluded_from_correlation_check(self, classification_df):
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_correlation(threshold=0.5)
        assert 'target' not in selected

    def test_fewer_than_two_features_returns_as_is(self):
        df = pd.DataFrame({'feature1': [1, 2, 3, 4], 'target': [0, 1, 0, 1]})
        selector = FeatureSelector(df, target_column='target')
        selected = selector.select_by_correlation()
        assert selected == ['feature1']

    def test_no_correlated_pairs_keeps_all(self, regression_df):
        df = regression_df.drop(columns=['target'])
        selector = FeatureSelector(df, target_column=None)
        selected = selector.select_by_correlation(threshold=0.99)
        assert set(selected) == set(df.columns)


class TestSelectByTargetCorrelation:
    """Tests for select_by_target_correlation(), including bug #3 regression."""

    def test_selects_top_k_by_absolute_correlation(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selected = selector.select_by_target_correlation(k=2)

        assert len(selected) == 2
        assert 'feature1' in selected  # strongest signal by construction

    def test_no_target_column_set_raises(self, regression_df):
        selector = FeatureSelector(regression_df, target_column=None)
        with pytest.raises(ValueError, match="target_column must be specified"):
            selector.select_by_target_correlation()

    def test_target_not_in_dataframe_raises(self, regression_df):
        """Regression test: this used to raise a plain ValueError; it now
        raises ColumnNotFoundError (NOT a ValueError subclass), a breaking
        change for anyone catching ValueError specifically here."""
        selector = FeatureSelector(regression_df, target_column='does_not_exist')
        with pytest.raises(ColumnNotFoundError, match="not found in dataframe"):
            selector.select_by_target_correlation()

    def test_non_numeric_target_raises_datatype_error(self, classification_df):
        """Regression test for bug #3: string-label target used to crash with
        a raw pandas ValueError from corrwith(); it must now raise the
        project's DataTypeError with a clear message instead."""
        df = classification_df.copy()
        df['target'] = df['target'].map({0: 'no', 1: 'yes'})

        selector = FeatureSelector(df, target_column='target')
        with pytest.raises(DataTypeError):
            selector.select_by_target_correlation()

    def test_k_larger_than_available_features_returns_all(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selected = selector.select_by_target_correlation(k=1000)
        assert len(selected) == 3  # feature1, feature2, noise

    def test_feature_scores_recorded(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selector.select_by_target_correlation(k=2)
        assert 'correlation_with_target' in selector.feature_scores


class TestSelectByStatisticalTest:
    """Tests for select_by_statistical_test(), including bug #4 regression."""

    def test_classification_default_score_func(self, classification_df):
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_statistical_test(k=2, task='classification')
        assert len(selected) == 2
        assert 'statistical_test' in selector.feature_scores

    def test_regression_default_score_func(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selected = selector.select_by_statistical_test(k=2, task='regression')
        assert len(selected) == 2

    def test_valid_score_func_string(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selected = selector.select_by_statistical_test(k=2, score_func='f_regression')
        assert len(selected) == 2

    def test_invalid_score_func_string_raises(self, regression_df):
        """Regression test for bug #4: a typo'd score_func used to silently
        fall back to f_classif instead of raising."""
        selector = FeatureSelector(regression_df, target_column='target')
        with pytest.raises(InvalidMethodError):
            selector.select_by_statistical_test(k=2, score_func='f_clasif')

    def test_invalid_score_func_message_lists_valid_options(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        with pytest.raises(InvalidMethodError, match="f_regression"):
            selector.select_by_statistical_test(k=2, score_func='not_a_real_func')

    def test_callable_score_func_accepted(self, regression_df):
        from sklearn.feature_selection import f_regression
        selector = FeatureSelector(regression_df, target_column='target')
        selected = selector.select_by_statistical_test(k=2, score_func=f_regression)
        assert len(selected) == 2

    def test_no_target_column_raises(self, regression_df):
        selector = FeatureSelector(regression_df, target_column=None)
        with pytest.raises(ValueError, match="target_column must be specified"):
            selector.select_by_statistical_test()

    def test_target_not_found_raises(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='missing')
        with pytest.raises(ColumnNotFoundError, match="not found in dataframe"):
            selector.select_by_statistical_test()


class TestSelectByImportance:
    """Tests for select_by_importance()."""

    def test_classification_importance(self, classification_df):
        selector = FeatureSelector(classification_df, target_column='target')
        selected = selector.select_by_importance(k=2, task='classification', n_estimators=10)
        assert len(selected) == 2
        assert 'importance' in selector.feature_scores

    def test_regression_importance(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selected = selector.select_by_importance(k=2, task='regression', n_estimators=10)
        assert len(selected) == 2
        # feature1 should dominate by construction
        assert 'feature1' in selected

    def test_no_target_column_raises(self, regression_df):
        selector = FeatureSelector(regression_df, target_column=None)
        with pytest.raises(ValueError, match="target_column must be specified"):
            selector.select_by_importance()

    def test_target_not_found_raises(self, regression_df):
        """Regression test: this used to raise a plain ValueError; it now
        raises ColumnNotFoundError (NOT a ValueError subclass), a breaking
        change for anyone catching ValueError specifically here."""
        selector = FeatureSelector(regression_df, target_column='missing')
        with pytest.raises(ColumnNotFoundError, match="not found in dataframe"):
            selector.select_by_importance()

    def test_k_larger_than_features_returns_all(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selected = selector.select_by_importance(k=100, task='regression', n_estimators=10)
        assert len(selected) == 3


class TestSelectByMissingValues:
    """Tests for select_by_missing_values(), including bug #2 regression."""

    @pytest.fixture
    def missing_df(self):
        return pd.DataFrame({
            'low_missing': [1, 2, np.nan, 4, 5],       # 20% missing
            'high_missing': [1, np.nan, np.nan, np.nan, 5],  # 60% missing
            'no_missing': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0],
        })

    def test_filters_by_threshold(self, missing_df):
        selector = FeatureSelector(missing_df, target_column='target')
        selected = selector.select_by_missing_values(threshold=0.5)

        assert 'low_missing' in selected
        assert 'high_missing' not in selected
        assert 'no_missing' in selected

    def test_exclude_columns_not_mutated(self, missing_df):
        """Regression test for bug #2: calling the method twice with the same
        exclude_columns list object must not grow that list via .append()."""
        selector = FeatureSelector(missing_df, target_column='target')
        exclude = ['no_missing']

        selector.select_by_missing_values(threshold=0.5, exclude_columns=exclude)
        assert exclude == ['no_missing']

        selector.select_by_missing_values(threshold=0.5, exclude_columns=exclude)
        assert exclude == ['no_missing']

    def test_target_excluded_by_default(self, missing_df):
        selector = FeatureSelector(missing_df, target_column='target')
        selected = selector.select_by_missing_values(threshold=1.0)
        assert 'target' not in selected

    def test_no_features_survive_returns_empty(self, missing_df):
        selector = FeatureSelector(missing_df, target_column='target')
        selected = selector.select_by_missing_values(threshold=0.0)
        # high_missing (60%) and low_missing (20%) both exceed a 0.0 threshold
        assert 'high_missing' not in selected
        assert 'low_missing' not in selected
        assert 'no_missing' in selected

    def test_feature_scores_recorded(self, missing_df):
        selector = FeatureSelector(missing_df, target_column='target')
        selector.select_by_missing_values(threshold=0.5)
        assert 'missing_ratios' in selector.feature_scores


class TestGetFeatureImportanceDf:
    """Tests for get_feature_importance_df()."""

    def test_no_scores_returns_empty_dataframe(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        result = selector.get_feature_importance_df()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_latest_score_type_sorted(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selector.select_by_variance(threshold=0.0)
        selector.select_by_target_correlation(k=3)

        result = selector.get_feature_importance_df(sort=True)

        assert list(result.columns) == ['feature', 'score', 'score_type']
        assert (result['score_type'] == 'correlation_with_target').all()
        # Sorted descending by score
        assert result['score'].is_monotonic_decreasing

    def test_unsorted_preserves_dict_order(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selector.select_by_variance(threshold=0.0)
        result = selector.get_feature_importance_df(sort=False)
        assert set(result['feature']) == {'feature1', 'feature2', 'noise'}


class TestApplySelection:
    """Tests for apply_selection()."""

    def test_applies_stored_selected_features(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selector.select_by_target_correlation(k=1)
        result = selector.apply_selection()

        assert 'target' in result.columns
        assert len(result.columns) == 2  # 1 feature + target

    def test_applies_explicit_features_arg(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        result = selector.apply_selection(['feature1', 'feature2'], keep_target=False)
        assert set(result.columns) == {'feature1', 'feature2'}

    def test_no_target_kept_when_keep_target_false(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        result = selector.apply_selection(['feature1'], keep_target=False)
        assert 'target' not in result.columns

    def test_no_selected_features_returns_original_df(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        result = selector.apply_selection(selected_features=[])
        assert set(result.columns) == set(regression_df.columns)

    def test_invalid_columns_silently_dropped(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        result = selector.apply_selection(['feature1', 'nonexistent_col'], keep_target=False)
        assert list(result.columns) == ['feature1']

    def test_returns_copy_not_view(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        result = selector.apply_selection(['feature1'], keep_target=False)
        result['feature1'] = 0
        assert not (selector.df['feature1'] == 0).all()


class TestGetSelectedFeatures:
    """Tests for get_selected_features()."""

    def test_returns_empty_list_before_selection(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        assert selector.get_selected_features() == []

    def test_returns_copy_not_reference(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selector.select_by_variance(threshold=0.0)

        result = selector.get_selected_features()
        result.append('mutated')

        assert 'mutated' not in selector.selected_features

    def test_matches_last_selection_call(self, regression_df):
        selector = FeatureSelector(regression_df, target_column='target')
        selector.select_by_target_correlation(k=1)
        assert len(selector.get_selected_features()) == 1


class TestSelectFeaturesAuto:
    """Tests for select_features_auto(), including bug #1 regression."""

    def test_returns_tuple_of_dataframe_and_selector(self, regression_df):
        """Regression test for bug #1: the function's type hint used to claim
        -> pd.DataFrame while the body actually returned a 2-tuple. Verify the
        real (documented) contract: (DataFrame, FeatureSelector)."""
        result = select_features_auto(regression_df, target_column='target', task='regression')

        assert isinstance(result, tuple)
        assert len(result) == 2

        result_df, selector = result
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(selector, FeatureSelector)

    def test_selector_exposes_selected_features_and_scores(self, regression_df):
        _, selector = select_features_auto(regression_df, target_column='target', task='regression')

        assert selector.selected_features
        assert 'importance' in selector.feature_scores

    def test_result_df_contains_target(self, regression_df):
        result_df, _ = select_features_auto(regression_df, target_column='target', task='regression')
        assert 'target' in result_df.columns

    def test_respects_max_features(self, regression_df):
        result_df, selector = select_features_auto(
            regression_df, target_column='target', task='regression', max_features=1
        )
        assert len(selector.selected_features) <= 1

    def test_classification_task(self, classification_df):
        result_df, selector = select_features_auto(
            classification_df, target_column='target', task='classification', max_features=2
        )
        assert isinstance(result_df, pd.DataFrame)
        assert 'target' in result_df.columns

    def test_removes_constant_and_correlated_features(self, classification_df):
        result_df, selector = select_features_auto(
            classification_df,
            target_column='target',
            task='classification',
            variance_threshold=0.01,
            correlation_threshold=0.9,
        )
        assert 'constant' not in result_df.columns
        # Only one of the near-duplicate pair should survive the pipeline
        assert not ({'feature1', 'feature1_dup'} <= set(result_df.columns))
