# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **`base.inplace_transform` decorator** - deleted outright rather than fixed. It was dead code (grep across the whole package confirmed zero `@inplace_transform` usages on any method) and structurally broken relative to its own docstring: the documented usage pattern told callers to reference a `df_result` variable inside the decorated method body, but the decorator never actually injected such a variable into the method's local scope (not possible from outside a function's frame), so a method written exactly as documented would raise `NameError` the moment it was called. Since it had no real callers anywhere in this codebase to preserve compatibility for, removing it was safer than patching a decorator nobody uses and whose "fixed" behavior nobody has validated against a real use case. **This is technically a breaking change** for any external code that imported `inplace_transform` directly from `feature_engineering_tk.base` - it was never re-exported from the top-level package (`feature_engineering_tk.__all__`), so this should affect nobody in practice, but is called out here for completeness. `Callable`/`functools.wraps` imports in `base.py` were also removed as they existed solely to support this decorator.

### Changed

- **BREAKING (dict-key rename): `TargetAnalyzer.analyze_mutual_information()` regression output** - the regression-path column is now `relative_mi` instead of `normalized_mi`. Classification is unaffected (`normalized_mi` unchanged there). See "Fixed" below for why. Anything parsing the `mutual_info` report section by the old key name (e.g. `report['mutual_info']` entries, or direct `analyze_mutual_information()` DataFrame access) for a regression task needs to switch from `normalized_mi` to `relative_mi`.
- **Behavior change**: features previously flagged as classification leakage suspects solely due to a tiny p-value at large sample sizes will **no longer** be flagged unless they also cross the new large-effect-size threshold. This is intentional - it removes false positives without weakening detection of genuine leakage (near-perfect proxy features still trigger both the p-value and effect-size conditions). The leakage-suspect `reason` text now reports the effect size alongside the p-value (e.g. `"Extremely significant relationship (p=1.2e-45) AND large effect size (eta²=0.912, large)"`) so users can see why a feature was flagged.

### Fixed

- **`utils.get_feature_columns()` silently ignored unmatched `exclude_columns` entries** - if a caller passed a typo'd or otherwise nonexistent column name (e.g. `exclude_columns=['taget']` instead of `'target'`), the function filtered against it as if it were valid, which is a no-op for anything that doesn't match, and gave no indication the exclusion request had no effect. Every other validation function in `utils.py` (`validate_columns()`, `validate_numeric_columns()`, `get_string_columns()`) already logs a warning in this situation; `get_feature_columns()` now does too (`"exclude_columns not found in DataFrame: [...]"`. This is a warning-only fix - the function still does not raise and still returns the same columns it always did, since callers throughout the codebase (`feature_selection.py`, `target_analyzer/quality.py`, `target_analyzer/statistical.py`, `target_analyzer/suggestions.py`) rely on its existing tolerant behavior.
- **Missing intercept in `DataAnalyzer.calculate_vif()`** - the design matrix passed to `statsmodels.stats.outliers_influence.variance_inflation_factor()` never included a constant/intercept column. VIF is only statistically meaningful relative to a regression that includes an intercept, so on any data that isn't already mean-zero (i.e. almost all real-world data) VIF values were badly wrong - features could be under- or over-reported by one to two orders of magnitude, including false positives above the library's own ">10 = high multicollinearity" guidance for features that are actually independent. Fixed by adding a constant column via `statsmodels.tools.tools.add_constant()` before computing per-feature VIF, and skipping the constant's own (not meaningful) VIF entry. `TargetAnalyzer.calculate_vif()` inherits the fix automatically since it delegates to `DataAnalyzer.calculate_vif()`.
- **Misleading mutual information normalization for regression** in `TargetAnalyzer.analyze_mutual_information()` - the regression path normalized each feature's MI score by `np.max(mi_scores)`, i.e. relative to the strongest feature *in that specific dataset*, not any fixed bound. On an all-noise-feature dataset, the noisiest-by-chance feature scored a "perfect" `1.000000` purely for being the best of a bad lot, which fed directly into `generate_recommendations()`'s low-MI feature-selection advice and could read as "well correlated." Classification already used a real fixed bound (`log(n_classes)`, since `MI(X;Y) <= H(Y) <= log(n_classes)` holds independent of other features) and was not affected. Continuous (differential) mutual information has no equivalent fixed bound, so rather than fake one, the regression output is now honestly labeled `relative_mi` (a within-analysis ranking aid only — see "Changed") and `generate_recommendations()` no longer thresholds it as if it were absolute; it now makes an explicitly relative claim with a caveat, and callers needing absolute magnitude are pointed at the raw `mutual_info` column.
- **Silent `fillna(0)` before mutual information computation** - `analyze_mutual_information()` filled missing feature values with `0` with no warning, which fabricates a fake mode and can bias the MI estimate if `0` is itself a meaningful, in-range observed value for a feature (not just "missing"). A `logger.warning` is now emitted (naming the affected columns and count) when this imputation actually fires, matching the imputation-warning style used elsewhere in the toolkit (e.g. `DataPreprocessor`'s multiple-modes warning).
- **Pandas 3.0 compatibility** - `clean_string_columns()`, `handle_whitespace_variants()`, `extract_string_length()`, and `get_categorical_summary()` silently no-op'd on string columns under pandas >= 3.0, which defaults string columns built from Python literals to a native `StringDtype` instead of `object`. Added a version-proof string-dtype check (`utils._is_string_like_dtype()`) used everywhere the toolkit was doing `dtype == 'object'`-based detection.
- **`handle_outliers()` cap action on integer columns** - assigning float cap bounds into an `int64` column via `.loc` now raises under pandas >= 3.0 instead of silently upcasting; integer columns are upcast to float before the cap assignment.
- **False-positive leakage detection on large classification datasets** - `TargetAnalyzer.analyze_data_quality()`'s classification leakage check flagged features based on `pvalue < 1e-10` alone, with no effect-size gate. At large N (hundreds of thousands+ rows), even a genuinely negligible true relationship produces a vanishingly small p-value purely from sample size, so real-world features with no practical leakage risk were being reported as "Extremely significant relationship" leakage suspects. The classification branch now additionally requires a large effect size (`statistical_utils.eta_squared()` for numeric features via ANOVA, `statistical_utils.cramers_v()` for categorical features via chi-square) before flagging - specifically a leakage-specific threshold of `effect_size >= 0.8`, stricter than the conventional "large" cutoff (eta² >= 0.14 / Cramér's V >= ~0.29-0.5), chosen to only catch near-deterministic "feature basically IS the target" relationships, mirroring the regression branch's existing `correlation > 0.99` gate.
- **Dead `has_outliers` branch in `TargetAnalyzer.recommend_models()`** - `analyze_target_distribution()` never populated a `has_outliers` key, so the regression-only recommendation of Huber Regressor for outlier-heavy targets could never fire. `analyze_target_distribution()` now runs `DataAnalyzer.detect_outliers_iqr()` on the target column (reusing the toolkit's existing IQR-based outlier detection rather than adding a new method) and populates `has_outliers` / `outlier_count` in the returned dict, so the Huber Regressor recommendation now actually triggers for regression targets with extreme outliers.
- **`FeatureEngineer.encode_categorical_onehot()` never populated `self.encoders`** - unlike `encode_categorical_label()` and `encode_categorical_ordinal()`, it built dummies with `pd.get_dummies()` directly and fit no transformer, so `save_transformers()`/`load_transformers()` silently omitted one-hot columns and unseen categories at inference time had no fitted encoder to fall back on. The method now also fits a `sklearn.preprocessing.OneHotEncoder` (`handle_unknown='ignore'`) per column and stores it in `self.encoders[f"{col}_onehot"]`, matching the storage convention used by the label/ordinal encoders; the returned DataFrame's columns/values are unchanged (still produced by `pd.get_dummies()`) so this is not a breaking change. Also added an opt-in `dummy_na` parameter (default `False`, preserving prior behavior) to explicitly flag missing values instead of silently encoding them as an all-zero row.
- **`check_normality(method='anderson')` ignored the `alpha` parameter** - the branch hardcoded `result.critical_values[2]` (scipy's 5% significance level) regardless of the requested `alpha`, so `alpha=0.01`, `0.05`, and `0.10` all produced identical output. On scipy >= 1.17, the branch now calls `stats.anderson(..., method='interpolate')` and compares the interpolated p-value against the requested `alpha` (same pattern as the `shapiro`/`normaltest` branches); on older scipy it maps `alpha` to the closest of scipy's fixed significance levels (15%/10%/5%/2.5%/1%) in the legacy critical-value table instead of always using index 2. A warning is logged when the requested `alpha` falls outside what the underlying table supports.
- **`check_normality(method='anderson')` scipy 1.19 crash risk** - calling `scipy.stats.anderson()` without a `method` argument already emits a `FutureWarning` on scipy >= 1.17 (`critical_values`/`significance_level`/`fit_result` are scheduled for removal in 1.19), which would have hard-crashed this branch on `AttributeError` once those attributes are dropped. Fixed by using the non-deprecated `method='interpolate'` API, with a `TypeError`-guarded fallback to the legacy attributes for scipy < 1.17.
- **`feature_selection.py` bugs found during a dedicated audit of this previously-untested module** (`FeatureSelector`, `select_features_auto()`):
  - `select_features_auto()`'s type hint claimed `-> pd.DataFrame` while the body actually returned a `(pd.DataFrame, FeatureSelector)` tuple (a v2.4.2 changelog entry described the tuple-returning fix but never updated the type hint/docstring). The type hint is now `Tuple[pd.DataFrame, FeatureSelector]` and the docstring documents both elements. Callers in `README.md`, `examples/quickstart.ipynb`, and `examples/tutorial_indepth.ipynb` that assigned the tuple to a single variable and then used it as a DataFrame (`.shape`, `.columns`, `.to_csv()`, passing it to `TargetAnalyzer`) were actually broken at runtime; these call sites now unpack the tuple correctly.
  - `select_by_missing_values()` mutated the caller's `exclude_columns` list via `.append()` instead of `+ [...]`, so reusing the same list object across calls silently grew it.
  - `select_by_target_correlation()` crashed with a raw, uninformative pandas `ValueError` when `target_column` held string/categorical labels; it now raises `DataTypeError` with a clear message before attempting the correlation.
  - `select_by_statistical_test()` silently fell back to `f_classif` for any unrecognized `score_func` string (e.g. a typo like `'f_clasif'`) instead of raising; it now raises `InvalidMethodError` listing the valid options.
- **`examples/tutorial_indepth.ipynb` crashed on the tree-based feature importance cell** - `FeatureSelector.select_by_importance()` returns `List[str]` (consistent with every other `select_by_*` method), but the cell treated the result as a DataFrame (`.shape[1]`, `.columns`), raising `AttributeError: 'list' object has no attribute 'shape'`. Every sibling cell in the same section already correctly threads the list through `selector.apply_selection(..., keep_target=True)` to get a DataFrame - this cell was just missing that call. Found and fixed by actually executing both example notebooks end-to-end (`jupyter nbconvert --execute`), not just checking they're valid JSON.
- **`handle_missing_values()` broke method chaining on an empty/all-invalid `columns` list** - when `inplace=True` and the resolved `columns` list ended up empty, the method returned `df_result` (a plain `DataFrame`) instead of `self`, silently breaking any subsequent chained call (`AttributeError: 'DataFrame' object has no attribute '...'`). Now matches the `return df_result if not inplace else self` pattern used by every sibling method.
- **Operation-history logging gaps** - `_log_operation()` was previously wired into only 6 of the ~18 inplace-capable `DataPreprocessor` methods, silently contradicting the documented "automatically logs all preprocessing operations when `inplace=True`" behavior. Added history logging to the remaining 12: `convert_dtypes()`, `clip_values()`, `remove_constant_columns()`, `remove_high_cardinality_columns()`, `rename_columns()`, `reorder_columns()`, `apply_custom_function()`, `reset_index_clean()`, `sample_data()`, `handle_whitespace_variants()`, `extract_string_length()`, and `create_missing_indicators()`. `get_preprocessing_summary()` / `export_summary()` now reflect the complete pipeline for any realistic sequence of `inplace=True` calls.
- **Pandas dtype fragility, round 2** - follow-up audit found the round-1 fix above was incomplete, plus one unrelated bug in the same class:
  - `utils.validate_numeric_columns()` raised `TypeError: Cannot interpret 'Int64Dtype()' as a data type` on pandas nullable dtypes (`Int64`, `Float64`, `boolean`) because it used `np.issubdtype(dtype, np.number)`. Switched to `pd.api.types.is_numeric_dtype()`, matching how `get_numeric_columns()` already handles these dtypes via `select_dtypes`. This crash previously broke `handle_outliers()`/`clip_values()` in `preprocessing.py` and several `feature_engineering.py` methods (`scale_features`, `create_polynomial_features`, `create_log_transform`, `create_sqrt_transform`, `create_binning`, `create_aggregations`, `create_ratio_features`).
  - `utils._is_string_like_dtype()` missed pyarrow-backed string columns (`pd.ArrowDtype(pyarrow.string())`, e.g. from `pd.read_parquet(dtype_backend="pyarrow")`) since `pd.ArrowDtype` is a sibling of `pd.StringDtype`, not a subclass. Now detected via an optional pyarrow check (pyarrow stays an optional, not required, dependency).
  - `utils.get_string_columns()` silently dropped Categorical-dtype columns, causing `clean_string_columns()`, `handle_whitespace_variants()`, and `extract_string_length()` to silently no-op on them. `_is_string_like_dtype()` now treats any Categorical as string-like; `data_analysis.get_categorical_summary()` simplified to rely on this instead of its own separate `select_dtypes(include=['category'])` workaround.
  - `handle_outliers(action='replace')` crashed with `TypeError: Invalid value ... for dtype 'int64'` when the replacement value (median/mean/nan) was fractional or NaN on an `int64` column; the sibling `action='cap'` branch already had this upcast-to-float fix, `action='replace'` was missed.
  - `data_analysis.detect_misclassified_categorical()` used a string-literal dtype check (`col_data.dtype in ['int64', 'int32', ...]`) that never matches pandas nullable integer dtypes (`Int64`, etc.); switched to `pd.api.types.is_integer_dtype()`.
- **`create_polynomial_features(degree=3, interaction_only=True)` silently created 0 features** - the branching logic only handled `degree == 2` (with or without `interaction_only`) and `degree == 3 and not interaction_only`; the `degree == 3 and interaction_only == True` combination matched neither branch and returned the dataframe unchanged with no warning. Added a dedicated branch that generates three-way interaction terms `{col1}_x_{col2}_x_{col3}` for every combination of 3 distinct valid numeric columns (requires at least 3 columns; with fewer, 0 features are created and a `logger.warning` is emitted instead of silently no-op'ing). Docstring updated to document what `interaction_only` means at each degree.
- **`encode_categorical_label()` silently encoded NaN as the literal string `'nan'`** - `.astype(str)` converted real missing values into the four-character string `'nan'` before fitting `LabelEncoder`, which then assigned it an ordinary integer class indistinguishable from a genuine category, destroying missingness information. Now only non-null values are fit/transformed; NaN positions are left as NaN in the result. Columns with no missing values keep their historical integer dtype; columns containing NaN are upcast to `float64` (the only way to represent NaN outside the nullable-Int dtypes), matching how pandas itself upcasts when NaN is introduced into a numeric column. The stored `self.encoders[f"{col}_label"]` is fit only on the observed categories (no fake `'nan'` class) and remains usable for `save_transformers()`/`load_transformers()`.
- **Mutable default argument in `create_aggregations()`** - `agg_funcs: List[str] = ['mean', 'sum', 'std', 'min', 'max']` used a mutable list literal as a default value (flagged by flake8 B006), which is shared across all calls that don't pass `agg_funcs` explicitly and is a classic source of subtle cross-call state bugs if that shared list were ever mutated in place. Changed to `agg_funcs: Optional[List[str]] = None` with the default list constructed fresh inside the method body when `None` is passed.
- **`DataAnalyzer.get_cardinality_info()` crashed on an empty DataFrame** - `unique_count / len(self.df)` is plain Python `int` division, so calling it on a DataFrame with columns but 0 rows raised `ZeroDivisionError` instead of returning a usable result (contrast with `get_missing_summary()`'s equivalent ratio, which is a pandas `Series` division and safely produces `NaN` rather than crashing). `cardinality_ratio` is now set to `NaN` for every column when the DataFrame has 0 rows, with a `logger.warning` noting the degenerate case, matching the toolkit's "log, don't crash on empty input" convention; a DataFrame with 0 rows and 0 columns now correctly returns an empty DataFrame with the expected `column`/`unique_count`/`cardinality_ratio`/`dtype` columns instead of a shapeless empty one. Audited the rest of `data_analysis.py` for the same `/ len(self.df)` / `/ len(df` pattern - the only other occurrence (`get_missing_summary()`) is a Series division and does not share this risk.

### Tests

- Added `tests/test_feature_selection.py` (52 tests) covering every public method of `FeatureSelector` and `select_features_auto()` - happy paths, edge cases (empty DataFrame, all-constant columns, target column excluded from candidates, no columns surviving a filter), and regression tests for each of the four bugs fixed above.
- Added `tests/test_utils.py` and `tests/test_base.py` (previously no dedicated coverage existed for `utils.py`/`base.py`), plus regression tests for the round-2 dtype-fragility fixes in `tests/test_preprocessing.py` and `tests/test_data_analysis.py`.
- Added a chaining regression test, one history-tracking test per newly-wired method (13 total), and an end-to-end multi-method history test for `DataPreprocessor`.
- Added 6 tests for `check_normality()`: alpha-changes-result regression, out-of-range-alpha warning, a deprecation-crash regression test scoped to the anderson call path, and basic `normaltest` coverage.
- Added 9 tests for `encode_categorical_onehot()`: encoder population (both inplace modes), output-column-naming pin, save/load round-trip, unseen-category transform, `dummy_na` behavior.
- Added tests proving `recommend_models()` now fires the Huber Regressor recommendation for a regression target with genuine extreme outliers, and does not fire it for a clean target.
- Added a false-positive regression test (large-N, negligible effect, not flagged) and a true-positive test (near-perfect proxy feature, still flagged) for classification leakage detection.
- Added an all-noise-features regression test proving no feature gets a misleadingly perfect `relative_mi` score, a strong-feature-distinguishable-from-noise sanity test, and a `fillna(0)` warning test for `analyze_mutual_information()`.
- Added tests for `create_polynomial_features(degree=3, interaction_only=True)`: 3-way product terms are created for 3+ columns, and a warning (no crash) is logged for fewer than 3 columns.
- Added tests for `encode_categorical_label()` NaN preservation (both `inplace=True` and `inplace=False`): NaN positions stay NaN, only real categories are fit into the encoder, no fake `'nan'` class appears.
- Added tests for `create_aggregations()` covering the mutable-default-argument fix: default `agg_funcs` still produces all 5 aggregations, and repeated calls with different explicit `agg_funcs` don't leak state between invocations.
- Added regression tests for `get_cardinality_info()`: a normal-DataFrame case pinning `cardinality_ratio` values and descending sort order, a 0-row/2-column case asserting `NaN` ratios instead of `ZeroDivisionError`, and a 0-row/0-column case asserting the correct empty-but-shaped output columns.

## [2.4.3] - 2026-01-20

### Fixed

- **Pandas 2.x compatibility** in `DataAnalyzer.get_basic_info()` - resolved `ValueError` when constructing a DataFrame from columns of different lengths by wrapping all values in single-element lists

### Tests

- Updated `get_basic_info` test to access values via `[0]` index to match the new construction. All 218 tests passing.

## [2.4.2] - 2026-01-20

### Added

- Quickstart and in-depth tutorial notebooks under `examples/`

### Fixed

- **`select_features_auto()`** in `feature_selection.py` - previously did not return an accessible reference to the underlying `FeatureSelector`; now returns the object so callers can inspect `selected_features` / `feature_scores` after the pipeline runs
- Bug fixes in the tutorial notebooks so they run end-to-end against the current API

## [2.4.1] - 2026-01-15

### Fixed

Critical bug fixes from TDD pass on the `fly_catcher` branch (7 total: 4 critical + 3 medium severity):

- **DataFrame reference bug** in `create_missing_indicators()` - fixed to use `df_result` instead of `self.df` when `inplace=False` (#1)
- **Division by zero** in class imbalance calculation - added protection for single-class targets (#2)
- **Unsafe `.iloc[0]` access** in `get_categorical_summary` - added validation for empty `value_counts` (#3)
- **NaN correlation handling** - added NaN checks in feature engineering suggestions to skip constant features (#4)
- **Incomplete outlier capping** - implemented zscore capping (previously only supported IQR method) (#5)
- **Mode calculation edge case** - added warning when multiple modes are detected during imputation (#6)
- **Missing groupby validation** - added upfront validation for single-class targets to improve efficiency (#7)

### Tests

- Added 7 comprehensive tests (218 total: 211 baseline + 7 new)
- Test-first approach: write failing test → fix bug → verify test passes
- 100% backward compatible, no regressions

## [2.4.0] - 2026-01-02

### Added

- **Statistical Robustness Utilities (statistical_utils.py)** - Comprehensive module for statistical validity and confidence
  - **Assumption Validation Functions**:
    - `check_normality()`: Shapiro-Wilk normality test with automatic fallback for large samples
    - `check_homogeneity_of_variance()`: Levene's test for equal variances across groups
    - `validate_sample_size()`: Sample size requirements validation for statistical tests
    - `check_chi2_expected_frequencies()`: Chi-square assumption checks (expected frequencies ≥5)

  - **Effect Size Calculations**:
    - `cohens_d()`: Cohen's d effect size for t-tests with interpretation (small/medium/large)
    - `eta_squared()`: Eta-squared (η²) effect size for ANOVA
    - `cramers_v()`: Cramér's V effect size for chi-square tests
    - `pearson_r_to_d()`: Convert Pearson correlation to Cohen's d

  - **Multiple Testing Corrections**:
    - `apply_multiple_testing_correction()`: Benjamini-Hochberg FDR and Bonferroni corrections
    - Controls false positive rates when testing multiple hypotheses

  - **Confidence Intervals**:
    - `calculate_mean_ci()`: Parametric confidence intervals for means (t-distribution)
    - `calculate_correlation_ci()`: Fisher Z-transformation for correlation confidence intervals
    - `bootstrap_ci()`: Non-parametric bootstrap confidence intervals for any statistic

- **Enhanced TargetAnalyzer Methods** - Optional statistical rigor for all statistical tests
  - **analyze_feature_target_relationship()** enhancements:
    - `check_assumptions=True`: Validates normality, homogeneity of variance, sample size
    - `report_effect_sizes=True`: Includes Cohen's d, eta-squared, Cramér's V
    - `correct_multiple_tests='fdr_bh'`: FDR or Bonferroni correction for multiple features
    - Non-parametric fallback: Automatic Kruskal-Wallis when ANOVA assumptions violated

  - **analyze_class_wise_statistics()** enhancements:
    - `include_ci=True`: Adds 95% confidence intervals for class means
    - `confidence_level=0.95`: Customizable confidence level (default 95%)

  - **analyze_feature_correlations()** enhancements:
    - `include_ci=True`: Fisher Z-transformation confidence intervals for correlations
    - `check_linearity=True`: Validates linear relationship assumption
    - `confidence_level=0.95`: Customizable confidence level (default 95%)

### Changed

- **TargetAnalyzer Statistical Methods** - Enhanced with opt-in statistical validation
  - All statistical tests now support comprehensive assumption checking
  - Effect sizes automatically calculated and interpreted when requested
  - Multiple testing corrections applied when analyzing multiple features
  - Non-parametric alternatives used when parametric assumptions violated

### Improved

- **Statistical Reliability** - Ensures valid, trustworthy statistical inferences
  - Prevents misuse of parametric tests when assumptions violated
  - Quantifies practical significance through effect sizes
  - Controls false discovery rates in multiple comparisons
  - Provides uncertainty quantification through confidence intervals

- **Code Quality**
  - New statistical_utils module with 11 well-tested utility functions
  - Clear documentation and examples for all statistical methods
  - All 211 tests passing - 100% backward compatibility maintained

### Tests

- Added 29 comprehensive tests in test_statistical_utils.py
  - 4 tests for normality checks (normal/non-normal data, sample size handling)
  - 3 tests for homogeneity of variance (equal/unequal variances, edge cases)
  - 2 tests for sample size validation (sufficient/insufficient data)
  - 2 tests for chi-square expected frequencies (valid/invalid tables)
  - 5 tests for effect sizes (Cohen's d, eta-squared, Cramér's V, conversions)
  - 2 tests for multiple testing corrections (FDR, Bonferroni)
  - 5 tests for confidence intervals (mean, correlation, bootstrap with custom statistics)
  - 3 tests for edge cases (NaN handling, zero variance, insufficient bootstrap data)
  - 3 integration tests (ANOVA, chi-square, correlation workflows)

All 211 tests pass successfully.

## [2.3.0] - 2025-12-10

### Added

- **Architecture Refactoring** - Major internal improvements for better maintainability
  - **FeatureEngineeringBase Class**: New base class for all toolkit classes
    - Shared `__init__()` method with DataFrame validation and copying
    - Shared `get_dataframe()` method
    - All 5 main classes (DataPreprocessor, FeatureEngineer, DataAnalyzer, TargetAnalyzer, FeatureSelector) now inherit from common base

  - **Utility Functions Module (utils.py)**: Centralized validation and column selection
    - `validate_and_copy_dataframe()`: DataFrame validation and copying
    - `validate_columns()`: Column existence validation with options
    - `get_numeric_columns()`: Extract numeric columns from DataFrame
    - `validate_numeric_columns()`: Validate and filter numeric columns
    - `get_string_columns()`: Extract string/object columns
    - `get_feature_columns()`: Get feature columns with exclusions
    - Eliminates ~250 lines of duplicate validation code across modules

  - **@inplace_transform Decorator**: Available for future method simplification

- **Benchmarking Infrastructure**
  - New `benchmarks/` directory with comprehensive benchmark suite
  - `benchmark_suite.py`: Performance testing for critical operations
  - `baseline_results.json`: Baseline performance measurements
  - `OPTIMIZATION_PLAN.md`: Detailed optimization strategy and results

### Changed

- **Performance Optimizations** - Significant speed improvements across the library
  - **Class-wise statistics 7x faster** (969ms → 138ms, 86% improvement)
    - Replaced nested filtering loops with single `groupby` operations
    - Eliminates N+1 query pattern in TargetAnalyzer

  - **Outlier detection 45% faster** (221ms → 120ms)
    - Accumulate rows to remove instead of removing in loop
    - Eliminates index alignment issues
    - Single removal operation at end

  - **Pre-computed aggregations**: Mean/median calculations optimized for large datasets
  - **Optimized string validation**: Set-based column existence checks

- **Code Reduction** - ~300 lines of redundant code eliminated (6.5% of codebase)
  - Single source of truth for validation operations
  - Consistent validation patterns across all classes
  - Outlier detection consolidated (DataPreprocessor delegates to DataAnalyzer)

### Improved

- **Code Quality**
  - Better separation of concerns (shared utilities vs domain logic)
  - Improved maintainability (changes to validation made once)
  - Cleaner, more organized codebase structure
  - All 182 tests passing - 100% backward compatibility maintained

- **Documentation**
  - Updated CLAUDE.md with refactoring details
  - Added OPTIMIZATION_PLAN.md with performance benchmarks
  - Comprehensive documentation of new architecture

### Technical Details

**Files Added**:
- `feature_engineering_tk/base.py`: Base class and decorators
- `feature_engineering_tk/utils.py`: Shared utility functions
- `benchmarks/benchmark_suite.py`: Benchmark infrastructure
- `benchmarks/__init__.py`: Package initialization
- `OPTIMIZATION_PLAN.md`: Optimization documentation

**Files Modified**:
- `feature_engineering_tk/preprocessing.py`: Uses base class and utilities, optimized outlier detection
- `feature_engineering_tk/feature_engineering.py`: Uses base class and utilities
- `feature_engineering_tk/data_analysis.py`: Uses base class and utilities, optimized N+1 patterns
- `feature_engineering_tk/feature_selection.py`: Uses base class and utilities

**Benefits**:
- Significantly faster statistical analysis (7x improvement for class-wise statistics)
- Improved code maintainability and consistency
- Single source of truth for validation logic
- Better performance for large datasets
- Cleaner architecture with clear separation of concerns

All 182 tests pass successfully.

## [2.2.0] - 2025-12-07

### Added

- **DataAnalyzer Enhancements** - Column type detection and binning suggestions

  - **Column Type Detection**
    - `detect_misclassified_categorical()`: Identifies numeric columns that should be categorical
    - Detects binary/flag columns (exactly 2 unique values)
    - Finds low cardinality numeric columns (≤10 unique values by default)
    - Identifies columns with very low unique ratios (many repeated values)
    - Catches integer columns with moderate cardinality (≤20 values)

  - **Binning Suggestions**
    - `suggest_binning()`: Recommends binning strategies based on distribution characteristics
    - Quantile binning for skewed distributions (abs(skewness) > 1.0)
    - Uniform binning for relatively uniform distributions
    - Handles outlier-heavy columns appropriately
    - Suggests appropriate number of bins (requires min 20 unique values)

  - **Enhanced `quick_analysis()` Function**
    - New "MISCLASSIFIED CATEGORICAL COLUMNS" section
    - New "BINNING SUGGESTIONS" section with actionable tips
    - Helps identify data type misclassifications during EDA
    - Provides intelligent binning recommendations without requiring a target column

- **DataPreprocessor Enhancements** - Major quality-of-life improvements

  - **Method Chaining Support**
    - All preprocessing methods now return `self` when `inplace=True` (previously returned `self.df`)
    - Enables fluent API pattern for cleaner, more readable code
    - Example: `preprocessor.method1(inplace=True).method2(inplace=True).method3(inplace=True)`

  - **Operation History Tracking**
    - Automatic logging of all preprocessing operations when `inplace=True`
    - `_operation_history`: Internal list tracking all operations with timestamps, parameters, and shape changes
    - `get_preprocessing_summary()`: Returns formatted text summary of all operations
    - `export_summary(filepath, format)`: Export preprocessing history to text/markdown/JSON formats
    - Enables full reproducibility and documentation of preprocessing pipelines

  - **String Preprocessing Methods (3 new methods)**
    - `clean_string_columns()`: Clean string columns with 7 operations (strip, lower, upper, title, remove_punctuation, remove_digits, remove_extra_spaces)
    - `handle_whitespace_variants()`: Standardize whitespace variants in categorical columns
    - `extract_string_length()`: Create length features from string columns

  - **Data Validation Methods (3 new methods)**
    - `validate_data_quality()`: Comprehensive data quality report (missing values, constant columns, infinite values, duplicate count)
    - `detect_infinite_values()`: Detect np.inf/-np.inf in numeric columns
    - `create_missing_indicators()`: Create binary indicator columns for missing values

  - **Enhanced Error Handling**
    - Better parameter validation across all preprocessing methods
    - Warnings for destructive operations (e.g., removing >30% of data)
    - Enhanced logging throughout preprocessing methods

### Changed

- **Breaking Change**: `DataPreprocessor` methods now return `self` when `inplace=True` instead of `self.df`
  - **Impact**: Code that assigns the return value when using `inplace=True` will now receive the preprocessor object instead of a DataFrame
  - **Benefit**: Enables method chaining
  - **Migration**: Use `.df` attribute to access DataFrame, or use method chaining
  - Example:
    ```python
    # Before v2.2.0:
    result = preprocessor.handle_missing_values(inplace=True)  # result was DataFrame

    # After v2.2.0:
    result = preprocessor.handle_missing_values(inplace=True)  # result is DataPreprocessor
    df = result.df  # Access DataFrame via .df attribute

    # Or use method chaining (recommended):
    preprocessor.method1(inplace=True).method2(inplace=True)
    ```

### Tests

- Added 51 comprehensive tests for new features (now 182 total tests)
  - **DataAnalyzer**: 9 tests for column type detection and binning suggestions
  - **DataPreprocessor**: 42 tests
    - 7 tests for string preprocessing
    - 6 tests for data validation
    - 6 tests for enhanced error handling
    - 6 tests for method chaining
    - 17 tests for operation history tracking

All 182 tests pass successfully.

## [2.1.1] - 2025-11-30

### Fixed

- **Critical Configuration Issues**
  - Fixed version mismatch: Updated `setup.py` from 2.0.0 to match 2.1.1 across all configuration files
  - Added missing `statsmodels>=0.14.0` dependency to `requirements.txt` and `setup.py`
  - Fixed `.gitignore` pattern conflict: Removed `test_*.py` pattern that conflicted with tracked test files

### Improved

- **Code Quality**
  - Removed unused `pointbiserialr` import from `data_analysis.py`
  - Replaced inefficient `.iterrows()` with `.to_dict('records')` for better performance (2 instances in `data_analysis.py`)

- **Documentation**
  - Added comprehensive `FeatureSelector` class docstring with attributes and usage examples
  - Added detailed `FeatureSelector.__init__()` docstring with Args/Raises sections
  - Added input validation to `FeatureSelector.__init__()` (TypeError and empty DataFrame checks)

- **Type Hints**
  - Enhanced type hint imports in `feature_selection.py` (added Dict, Callable, Any)
  - Updated `feature_scores` type hint from `dict` to `Dict[str, Dict[str, float]]`
  - Added explicit type hints to `selected_features: List[str]`
  - Improved `score_func` parameter type hint to `Optional[Union[str, Callable]]`

- **Configuration Files**
  - Fixed `MANIFEST.in` case sensitivity issue: `claude.md` → `CLAUDE.md`

All 131 tests pass successfully. Changes maintain backward compatibility.

## [2.1.0] - 2025-11-24

### Added

- **TargetAnalyzer Class** - Comprehensive target-aware statistical analysis for ML tasks
  - **Auto Task Detection**: Automatically detects classification vs regression based on target column characteristics
  - **Initialization**: `TargetAnalyzer(df, target_column, task='auto')` with intelligent task inference

- **Phase 1: Core Infrastructure**
  - `get_task_info()`: Get detected task type and target column information
  - `analyze_class_distribution()`: Class counts, percentages, and imbalance ratios (classification)
  - `get_class_imbalance_info()`: Detailed imbalance analysis with severity levels (mild/moderate/severe/extreme)
  - `analyze_target_distribution()`: Comprehensive target statistics with optional normality tests (regression)
  - `plot_class_distribution()`: Visualize class distribution with counts and percentages
  - `plot_target_distribution()`: Target histogram with KDE and Q-Q plot for normality assessment
  - `generate_summary_report()`: Legacy formatted text report for quick analysis
  - Caching mechanism for expensive computations

- **Phase 2: Classification Statistical Tests**
  - `analyze_feature_target_relationship()`: Chi-square tests for categorical features, ANOVA F-tests for numeric features
  - `analyze_class_wise_statistics()`: Mean, median, and std of numeric features per class
  - `plot_feature_by_class()`: Box plots, violin plots, or histograms showing feature distributions by class

- **Phase 3: Regression Analysis & Correlations**
  - `analyze_feature_correlations()`: Pearson and Spearman correlations with target
  - `analyze_mutual_information()`: Feature importance via mutual information (both classification and regression)
  - `plot_feature_vs_target()`: Scatter plots with regression lines for top correlated features
  - `analyze_residuals()`: Residual analysis with MAE, RMSE, R², and normality tests
  - `plot_residuals()`: Residual plots (residuals vs predicted, Q-Q plot for residual normality)

- **Phase 4: Data Quality & Recommendations**
  - `analyze_data_quality()`: Comprehensive checks for missing values, constant features, duplicates
  - Potential data leakage detection: Perfect correlations, suspicious p-values, zero variance features
  - `calculate_vif()`: Multicollinearity detection using Variance Inflation Factor (delegates to DataAnalyzer, auto-excludes target)
  - `generate_recommendations()`: Actionable recommendations with priority levels (high/medium/low) based on all analyses

- **Phase 5: Report Generation & Export**
  - `generate_full_report()`: Structured dictionary containing all analyses (distribution, relationships, MI scores, quality, VIF, recommendations)
  - `export_report()`: Multi-format export with three options:
    - **HTML**: Professional report with CSS styling, tables, and formatting
    - **Markdown**: Well-structured markdown with tables for documentation/GitHub
    - **JSON**: Machine-readable format for programmatic access
  - Reports include all relevant analyses based on task type

- **Phase 7: Feature Engineering Suggestions**
  - `suggest_feature_engineering()`: Intelligent feature transformation recommendations
  - **Skewness-based transforms**: Log, sqrt, or polynomial transforms for skewed distributions
  - **Categorical encoding strategies**: One-hot (low cardinality), target encoding (medium), ordinal based on data characteristics
  - **Scaling recommendations**: Based on feature value ranges and distributions
  - **Non-linear relationships**: Polynomial feature suggestions for features with non-linear target relationships
  - **Interaction terms**: Suggestions for correlated features that may benefit from interactions
  - **Missing value indicators**: Binary flags for features with significant missing data
  - **Binning suggestions**: For high-cardinality numeric features in classification tasks
  - Priority-sorted suggestions (high/medium/low) with detailed reasoning

- **Phase 8: Model Recommendations**
  - `recommend_models()`: ML algorithm suggestions tailored to dataset characteristics
  - **Classification models**: Handles class imbalance (SMOTE, class weights), dimensionality, binary vs multiclass
  - **Regression models**: Considers outliers, target distribution, feature relationships, non-linearity
  - **Dataset size awareness**: Different recommendations for small (<1000), medium, and large datasets
  - **Model-specific guidance**: Hyperparameter tuning suggestions, regularization recommendations
  - **Priority-sorted**: Random Forest, XGBoost, LightGBM, Linear models, Neural Networks based on data
  - Practical considerations for each recommended model

- **DataAnalyzer Enhancements**
  - `calculate_vif()`: Variance Inflation Factor calculation for multicollinearity detection (VIF > 10 indicates high collinearity)
  - Moved from TargetAnalyzer to DataAnalyzer for better separation of concerns (VIF is target-independent)
  - TargetAnalyzer delegates to DataAnalyzer for VIF, automatically excluding target column

- **Comprehensive Test Suite**
  - 87 new tests for TargetAnalyzer (total: 131 tests across all modules)
  - `test_target_analyzer.py`: Complete coverage of all 8 phases
    - Initialization and task detection (7 tests)
    - Classification analysis (6 tests)
    - Regression analysis (5 tests)
    - Summary reports and caching (4 tests)
    - Edge cases (3 tests)
    - Phase 2: Classification statistical tests (6 tests)
    - Phase 3: Regression correlations and MI (9 tests)
    - Phase 4: Data quality and recommendations (8 tests)
    - Phase 2-4: Integration tests (2 tests)
    - Phase 5: Report generation and export (9 tests)
    - Phase 7: Feature engineering suggestions (10 tests)
    - Phase 8: Model recommendations (10 tests)

- **Documentation**
  - Comprehensive README update with TargetAnalyzer usage examples
  - API Reference documentation for all 30+ TargetAnalyzer methods
  - Updated CLAUDE.md with architecture decisions and implementation details
  - FEATURE_PLAN.md documenting the phased development approach

### Changed

- **Architecture Refactoring**
  - VIF calculation relocated from TargetAnalyzer to DataAnalyzer
  - Improved separation of concerns: general EDA (DataAnalyzer) vs target-specific analysis (TargetAnalyzer)
  - TargetAnalyzer now delegates to DataAnalyzer for VIF with automatic target exclusion

- **Dependencies**
  - Added `statsmodels>=0.14.0` for VIF calculation and advanced statistical tests

- **README**
  - Updated header to "feature-engineering-tk v2.1.0" with professional badges
  - Added Features section highlighting 8 key capabilities
  - Added "What's New in v2.1.0" section with comprehensive TargetAnalyzer documentation
  - Expanded API Reference with complete method categorization
  - Added Contributing, Support, and Links sections

### Fixed

- Minor improvements to error handling in statistical tests for edge cases (constant features, small datasets)

## [2.0.0] - 2025-11-22

### Breaking Changes

- **Inplace parameter default changed from `True` to `False`** for all methods in `DataPreprocessor` and `FeatureEngineer`
  - This aligns with pandas conventions and prevents accidental data mutations
  - Migration: Add `inplace=True` to existing method calls or refactor to use returned DataFrames
  - See README.md for detailed migration guide

### Added

- **Transformer Persistence** (`FeatureEngineer`)
  - New `save_transformers(filepath)` method to save fitted encoders and scalers
  - New `load_transformers(filepath)` method to load previously fitted transformers
  - Enables deployment of consistent transformations in production environments

- **Custom Exception Hierarchy**
  - `MLToolkitError` - Base exception class
  - `ValidationError` - For validation failures
    - `InvalidStrategyError` - Invalid strategy parameter
    - `InvalidMethodError` - Invalid method parameter
    - `ColumnNotFoundError` - Column not found in DataFrame
    - `DataTypeError` - Invalid data type
    - `EmptyDataFrameError` - Empty DataFrame error
    - `InsufficientDataError` - Insufficient data for operation
  - `TransformerNotFittedError` - Attempting to save unfitted transformers
  - `ConstantColumnError` - Operation on constant column
  - All exceptions provide clear, actionable error messages

- **Comprehensive Logging System**
  - Replaced ~40 `print()` statements with proper logging using Python's `logging` module
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR)
  - Applied across all modules for production-ready error tracking

- **Input Validation**
  - Type checking for all method parameters
  - Value range validation where applicable
  - Clear error messages with valid options
  - Prevents silent failures and data corruption

- **Comprehensive Test Suite**
  - 42 tests across 5 test files
  - `test_preprocessing.py` - 12 tests
  - `test_feature_engineering.py` - 13 tests
  - `test_data_analysis.py` - 6 tests
  - `test_exceptions.py` - 4 tests
  - `test_plotting.py` - 8 tests
  - Tests cover critical bugs, edge cases, and new features

- **Enhanced Documentation**
  - Comprehensive docstrings with Args/Returns/Raises sections
  - Developer documentation in `CLAUDE.md`
  - Migration guide in README.md

### Changed

- **Package Structure**
  - Fixed broken package structure by moving all modules to `feature_engineering_tk/` directory
  - Now properly installable via `pip install -e .`
  - Corrected `setup.py` configuration using `find_packages()`

- **Plotting Methods Return Values** (`DataAnalyzer`)
  - `plot_missing_values()` now returns `matplotlib.figure.Figure` object (or `None` if no data)
  - `plot_correlation_heatmap()` now returns `Figure` object (or `None` if insufficient data)
  - `plot_distributions()` now returns `Figure` object (or `None` if no numeric columns)
  - All plotting methods accept `show` parameter (default `True`)
  - Enables programmatic plot manipulation and saving
  - Example: `fig = analyzer.plot_missing_values(show=False); fig.savefig('plot.png')`

### Fixed

- **Critical: Inplace Operation Bugs** (9 methods affected)
  - `DataPreprocessor.convert_dtypes()` - Now correctly updates `self.df` when `inplace=True`
  - `DataPreprocessor.clip_values()` - Now correctly updates `self.df` when `inplace=True`
  - `DataPreprocessor.apply_custom_function()` - Now correctly updates `self.df` when `inplace=True`
  - `FeatureEngineer.encode_categorical_label()` - Fixed inplace behavior
  - `FeatureEngineer.encode_categorical_onehot()` - Fixed inplace behavior
  - `FeatureEngineer.encode_categorical_ordinal()` - Fixed inplace behavior
  - `FeatureEngineer.scale_features()` - Fixed inplace behavior
  - `FeatureEngineer.create_binning()` - Fixed inplace behavior
  - `FeatureEngineer.create_log_transform()` - Fixed inplace behavior
  - `FeatureEngineer.create_sqrt_transform()` - Fixed inplace behavior
  - These bugs caused silent data loss when `inplace=False` and incorrect behavior when `inplace=True`

- **Critical: Division by Zero Protection**
  - `DataAnalyzer.detect_outliers_zscore()` - Skips columns with zero standard deviation
  - `DataPreprocessor.handle_outliers()` (zscore method) - Skips constant columns
  - Prevents crashes and provides clear warning messages

- **Critical: Deprecated Pandas Methods**
  - Replaced `fillna(method='ffill')` with `ffill()`
  - Replaced `fillna(method='bfill')` with `bfill()`
  - Ensures compatibility with pandas >=2.0

### Development

- Added development dependencies: pytest, pytest-cov, black, flake8, mypy
- Set up proper package structure for pip installation
- Configured non-interactive matplotlib backend for testing

## [1.0.0] - 2025-11-20

### Added

- Initial release of Feature Engineering Toolkit
- `DataAnalyzer` - Exploratory data analysis and visualization
- `DataPreprocessor` - Data cleaning and preprocessing
- `FeatureEngineer` - Feature transformation and creation
- `FeatureSelector` - Feature selection methods
- Basic documentation and examples

[2.3.0]: https://github.com/bluelion1999/feature_engineering_tk/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/bluelion1999/feature_engineering_tk/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/bluelion1999/feature_engineering_tk/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/bluelion1999/feature_engineering_tk/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/bluelion1999/feature_engineering_tk/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/bluelion1999/feature_engineering_tk/releases/tag/v1.0.0
