# Feature Engineering Toolkit - Claude Code Development Documentation

> **Purpose**: Development context for maintaining and extending the Feature Engineering Toolkit repository.

## Project Overview

**Feature Engineering Toolkit** (package name: `feature-engineering-tk`) is a comprehensive Python library for feature engineering and data analysis to prepare dataframes for machine learning.

- **Repository**: https://github.com/bluelion1999/feature_engineering_tk
- **Default Branch**: master
- **Python Version**: 3.8+
- **Current Version**: 2.3.0
- **Last Major Refactor**: 2025-12-09 (Redundancy Reduction)

---

## Recent Major Changes

### Version 2.3.0 Release (2025-12-10)
**Status**: Completed on fly_catcher branch
**Focus**: Architecture refactoring, code quality, and performance optimizations

#### Architecture Improvements

**Refactored Codebase** (~300 lines eliminated, 6.5% reduction):
- Created `FeatureEngineeringBase` base class for all toolkit classes
- Created `utils.py` module with 7 shared utility functions
- All 5 main classes now inherit from common base
- Consolidated validation logic into reusable utilities
- Eliminated duplicate `get_dataframe()` implementations
- DataPreprocessor outlier handling now uses DataAnalyzer's detection methods

#### Performance Optimizations

**Phase 1 - Quick Wins**:
- Pre-computed means/medians in `handle_missing_values()` for better scalability
- Optimized string column validation using set-based operations
- **Fixed outlier detection**: 45% faster (221ms → 120ms) by accumulating row removals

**Phase 2 - N+1 Query Pattern Elimination**:
- **Class-wise statistics**: 7x faster (969ms → 138ms, 86% improvement)
- **Feature-target relationship**: Optimized ANOVA grouping operations
- Replaced nested filtering loops with single `groupby` operations
- Eliminates redundant DataFrame scans

**Benchmarking Infrastructure**:
- Created `benchmarks/` directory with comprehensive benchmark suite
- Baseline measurements for all critical operations
- `OPTIMIZATION_PLAN.md` documents implementation strategy and results

**Benefits**:
- Single source of truth for validation operations
- Significantly faster statistical analysis (7x improvement)
- Improved code maintainability and consistency
- 100% backward compatibility (all 182 tests passing)
- Cleaner, more organized codebase

**Technical Details**: See "Redundancy Reduction Refactoring (2025-12-09)" and "Performance Optimizations (2025-12-10)" sections below for complete details.

---

### Statistical Robustness Features (2026-01-02)
**Status**: Completed on feature/statistical-robustness branch
**Focus**: Add comprehensive statistical robustness to ensure valid, reliable analyses

#### New Module: statistical_utils.py

**Assumption Validation Functions** (4 functions):
- `check_normality()`: Shapiro-Wilk test for normality (supports large samples with subsampling)
- `check_homogeneity_of_variance()`: Levene's test for equal variances
- `validate_sample_size()`: Sample size requirements for different tests
- `check_chi2_expected_frequencies()`: Expected frequency validation for chi-square

**Effect Size Calculations** (4 functions):
- `cohens_d()`: Cohen's d for t-tests with interpretation (small/medium/large)
- `eta_squared()`: Eta-squared (η²) for ANOVA with variance explained percentage
- `cramers_v()`: Cramér's V for chi-square tests with bias correction
- `pearson_r_to_d()`: Convert Pearson r to Cohen's d for comparison

**Multiple Testing Corrections** (1 function):
- `apply_multiple_testing_correction()`: Benjamini-Hochberg FDR, Bonferroni, Holm methods

**Confidence Interval Utilities** (3 functions):
- `calculate_mean_ci()`: Parametric CI using t-distribution
- `calculate_correlation_ci()`: Fisher Z-transformation for correlation CIs
- `bootstrap_ci()`: Non-parametric bootstrap for any statistic

#### Enhanced TargetAnalyzer Methods

**analyze_feature_target_relationship()** - New parameters:
- `check_assumptions=False`: Validates normality, homogeneity with automatic fallback to Kruskal-Wallis
- `report_effect_sizes=False`: Includes eta-squared (ANOVA) or Cramér's V (chi-square)
- `correct_multiple_tests=False`: Applies Benjamini-Hochberg FDR correction
- `alpha=0.05`: Significance level

Returns additional columns:
- `pvalue_corrected`: FDR-corrected p-value
- `significant_raw`, `significant_corrected`: Before/after correction significance
- `effect_size`, `effect_interpretation`: Practical significance measures
- `assumptions_met`: Boolean flag for assumption validity
- `warnings`: Assumption violations or test recommendations

**analyze_class_wise_statistics()** - New parameters:
- `include_ci=False`: Adds confidence intervals for mean and median
- `confidence_level=0.95`: CI confidence level

Returns additional columns:
- `mean_ci_lower`, `mean_ci_upper`: Parametric CIs for means
- `median_ci_lower`, `median_ci_upper`: Bootstrap CIs for medians

**analyze_feature_correlations()** - New parameters:
- `include_ci=False`: Fisher Z-transformation CIs for correlations
- `check_linearity=False`: Compares Pearson vs Spearman to detect non-linearity
- `confidence_level=0.95`: CI confidence level

Returns additional columns:
- `ci_lower`, `ci_upper`: Correlation confidence intervals
- `linearity_warning`: Flags when Pearson/Spearman differ >0.2

#### Testing
- Added 29 comprehensive tests in `test_statistical_utils.py`
- Total: 211 tests (182 baseline + 29 new)
- Coverage: assumption checks, effect sizes, corrections, CIs, edge cases, integration tests
- All tests passing ✅

#### Backward Compatibility
- All enhancements use optional parameters defaulting to False
- 100% backward compatible - existing code unchanged
- Opt-in design allows gradual adoption

#### Benefits
- **Valid Tests**: Automatic assumption checking prevents invalid statistical conclusions
- **Controlled Errors**: Multiple testing corrections prevent false positives (5% → <1% false discovery rate)
- **Practical Significance**: Effect sizes distinguish meaningful vs trivial differences
- **Uncertainty Quantification**: Confidence intervals show estimate reliability
- **Non-parametric Fallbacks**: Automatic switch to Kruskal-Wallis when ANOVA assumptions violated

---

### Version 2.2.0 Release (2025-12-07)
**Status**: Ready for release
**Branches**: feature/preprocessor-enhancements (includes column-type-detection)

#### DataPreprocessor Enhancements

1. **String Preprocessing Methods** (3 new methods):
   - `clean_string_columns()`: 7 operations (strip, lower, upper, title, remove_punctuation, remove_digits, remove_extra_spaces)
   - `handle_whitespace_variants()`: Standardize whitespace variants in categorical columns
   - `extract_string_length()`: Create length features from string columns

2. **Data Validation Methods** (3 new methods):
   - `validate_data_quality()`: Comprehensive quality report (missing values, constant columns, infinite values)
   - `detect_infinite_values()`: Detect np.inf/-np.inf in numeric columns
   - `create_missing_indicators()`: Create binary indicator columns for missing values

3. **Method Chaining Support**:
   - All preprocessing methods now return `self` when `inplace=True`
   - Enables fluent API pattern: `preprocessor.method1().method2().method3()`

4. **Operation History Tracking**:
   - Automatic logging of all preprocessing operations when `inplace=True`
   - `get_preprocessing_summary()`: Returns formatted text summary
   - `export_summary()`: Export to text/markdown/JSON formats
   - Tracks: timestamps, parameters, shape changes, method-specific details

5. **Enhanced Error Handling**:
   - Better parameter validation across all methods
   - Warnings for destructive operations (e.g., removing >30% of data)
   - Improved logging throughout

**Test Coverage**: Added 42 new tests (now 173 total tests)
- 7 tests for string preprocessing
- 6 tests for data validation
- 6 tests for enhanced error handling
- 6 tests for method chaining
- 17 tests for operation history tracking

#### DataAnalyzer Enhancements (Column Type Detection)

1. **New Methods** (2 total):
   - `detect_misclassified_categorical()`: Identifies numeric columns that should be categorical
     - Binary/flag columns (exactly 2 unique values)
     - Low cardinality numeric columns (≤10 unique values by default)
     - Columns with very low unique ratios (many repeated values)
     - Integer columns with moderate cardinality (≤20 values)
   - `suggest_binning()`: Recommends binning strategies based on distribution characteristics
     - Quantile binning for skewed distributions (abs(skewness) > 1.0)
     - Uniform binning for relatively uniform distributions
     - Handles outlier-heavy columns with quantile strategy
     - Suggests appropriate number of bins (min 20 unique values required)

2. **Enhanced `quick_analysis()` Function**:
   - New "MISCLASSIFIED CATEGORICAL COLUMNS" section
   - New "BINNING SUGGESTIONS" section with actionable tips

3. **Benefits**:
   - Helps identify data type misclassifications during EDA
   - Provides intelligent binning recommendations without requiring a target column
   - Complements TargetAnalyzer's target-dependent suggestions (which require a target)

**Testing** (CLAUDE.md only - DO NOT include in README):
- Added 9 new tests (now 17 total tests in test_data_analysis.py)
- 4 tests for categorical detection (binary, integer, low ratio, edge cases)
- 5 tests for binning suggestions (skewed, uniform, outliers, thresholds, edge cases)
- All tests passing ✅

### Redundancy Reduction Refactoring (2025-12-09)
**Status**: Completed on fly_catcher branch
**Impact**: ~300 lines of redundant code eliminated, improved maintainability

#### New Shared Infrastructure

1. **FeatureEngineeringBase Base Class** (`base.py`):
   - Shared `__init__()` method with DataFrame validation and copying
   - Shared `get_dataframe()` method
   - All 5 main classes inherit from this base
   - Eliminates ~50 lines of duplicate initialization code

2. **Utility Functions Module** (`utils.py`):
   - `validate_and_copy_dataframe()`: DataFrame validation and copying
   - `validate_columns()`: Column existence validation with options
   - `get_numeric_columns()`: Extract numeric columns from DataFrame
   - `validate_numeric_columns()`: Validate and filter numeric columns
   - `get_string_columns()`: Extract string/object columns
   - `get_feature_columns()`: Get feature columns with exclusions
   - Eliminates ~250 lines of duplicate validation code

3. **@inplace_transform Decorator** (`base.py`):
   - Handles inplace transformation pattern automatically
   - Available for future use in simplifying method implementations

#### Refactored Classes

1. **DataPreprocessor** (30+ methods updated):
   - Inherits from FeatureEngineeringBase
   - Uses utility functions throughout (22 validation patterns replaced)
   - `handle_outliers()` now uses DataAnalyzer detection methods
   - Net reduction: 45 lines removed

2. **FeatureEngineer** (12+ methods updated):
   - Inherits from FeatureEngineeringBase
   - Removed duplicate `get_dataframe()` method
   - Uses utility functions for validation
   - Net reduction: 41 lines removed

3. **DataAnalyzer**:
   - Inherits from FeatureEngineeringBase
   - Uses utility functions for numeric column selection (8 methods)
   - Removed duplicate `__init__` method

4. **TargetAnalyzer**:
   - Inherits from FeatureEngineeringBase
   - Simplified `__init__` to call `super()`
   - Uses `get_feature_columns()` utility for target exclusion

5. **FeatureSelector**:
   - Inherits from FeatureEngineeringBase
   - Removed internal `_get_feature_columns()` helper
   - Uses shared `get_feature_columns()` utility

#### Benefits

- **Code Reduction**: ~300 lines of redundant code eliminated (6.5% of codebase)
- **Single Source of Truth**: Validation logic centralized in utils.py
- **Consistency**: All classes use identical validation patterns
- **Maintainability**: Changes to validation need only be made once
- **Testing**: Utility functions can be tested independently
- **Outlier Detection**: DataPreprocessor delegates to DataAnalyzer (DRY principle)
- **All 182 tests passing**: 100% backward compatibility maintained

### Architecture Refactoring (2025-11-24)
1. **VIF Relocation**: Moved `calculate_vif()` from TargetAnalyzer to DataAnalyzer
   - VIF is target-independent multicollinearity detection
   - TargetAnalyzer now delegates to DataAnalyzer (auto-excludes target column)
   - Improved separation of concerns: general EDA vs target-specific analysis

### Major Refactoring (2025-11-22)

#### Critical Fixes
1. **Package Structure**: Fixed broken structure - moved all modules to `feature_engineering_tk/` directory
2. **Inplace Bugs** (9 methods): Fixed methods not updating `self.df` when `inplace=True`
3. **Division by Zero**: Added checks in z-score outlier detection
4. **Deprecated Methods**: Replaced `fillna(method=...)` with `ffill()`/`bfill()`

### High Priority Improvements
1. **Logging**: Added comprehensive logging system using `logger` instead of `print()`
2. **Inplace Default**: Changed from `True` to `False` (BREAKING CHANGE - matches pandas conventions)
3. **Input Validation**: Added type/value/range checking throughout
4. **Transformer Persistence**: New `save_transformers()`/`load_transformers()` methods
5. **Documentation**: Comprehensive docstrings with Args/Returns/Raises

### Medium Priority Improvements
1. **Custom Exceptions**: Created exception hierarchy (`InvalidStrategyError`, `TransformerNotFittedError`, etc.)
2. **Plotting Returns Figures**: All plot methods now return `Figure` objects with `show` parameter
3. **Test Suite**: 43 comprehensive tests covering all fixes

---

## Project Structure

```
mltoolkit/
├── feature_engineering_tk/    # Main package
│   ├── __init__.py
│   ├── base.py                # Base class and decorators (NEW)
│   ├── utils.py               # Shared utility functions (NEW)
│   ├── data_analysis.py       # EDA and visualization
│   ├── feature_engineering.py # Feature transformation
│   ├── preprocessing.py       # Data cleaning
│   ├── feature_selection.py   # Feature selection
│   └── exceptions.py          # Custom exceptions
├── tests/                     # Test suite (131 tests)
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_data_analysis.py
│   ├── test_target_analyzer.py  # NEW: Phase 1 TargetAnalyzer tests
│   ├── test_exceptions.py
│   └── test_plotting.py
├── setup.py
├── README.md
└── claude.md                  # This file
```

---

## Core Design Patterns

### Class Constructor Pattern
```python
class ClassName:
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            logger.warning("Initializing with empty DataFrame")
        self.df = df.copy()  # Always copy to prevent mutations
```

### Method Signature Pattern
```python
def method_name(self,
                columns: Union[str, List[str]],
                inplace: bool = False) -> Union[pd.DataFrame, 'ClassName']:
    """
    Brief description.

    Args:
        columns: Column names
        inplace: If True, modifies internal dataframe. Default False.

    Returns:
        Self if inplace=True (enables chaining), otherwise modified DataFrame copy

    Raises:
        InvalidStrategyError: If strategy invalid
    """
    # 1. Type validation
    if not isinstance(columns, list):
        raise TypeError("columns must be a list")

    # 2. Capture shape before operation (for history tracking)
    rows_before, cols_before = self.df.shape

    # 3. Handle inplace
    df_result = self.df if inplace else self.df.copy()

    # 4. Validate data
    invalid_cols = [col for col in columns if col not in df_result.columns]
    if invalid_cols:
        logger.warning(f"Columns not found: {invalid_cols}")
        columns = [col for col in columns if col in df_result.columns]

    # 5. Edge cases
    if not columns:
        logger.warning("No valid columns to process")
        return df_result if not inplace else self

    # 6. Transform
    # ... actual logic ...

    # 7. Return (CRITICAL: Update self.df when inplace=True, return self for chaining)
    if inplace:
        rows_after, cols_after = df_result.shape
        # Log operation for history tracking (DataPreprocessor only)
        if hasattr(self, '_log_operation'):
            self._log_operation(
                method_name='method_name',
                parameters={'columns': columns},
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after
            )
        self.df = df_result
        return self  # NEW v2.2.0: Return self for method chaining
    return df_result
```

### Error Handling Pattern
```python
# Type errors → TypeError
if not isinstance(df, pd.DataFrame):
    raise TypeError("Input must be a pandas DataFrame")

# Invalid values → Custom exceptions
if strategy not in valid_strategies:
    raise InvalidStrategyError(strategy, valid_strategies)

# Missing data → Log warning, continue
if col not in df.columns:
    logger.warning(f"Column '{col}' not found")
    continue

# Insufficient data → Return early
if not columns:
    logger.warning("No valid columns")
    return df_result if not inplace else self.df
```

---

## Module Details

### base.py (NEW)
**Foundation module providing shared infrastructure**

**FeatureEngineeringBase Class**:
- Base class for all toolkit classes
- Shared `__init__(df)`: DataFrame validation and copying
- Shared `get_dataframe()`: Returns copy of internal DataFrame
- Inherited by: DataPreprocessor, FeatureEngineer, DataAnalyzer, TargetAnalyzer, FeatureSelector

**@inplace_transform Decorator**:
- Handles inplace transformation pattern automatically
- Available for future use in method simplification
- Manages DataFrame copying and return value logic

### utils.py (NEW)
**Utility functions for validation and column selection**

**DataFrame Operations**:
- `validate_and_copy_dataframe(df)`: Validates DataFrame type and creates copy

**Column Validation**:
- `validate_columns(df, columns, raise_on_missing)`: Returns valid columns, logs/raises for invalid
- `validate_numeric_columns(df, columns)`: Returns only numeric columns
- `get_string_columns(df, columns)`: Returns only string/object columns

**Column Selection**:
- `get_numeric_columns(df, columns)`: Extracts numeric columns
- `get_feature_columns(df, exclude_columns, numeric_only)`: Gets feature columns with exclusions

**Usage**: All validation and column selection operations throughout the codebase use these utilities for consistency.

### statistical_utils.py (NEW)
**Statistical robustness utilities for assumption validation, effect sizes, and confidence intervals**

**Assumption Validation Functions**:
- `check_normality(data, method='shapiro', alpha=0.05)`: Tests normality assumption
  - Returns: `{'is_normal': bool, 'pvalue': float, 'recommendation': str, 'sample_size': int}`
  - Handles large samples (>5000) with automatic subsampling
  - Supports: shapiro, normaltest, anderson

- `check_homogeneity_of_variance(groups, method='levene', alpha=0.05)`: Tests equal variance assumption
  - Returns: `{'equal_variances': bool, 'pvalue': float, 'recommendation': str}`
  - Recommends standard ANOVA, Welch's ANOVA, or non-parametric tests

- `validate_sample_size(groups, test_type='anova', min_size=30)`: Validates sample size requirements
  - Returns: `{'sufficient': bool, 'actual_sizes': List[int], 'warning': Optional[str]}`
  - Test types: anova, ttest, chi2, correlation

- `check_chi2_expected_frequencies(contingency_table, min_expected=5)`: Validates chi-square assumptions
  - Returns: `{'valid': bool, 'min_expected': float, 'percent_cells_below_threshold': float}`
  - Recommends Fisher's exact test for 2×2 tables with low expected frequencies

**Effect Size Calculations**:
- `cohens_d(group1, group2, pooled=True)`: Cohen's d for t-tests
  - Returns: `{'cohens_d': float, 'interpretation': str, 'description': str}`
  - Interpretations: small (0.2), medium (0.5), large (0.8)

- `eta_squared(groups, f_statistic, df_between, df_within)`: Eta-squared for ANOVA
  - Returns: `{'eta_squared': float, 'interpretation': str, 'percent_variance_explained': float}`
  - Alternative: can calculate from groups directly

- `cramers_v(contingency_table, correction=True)`: Cramér's V for chi-square
  - Returns: `{'cramers_v': float, 'interpretation': str, 'chi2_statistic': float, 'pvalue': float}`
  - Bias correction for small samples

- `pearson_r_to_d(r)`: Converts Pearson r to Cohen's d for comparison
  - Formula: d = 2r / √(1-r²)

**Multiple Testing Corrections**:
- `apply_multiple_testing_correction(pvalues, method='fdr_bh', alpha=0.05)`: Apply corrections
  - Returns: `{'corrected_pvalues': array, 'reject': array, 'num_significant_raw': int, 'num_significant_corrected': int}`
  - Methods: fdr_bh (Benjamini-Hochberg FDR), bonferroni, holm
  - Uses statsmodels.stats.multitest

**Confidence Interval Utilities**:
- `calculate_mean_ci(data, confidence=0.95)`: Parametric CI for mean
  - Returns: `{'mean': float, 'ci_lower': float, 'ci_upper': float, 'margin_of_error': float}`
  - Uses t-distribution

- `calculate_correlation_ci(r, n, confidence=0.95)`: CI for Pearson correlation
  - Returns: `{'r': float, 'ci_lower': float, 'ci_upper': float}`
  - Uses Fisher Z-transformation

- `bootstrap_ci(data, statistic_func, n_bootstrap=1000, confidence=0.95, random_state=None)`: Non-parametric bootstrap CI
  - Returns: `{'statistic': float, 'ci_lower': float, 'ci_upper': float, 'bootstrap_distribution': array}`
  - Works with any statistic function (e.g., np.median, custom functions)

**Usage**: Import as `from feature_engineering_tk import statistical_utils` or use via enhanced TargetAnalyzer methods.

### data_analysis.py
**Classes**: `DataAnalyzer`, `TargetAnalyzer` (read-only, no inplace operations)

**DataAnalyzer - Key Features**:
- Basic info, missing value analysis
- Outlier detection (IQR, Z-score with division-by-zero protection)
- Correlation analysis (Pearson, Spearman)
- **VIF calculation** for multicollinearity detection (VIF > 10 = high collinearity)
- Cardinality analysis
- **NEW (v2.2.0)**: Categorical column detection - identifies numeric columns that should be categorical
- **NEW (v2.2.0)**: Binning suggestions - recommends binning strategies based on distribution
- Plotting methods that return `Figure` objects

**TargetAnalyzer - Phases 1-5, 7-8 Complete**:
**State**: `self.task` (auto-detected or specified), `self._analysis_cache` (dict)

**Phase 1 - Core Infrastructure**:
- Auto task detection (classification vs regression)
- Classification: class distribution, imbalance analysis, severity levels
- Regression: comprehensive stats (mean, median, skewness, kurtosis, normality tests)
- Basic plotting methods (class/target distributions, Q-Q plots)
- Caching mechanism

**Phase 2 - Classification Statistical Tests**:
- Feature-target relationship analysis (Chi-square, ANOVA F-test)
- Class-wise feature statistics (mean, median, std per class)
- Feature distribution plotting by class (box, violin, histogram)

**Phase 3 - Regression Analysis**:
- Correlation analysis (Pearson, Spearman)
- Mutual information scores (classification and regression)
- Scatter plots with regression lines
- Residual analysis (MAE, RMSE, R², normality tests)
- Residual plots (residuals vs predicted, Q-Q plot)

**Phase 4 - Data Quality & Recommendations**:
- Comprehensive data quality checks (missing values, constant features)
- Potential data leakage detection (perfect correlations, suspicious p-values)
- Multicollinearity detection (`calculate_vif()` - delegates to DataAnalyzer, auto-excludes target)
- Actionable recommendation engine with severity levels

**Phase 5 - Report Generation & Export**:
- `generate_full_report()`: Structured dict with all analyses (distribution, relationships, MI, quality, VIF, recommendations)
- `export_report()`: Multi-format export (HTML with CSS, Markdown with tables, JSON)
- Comprehensive reports combining all Phase 1-4 analyses in user-friendly formats

**Phase 7 - Feature Engineering Suggestions**:
- `suggest_feature_engineering()`: Intelligent recommendations for feature transformations
- Skewness-based transform suggestions (log, sqrt, polynomial)
- Categorical encoding strategies (one-hot, target, ordinal based on cardinality)
- Scaling recommendations based on value ranges
- Non-linear relationship detection (polynomial features)
- Interaction term suggestions for correlated features
- Missing value indicator recommendations
- Priority-sorted actionable suggestions (high/medium/low)

**Phase 8 - Model Recommendations**:
- `recommend_models()`: ML algorithm suggestions based on data characteristics
- Classification: handles imbalance, dimensionality, binary/multiclass
- Regression: considers outliers, target distribution, feature relationships
- Dataset size awareness (small/medium/large)
- Model-specific considerations and tuning guidance
- Priority-sorted recommendations (Random Forest, XGBoost, LightGBM, Linear models, Neural Networks)
- Practical guidance on model selection and hyperparameter tuning

**Usage Pattern**:
```python
# Initialize
analyzer = TargetAnalyzer(df, target_column='target', task='auto')

# Phase 1: Basic analysis
dist = analyzer.analyze_class_distribution()
imbalance = analyzer.get_class_imbalance_info()

# Phase 2: Feature relationships
relationships = analyzer.analyze_feature_target_relationship()
class_stats = analyzer.analyze_class_wise_statistics()
fig = analyzer.plot_feature_by_class('feature1', plot_type='box', show=False)

# Phase 3: Correlations & MI
correlations = analyzer.analyze_feature_correlations(method='pearson')
mi_scores = analyzer.analyze_mutual_information()
fig = analyzer.plot_feature_vs_target(max_features=6, show=False)

# Residual analysis (with predictions)
residuals = analyzer.analyze_residuals(predictions)
fig = analyzer.plot_residuals(predictions, show=False)

# Phase 4: Data quality
quality = analyzer.analyze_data_quality()
vif = analyzer.calculate_vif()
recommendations = analyzer.generate_recommendations()

# Phase 5: Report generation and export
report_dict = analyzer.generate_full_report()  # Structured dict with all analyses
analyzer.export_report('analysis_report.html', format='html')  # HTML with CSS
analyzer.export_report('analysis_report.md', format='markdown')  # Markdown with tables
analyzer.export_report('analysis_report.json', format='json')  # JSON for programmatic use

# Phase 7: Feature engineering suggestions
suggestions = analyzer.suggest_feature_engineering()
for sugg in suggestions:
    print(f"{sugg['priority'].upper()}: {sugg['feature']} - {sugg['suggestion']}")
    print(f"  Reason: {sugg['reason']}")

# Phase 8: Model recommendations
model_recs = analyzer.recommend_models()
for rec in model_recs:
    print(f"{rec['priority'].upper()}: {rec['model']}")
    print(f"  Why: {rec['reason']}")
    print(f"  Note: {rec['considerations']}")

# Legacy: Quick summary report (Phase 1)
report = analyzer.generate_summary_report()
```

**Helper**: `quick_analysis(df)` - prints formatted analysis

### preprocessing.py
**Class**: `DataPreprocessor`

**State**: `self.df` (DataFrame), `self._operation_history` (List[Dict])

**Key Features**:
- Missing values: 8 strategies (drop, mean, median, mode, ffill, bfill, interpolate, fill_value)
- Duplicates, outliers (with z-score div-by-zero protection)
- Type conversion, clipping, filtering
- Column operations (drop, rename, reorder)
- **NEW (v2.2.0)**: String preprocessing (clean_string_columns, handle_whitespace_variants, extract_string_length)
- **NEW (v2.2.0)**: Data validation (validate_data_quality, detect_infinite_values, create_missing_indicators)
- **NEW (v2.2.0)**: Method chaining support (returns `self` when `inplace=True`)
- **NEW (v2.2.0)**: Operation history tracking and summary export

**Operation History Tracking (NEW v2.2.0)**:
- Automatically logs all preprocessing operations when `inplace=True`
- Tracks: method name, timestamp, parameters, shape changes, additional details
- `get_preprocessing_summary()`: Returns formatted text summary of all operations
- `export_summary(filepath, format)`: Export to text/markdown/JSON formats
- Enables full reproducibility and documentation of preprocessing pipelines

**All methods have `inplace=False` default**

**Usage Pattern**:
```python
# Initialize
preprocessor = DataPreprocessor(df)

# Method chaining (NEW v2.2.0)
preprocessor\
    .handle_missing_values(strategy='mean', inplace=True)\
    .remove_duplicates(inplace=True)\
    .clean_string_columns(['name'], operations=['strip', 'lower'], inplace=True)\
    .drop_columns(['id'], inplace=True)

# Get preprocessing summary (NEW v2.2.0)
summary = preprocessor.get_preprocessing_summary()
print(summary)

# Export summary to file (NEW v2.2.0)
preprocessor.export_summary('preprocessing_report.md', format='markdown')
preprocessor.export_summary('preprocessing_report.json', format='json')

# Data validation (NEW v2.2.0)
quality_report = preprocessor.validate_data_quality()
infinite_vals = preprocessor.detect_infinite_values()
preprocessor.create_missing_indicators(['age', 'income'], inplace=True)

# String preprocessing (NEW v2.2.0)
preprocessor.clean_string_columns(['name', 'city'],
                                  operations=['strip', 'lower', 'remove_punctuation'],
                                  inplace=True)
preprocessor.handle_whitespace_variants(['category'], inplace=True)
preprocessor.extract_string_length(['description'], suffix='_len', inplace=True)
```

### feature_engineering.py
**Class**: `FeatureEngineer`

**State**: `self.encoders` (dict), `self.scalers` (dict)

**Key Features**:
- Encoding: label, one-hot, ordinal
- Scaling: standard, minmax, robust
- Feature creation: polynomial, binning, log/sqrt transforms
- Datetime extraction
- Aggregations, ratios, flags
- **NEW**: `save_transformers()` / `load_transformers()` for production

**All methods have `inplace=False` default**

### feature_selection.py
**Class**: `FeatureSelector`

**State**: `self.selected_features`, `self.feature_scores`

**Key Features**:
- Variance, correlation, target correlation
- Statistical tests (F-test, mutual info, chi2)
- Tree-based importance
- Missing value filtering

**Helper**: `select_features_auto()` - 3-step selection pipeline

### exceptions.py (NEW)
Custom exception hierarchy:
- `MLToolkitError` (base)
  - `ValidationError`
    - `InvalidStrategyError`
    - `InvalidMethodError`
    - `ColumnNotFoundError`
    - `DataTypeError`
    - etc.
  - `TransformerNotFittedError`

---

## Implementation Notes

### Sklearn Integration
Uses sklearn for transformers, stores them for reuse:
```python
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])
self.scalers[f"{method}_scaler"] = scaler  # Store for save/load
```

### Plotting Methods
All return `Figure` or `None`:
```python
fig = analyzer.plot_missing_values(show=False)
if fig:
    fig.savefig('plot.png')
    plt.close(fig)
```

### Feature Naming Conventions
- Polynomial: `{col}_squared`, `{col}_cubed`
- Interactions: `{col1}_x_{col2}`
- Transforms: `{col}_log`, `{col}_sqrt`
- Binning: `{col}_binned`
- Datetime: `{col}_year`, `{col}_month`, etc.
- Aggregations: `{agg_col}_{groupby}_{func}`
- Ratios: `{num}_to_{denom}_ratio`
- Flags: `{col}_flag`

### Division by Zero Protection
```python
col_std = df[col].std()
if col_std == 0:
    logger.warning(f"Column '{col}' has zero std, skipping")
    continue
z_scores = np.abs((df[col] - df[col].mean()) / col_std)
```

---

## Testing

**211 tests** across 7 test files:
- `test_preprocessing.py`: 53 tests
  - 11 core tests (inplace bugs, deprecated methods, div-by-zero)
  - 7 string preprocessing tests (v2.2.0)
  - 6 data validation tests (v2.2.0)
  - 6 enhanced error handling tests (v2.2.0)
  - 6 method chaining tests (v2.2.0)
  - 17 operation history tracking tests (v2.2.0)
- `test_feature_engineering.py`: 13 tests (inplace bugs, transformer persistence)
- `test_data_analysis.py`: 17 tests (div-by-zero protection, VIF calculation, **NEW v2.2.0**: categorical detection, binning suggestions)
- `test_statistical_utils.py`: 29 tests (**NEW 2026-01-02**: assumption checks, effect sizes, multiple testing corrections, confidence intervals, edge cases, integration tests)
- `test_target_analyzer.py`: 87 tests (Phases 1-5,7-8: task detection, statistical tests, correlations, MI, VIF delegation, data quality, recommendations, report generation, feature engineering suggestions, model recommendations, integration tests)
- `test_exceptions.py`: 4 tests (custom exception messages)
- `test_plotting.py`: 8 tests (figure returns, save capability)

**Run tests**:
```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/ --cov=feature_engineering_tk --cov-report=html
```

---

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
statsmodels>=0.14.0  # NEW: for VIF calculation in TargetAnalyzer
```

Dev dependencies: pytest, pytest-cov, black, flake8, mypy

---

## Key Design Principles

1. **Always copy DataFrames** in constructors (`df.copy()`)
2. **Inplace defaults to False** (matches pandas)
3. **Validate then transform** (fail fast with clear errors)
4. **Log, don't print** (except user-facing output functions)
5. **Return self when inplace=True** (enables method chaining) - NEW v2.2.0
6. **Store fitted transformers** (enables production deployment)
7. **Track operations automatically** (DataPreprocessor logs when inplace=True) - NEW v2.2.0

---

## Breaking Changes from Original

1. **Inplace default**: Changed from `True` → `False`
2. **Exceptions**: Some `ValueError` → Custom exceptions (programmatically catchable)
3. **Plotting**: Now returns `Figure` objects (was `None`)
4. **Return value change (v2.2.0)**: Methods now return `self` when `inplace=True` (was `self.df`)
   - **Impact**: Code expecting DataFrame when inplace=True needs update
   - **Benefit**: Enables method chaining

Migration:
```python
# OLD (implicit inplace=True):
preprocessor.handle_missing_values(strategy='mean')

# NEW (explicit):
preprocessor.handle_missing_values(strategy='mean', inplace=True)

# v2.2.0 Return Value Change:
# OLD (before v2.2.0):
result = preprocessor.handle_missing_values(strategy='mean', inplace=True)
# result was a DataFrame (self.df)

# NEW (v2.2.0+):
result = preprocessor.handle_missing_values(strategy='mean', inplace=True)
# result is now DataPreprocessor (self), enables chaining:
preprocessor.handle_missing_values(strategy='mean', inplace=True)\
           .remove_duplicates(inplace=True)\
           .drop_columns(['id'], inplace=True)
```

---

## Common Pitfalls

1. **Don't forget to update `self.df`** when `inplace=True`
2. **Return self, not self.df** when `inplace=True` (v2.2.0+)
3. **Log operation to history** after updating self.df (DataPreprocessor only, v2.2.0+)
4. **Check for zero std/variance** before division
5. **Validate numeric columns** before math operations
6. **Use custom exceptions** for better error handling
7. **Close plot figures** to prevent memory leaks: `plt.close(fig)`

---

## Future Development Guidelines

### Adding New Features
1. Follow the method signature pattern
2. Add `inplace=False` parameter
3. Use custom exceptions
4. Add comprehensive docstring
5. Write tests
6. Update this document

### Code Style
- Type hints on parameters
- Logging instead of print (except user output)
- Defensive validation
- Clear, descriptive variable names

### README Guidelines
- **CRITICAL: Do NOT include ANY testing information in README**
  - NO test counts (e.g., "Added 9 tests", "173 total tests")
  - NO test coverage percentages or details
  - NO pytest commands or testing instructions
  - NO mentions of "All tests passing" or test status
- **Testing documentation belongs ONLY in CLAUDE.md**, never in user-facing README
- **README is for end-users**, not developers:
  - Focus on features, usage examples, and API documentation
  - Show what users can do with the library
  - Include installation, quick start, and usage examples
  - Document public API methods and parameters
- **Keep README focused on user value**, not internal development practices
- When documenting new features, describe WHAT they do and HOW to use them, not how they're tested

---

## Performance Optimizations (2025-12-10)

### Overview
Comprehensive performance optimization focused on eliminating bottlenecks in statistical analysis operations. Achieved 7x improvement for class-wise statistics and 45% improvement for outlier detection.

### Implementation

**Phase 1: Quick Wins** (Commits: `76208a1`)
1. **Pre-compute aggregations** in `preprocessing.py`:
   ```python
   # Before (computed during fillna):
   df_result[numeric_cols] = df_result[numeric_cols].fillna(df_result[numeric_cols].mean())

   # After (pre-computed once):
   means = df_result[numeric_cols].mean()
   df_result[numeric_cols] = df_result[numeric_cols].fillna(means)
   ```

2. **Optimized string validation** in `utils.py`:
   - Use set-based column existence checks
   - Avoid DataFrame subsetting for dtype validation
   - Cleaner validation logic with batch warnings

3. **Fixed outlier detection** in `preprocessing.py`:
   - Accumulate rows to remove instead of removing in loop
   - Eliminates index alignment issues
   - Single removal operation at end
   - **Result**: 45% faster (221ms → 120ms)

**Phase 2: N+1 Query Pattern Elimination** (Commits: `6b954e6`)
1. **Optimized class-wise statistics** in `data_analysis.py` (line 1020):
   ```python
   # Before (N+1 pattern - filters entire DataFrame for each class):
   for feature in feature_columns:
       for cls in classes:
           class_data = self.df[self.df[self.target_column] == cls][feature]
           # Compute stats...

   # After (single groupby):
   grouped = self.df.groupby(self.target_column)
   for feature in feature_columns:
       stats_df = grouped[feature].agg([
           ('count', 'count'),
           ('mean', 'mean'),
           ('median', lambda x: x.quantile(0.5)),
           ('std', 'std'),
           ('min', 'min'),
           ('max', 'max')
       ])
   ```
   - **Result**: 7x faster (969ms → 138ms, 86% improvement)

2. **Optimized feature-target relationship** in `data_analysis.py` (lines 938, 978):
   ```python
   # Before (filters for each class/category):
   groups = [self.df[self.df[column] == value][feature].dropna() for value in unique_values]

   # After (single groupby):
   groups = [group.dropna() for _, group in self.df.groupby(column)[feature]]
   ```

**Phase 3: Copy-on-Write** (Deferred)
- High implementation complexity (40+ methods to modify)
- High risk of introducing bugs
- Limited benefit for typical use cases (most operations modify data)
- Decision: Defer for future consideration

### Benchmarking Infrastructure

**Created Files**:
- `benchmarks/benchmark_suite.py`: Comprehensive benchmark suite
- `benchmarks/__init__.py`: Package initialization
- `OPTIMIZATION_PLAN.md`: Detailed optimization strategy and results
- `baseline_results.json`: Performance baseline measurements

**Benchmark Results**:
```
Operation                           Before    After    Improvement
----------------------------------------------------------------
Class-wise statistics (100K rows)   969ms     138ms    7x faster
Outlier detection (5 columns)       221ms     120ms    45% faster
String validation (12 columns)      0.03ms    0.03ms   Maintained
DataFrame init (1M rows)            231ms     255ms    Within variance
```

### Testing
- All 182 tests passing ✅
- 100% backward compatibility maintained
- No API changes required

### Files Modified
- `feature_engineering_tk/preprocessing.py`: Outlier detection, mean pre-compute
- `feature_engineering_tk/utils.py`: String validation optimization
- `feature_engineering_tk/data_analysis.py`: N+1 pattern fixes in TargetAnalyzer
- `benchmarks/benchmark_suite.py`: New benchmarking infrastructure
- `OPTIMIZATION_PLAN.md`: Optimization documentation

---

## Git Configuration

- **Remote**: origin → https://github.com/bluelion1999/feature_engineering_tk
- **Default Branch**: master (not main)
- **Branch Naming**: `master` and `main` are interchangeable terms when referring to the default branch
- **Initial Commit**: c146f94

---

**For Claude Code**: This document provides essential context for maintaining consistency. When making changes, update relevant sections to reflect new patterns or decisions.
