# MLToolkit - Claude Code Development Documentation

> **Purpose**: Development context for maintaining and extending the MLToolkit repository.

## Project Overview

**MLToolkit** is a Python library for feature engineering and data analysis to prepare dataframes for machine learning.

- **Repository**: https://github.com/bluelion1999/feature_engineering_tk
- **Default Branch**: master
- **Python Version**: 3.8+
- **Last Major Refactor**: 2025-11-22

---

## Recent Major Changes

### DataPreprocessor Enhancements (v2.2.0 - 2025-11-30)
**Branch**: feature/preprocessor-enhancements

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

### Column Type Detection Enhancements (v2.2.0 - In Development, NOT YET RELEASED)
**Branch**: feature/column-type-detection

**User-Facing Features** (for README):
1. **DataAnalyzer Enhancements** (2 new methods):
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

**182 tests** across 6 test files:
- `test_preprocessing.py`: 53 tests
  - 11 core tests (inplace bugs, deprecated methods, div-by-zero)
  - 7 string preprocessing tests (v2.2.0)
  - 6 data validation tests (v2.2.0)
  - 6 enhanced error handling tests (v2.2.0)
  - 6 method chaining tests (v2.2.0)
  - 17 operation history tracking tests (v2.2.0)
- `test_feature_engineering.py`: 13 tests (inplace bugs, transformer persistence)
- `test_data_analysis.py`: 17 tests (div-by-zero protection, VIF calculation, **NEW v2.2.0**: categorical detection, binning suggestions)
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

## Git Configuration

- **Remote**: origin → https://github.com/bluelion1999/feature_engineering_tk
- **Default Branch**: master (not main)
- **Branch Naming**: `master` and `main` are interchangeable terms when referring to the default branch
- **Initial Commit**: c146f94

---

**For Claude Code**: This document provides essential context for maintaining consistency. When making changes, update relevant sections to reflect new patterns or decisions.
