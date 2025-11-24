# MLToolkit - Claude Code Development Documentation

> **Purpose**: Development context for maintaining and extending the MLToolkit repository.

## Project Overview

**MLToolkit** is a Python library for feature engineering and data analysis to prepare dataframes for machine learning.

- **Repository**: https://github.com/bluelion1999/feature_engineering_tk
- **Default Branch**: master
- **Python Version**: 3.8+
- **Last Major Refactor**: 2025-11-22

---

## Recent Major Refactoring (2025-11-22)

### Critical Fixes
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
├── tests/                     # Test suite (109 tests)
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
                inplace: bool = False) -> pd.DataFrame:  # Default False
    """
    Brief description.

    Args:
        columns: Column names
        inplace: If True, modifies internal dataframe. Default False.

    Returns:
        Modified DataFrame

    Raises:
        InvalidStrategyError: If strategy invalid
    """
    # 1. Type validation
    if not isinstance(columns, list):
        raise TypeError("columns must be a list")

    # 2. Handle inplace
    df_result = self.df if inplace else self.df.copy()

    # 3. Validate data
    invalid_cols = [col for col in columns if col not in df_result.columns]
    if invalid_cols:
        logger.warning(f"Columns not found: {invalid_cols}")
        columns = [col for col in columns if col in df_result.columns]

    # 4. Edge cases
    if not columns:
        logger.warning("No valid columns to process")
        return df_result if not inplace else self.df

    # 5. Transform
    # ... actual logic ...

    # 6. Return (CRITICAL: Update self.df when inplace=True)
    if inplace:
        self.df = df_result
        return self.df
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
- Correlation analysis
- Plotting methods that return `Figure` objects

**TargetAnalyzer - Phases 1-5 Complete**:
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
- Multicollinearity detection (VIF calculation)
- Actionable recommendation engine with severity levels

**Phase 5 - Report Generation & Export**:
- `generate_full_report()`: Structured dict with all analyses (distribution, relationships, MI, quality, VIF, recommendations)
- `export_report()`: Multi-format export (HTML with CSS, Markdown with tables, JSON)
- Comprehensive reports combining all Phase 1-4 analyses in user-friendly formats

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

# Legacy: Quick summary report (Phase 1)
report = analyzer.generate_summary_report()
```

**Helper**: `quick_analysis(df)` - prints formatted analysis

### preprocessing.py
**Class**: `DataPreprocessor`

**Key Features**:
- Missing values: 8 strategies (drop, mean, median, mode, ffill, bfill, interpolate, fill_value)
- Duplicates, outliers (with z-score div-by-zero protection)
- Type conversion, clipping, filtering
- Column operations (drop, rename, reorder)

**All methods have `inplace=False` default**

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

**100+ tests** across 6 test files:
- `test_preprocessing.py`: 12 tests (inplace bugs, deprecated methods, div-by-zero)
- `test_feature_engineering.py`: 13 tests (inplace bugs, transformer persistence)
- `test_data_analysis.py`: 6 tests (div-by-zero protection)
- `test_target_analyzer.py`: 67 tests (Phases 1-5: task detection, statistical tests, correlations, MI, VIF, data quality, recommendations, report generation, integration tests)
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
5. **Return DataFrame always** (enables method chaining at pandas level)
6. **Store fitted transformers** (enables production deployment)

---

## Breaking Changes from Original

1. **Inplace default**: Changed from `True` → `False`
2. **Exceptions**: Some `ValueError` → Custom exceptions (programmatically catchable)
3. **Plotting**: Now returns `Figure` objects (was `None`)

Migration:
```python
# OLD (implicit inplace=True):
preprocessor.handle_missing_values(strategy='mean')

# NEW (explicit):
preprocessor.handle_missing_values(strategy='mean', inplace=True)
```

---

## Common Pitfalls

1. **Don't forget to update `self.df`** when `inplace=True`
2. **Check for zero std/variance** before division
3. **Validate numeric columns** before math operations
4. **Use custom exceptions** for better error handling
5. **Close plot figures** to prevent memory leaks: `plt.close(fig)`

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

---

## Git Configuration

- **Remote**: origin → https://github.com/bluelion1999/feature_engineering_tk
- **Default Branch**: master (not main)
- **Initial Commit**: c146f94

---

**For Claude Code**: This document provides essential context for maintaining consistency. When making changes, update relevant sections to reflect new patterns or decisions.
