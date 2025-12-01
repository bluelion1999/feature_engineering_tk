# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-11-30

### Added

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

- Added 42 comprehensive tests for new features (now 173 total tests)
  - 7 tests for string preprocessing
  - 6 tests for data validation
  - 6 tests for enhanced error handling
  - 6 tests for method chaining
  - 17 tests for operation history tracking

All 173 tests pass successfully.

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
  - Developer documentation in `claude.md`
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

- Initial release of MLToolkit
- `DataAnalyzer` - Exploratory data analysis and visualization
- `DataPreprocessor` - Data cleaning and preprocessing
- `FeatureEngineer` - Feature transformation and creation
- `FeatureSelector` - Feature selection methods
- Basic documentation and examples

[2.1.1]: https://github.com/bluelion1999/feature_engineering_tk/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/bluelion1999/feature_engineering_tk/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/bluelion1999/feature_engineering_tk/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/bluelion1999/feature_engineering_tk/releases/tag/v1.0.0
