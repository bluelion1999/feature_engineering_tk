# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[2.0.0]: https://github.com/bluelion1999/feature_engineering_tk/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/bluelion1999/feature_engineering_tk/releases/tag/v1.0.0
