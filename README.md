# Feature Engineering Toolkit v2.3.0

[![PyPI version](https://badge.fury.io/py/feature-engineering-tk.svg)](https://badge.fury.io/py/feature-engineering-tk)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Feature Engineering Toolkit** is a comprehensive Python library for feature engineering and advanced data analysis to prepare dataframes for machine learning. Provides intelligent automation for ML workflows including statistical analysis, feature engineering suggestions, and model recommendations.

## Features

- **Smart Data Analysis**: Automatic EDA with comprehensive statistics and visualizations
- **Column Type Detection**: Identify misclassified categorical columns and binning opportunities
- **Target-Aware Analysis**: Advanced statistical analysis that auto-detects classification vs regression tasks
- **Intelligent Recommendations**: Automated feature engineering suggestions based on data characteristics
- **Model Recommendations**: ML algorithm suggestions tailored to your dataset
- **Complete Preprocessing**: Handle missing values, outliers, duplicates with 8+ strategies
- **String Preprocessing**: Clean and extract features from text columns
- **Data Validation**: Comprehensive quality checks and infinite value detection
- **Method Chaining**: Fluent API for chaining preprocessing operations
- **Operation Tracking**: Automatic logging and export of preprocessing history
- **Feature Engineering**: 12+ transformation methods including encoding, scaling, binning, datetime extraction
- **Feature Selection**: 6+ selection methods with automatic pipeline
- **Report Generation**: Export comprehensive analysis reports in HTML, Markdown, or JSON

## Installation

```bash
pip install feature-engineering-tk
```

**Requirements:** Python 3.8+

## What's New in v2.3.0

**Architecture & Code Quality Improvements:**

### 🏗️ Refactored Architecture
Major internal refactoring for improved maintainability and performance:

- **New Base Class**: All classes now inherit from `FeatureEngineeringBase` for consistent initialization and shared functionality
- **Shared Utilities**: Centralized validation and column selection functions in `utils.py` module
- **Code Reduction**: Eliminated ~300 lines of redundant code (6.5% of codebase)
- **Better Organization**: Clear separation between shared infrastructure (`base.py`, `utils.py`) and domain-specific modules

### ✨ Benefits
- **Single Source of Truth**: All validation logic centralized for consistency
- **Significantly Faster**: 7x performance improvement for statistical analysis, 45% faster outlier detection
- **Better Maintainability**: Changes to common operations only need to be made once
- **100% Backward Compatible**: All existing code continues to work without modification
- **All 182 tests passing**: Comprehensive test coverage ensures reliability

### 🔧 Internal Improvements
- Consolidated outlier detection logic (DataPreprocessor now uses DataAnalyzer's detection methods)
- Eliminated duplicate `get_dataframe()` implementations
- Streamlined DataFrame initialization across all classes
- Consistent error handling and logging patterns

### ⚡ Performance Optimizations
- **Class-wise statistics 7x faster**: Optimized from 969ms to 138ms using groupby operations
- **Outlier detection 45% faster**: Improved from 221ms to 120ms with better index handling
- **Pre-computed aggregations**: Mean and median calculations optimized for large datasets
- Eliminated N+1 query patterns in statistical analysis methods

**Note**: This release focuses on internal improvements and performance optimizations. The public API remains unchanged, so all your existing code will continue to work exactly as before.

---

## What's New in v2.2.0

**DataAnalyzer Enhancements - Column Type Detection & Binning Suggestions:**

### 🔍 Column Type Detection
Identify numeric columns that should actually be categorical:

```python
from feature_engineering_tk import DataAnalyzer

analyzer = DataAnalyzer(df)

# Detect misclassified categorical columns
misclassified = analyzer.detect_misclassified_categorical(max_unique=10)
print(misclassified)
# Returns DataFrame with columns:
#   - column: column name
#   - unique_count: number of unique values
#   - unique_ratio: ratio of unique values to total rows
#   - dtype: current data type
#   - suggestion: why it should be categorical

# Automatically detects:
# - Binary/flag columns (exactly 2 unique values)
# - Low cardinality numeric columns (≤10 unique values by default)
# - Columns with very low unique ratios (many repeated values)
# - Integer columns with moderate cardinality (≤20 values)
```

### 📊 Binning Suggestions
Get intelligent binning recommendations based on distribution characteristics:

```python
# Get binning suggestions
binning_suggestions = analyzer.suggest_binning(min_unique=20)
print(binning_suggestions)
# Returns DataFrame with columns:
#   - column: column name
#   - current_unique: number of unique values
#   - suggested_bins: recommended number of bins
#   - binning_strategy: 'quantile' or 'uniform'
#   - reason: explanation for the suggestion

# Strategies:
# - Quantile binning for skewed distributions (abs(skewness) > 1.0)
# - Uniform binning for relatively uniform distributions
# - Handles outlier-heavy columns appropriately
```

### 📈 Enhanced quick_analysis()
The `quick_analysis()` function now includes two new sections:
- **MISCLASSIFIED CATEGORICAL COLUMNS**: Identifies data type issues
- **BINNING SUGGESTIONS**: Recommends binning strategies for continuous features

```python
from feature_engineering_tk import quick_analysis

quick_analysis(df)
# Now shows additional insights for better EDA
```

---

**DataPreprocessor Enhancements - Major Quality-of-Life Improvements:**

### 🎯 Method Chaining Support
Chain preprocessing operations for cleaner, more readable code:

```python
preprocessor = DataPreprocessor(df)
preprocessor\
    .handle_missing_values(strategy='mean', inplace=True)\
    .remove_duplicates(inplace=True)\
    .clean_string_columns(['name'], operations=['strip', 'lower'], inplace=True)\
    .drop_columns(['id'], inplace=True)
```

### 📊 Operation History Tracking
Automatically track all preprocessing operations for full reproducibility:

```python
# Perform operations (automatically logged when inplace=True)
preprocessor.handle_missing_values(strategy='mean', inplace=True)
preprocessor.remove_duplicates(inplace=True)

# Get formatted summary
summary = preprocessor.get_preprocessing_summary()
print(summary)
# Output:
# ================================================================================
# PREPROCESSING SUMMARY
# ================================================================================
# 1. HANDLE_MISSING_VALUES
#    Timestamp: 2025-11-30T14:23:45.123456
#    Shape: (1000, 10) → (1000, 10)
#    Parameters: strategy='mean', columns=['age', 'income']
# 2. REMOVE_DUPLICATES
#    Shape: (1000, 10) → (987, 10)
#    Details: rows_removed=13
# ================================================================================

# Export preprocessing history to file
preprocessor.export_summary('preprocessing_report.md', format='markdown')
preprocessor.export_summary('preprocessing_report.json', format='json')
```

### 🧹 String Preprocessing
New methods for cleaning and extracting features from text columns:

```python
# Clean string columns
preprocessor.clean_string_columns(
    columns=['name', 'city'],
    operations=['strip', 'lower', 'remove_punctuation'],
    inplace=True
)

# Standardize whitespace variants
preprocessor.handle_whitespace_variants(['category'], inplace=True)

# Extract string length features
preprocessor.extract_string_length(['description'], suffix='_len', inplace=True)
```

### ✅ Data Validation
Proactive data quality checks:

```python
# Comprehensive quality report
quality_report = preprocessor.validate_data_quality()
# Returns: {
#   'missing_values': {'age': 25, 'income': 10},
#   'constant_columns': ['id'],
#   'infinite_values': {'score': 3},
#   'duplicate_count': 5
# }

# Detect infinite values
infinite_vals = preprocessor.detect_infinite_values()

# Create missing value indicators
preprocessor.create_missing_indicators(['age', 'income'], inplace=True)
# Creates: age_was_missing, income_was_missing columns
```

### 🛡️ Enhanced Error Handling
- Better parameter validation across all methods
- Warnings for potentially destructive operations (e.g., removing >30% of data)
- Improved logging throughout

**Test Coverage:** Added 51 comprehensive tests for v2.2.0 (now 182 total tests across the library)
- **DataAnalyzer**: 9 tests for column type detection and binning suggestions
- **DataPreprocessor**: 42 tests for string preprocessing, data validation, error handling, method chaining, and operation history

## What's New in v2.1.1

**Bug Fixes & Code Quality:**
- Fixed version mismatch across configuration files
- Added missing `statsmodels>=0.14.0` dependency
- Removed unused imports and improved performance
- Enhanced type hints and documentation for `FeatureSelector`
- Fixed configuration file issues (.gitignore, MANIFEST.in)

## What's New in v2.1.0

**TargetAnalyzer** - A powerful new class for comprehensive target-aware statistical analysis:

- **Auto Task Detection**: Automatically detects classification vs regression tasks
- **Statistical Analysis**: Chi-square tests, ANOVA, correlations, mutual information
- **Data Quality Checks**: Missing values, multicollinearity (VIF), potential data leakage detection
- **Feature Engineering Suggestions**: Intelligent recommendations for transformations based on skewness, cardinality, and relationships
- **Model Recommendations**: ML algorithm suggestions based on dataset size, imbalance, dimensionality, and other characteristics
- **Comprehensive Reports**: Export analysis in HTML (with CSS styling), Markdown, or JSON formats

## Breaking Changes (v2.0.0)

**Version 2.0.0 introduces important breaking changes. Please review carefully before upgrading.**

### Inplace Parameter Default Changed

The `inplace` parameter default has changed from `True` to `False` for all methods in `DataPreprocessor` and `FeatureEngineer`. This aligns with pandas conventions and prevents accidental data mutations.

**Before (v1.x):**
```python
preprocessor = DataPreprocessor(df)
preprocessor.handle_missing_values(strategy='mean')  # Modified internal df by default
cleaned_df = preprocessor.get_dataframe()
```

**After (v2.0.0):**
```python
preprocessor = DataPreprocessor(df)

# Option 1: Explicitly use inplace=True (old behavior)
preprocessor.handle_missing_values(strategy='mean', inplace=True)
cleaned_df = preprocessor.get_dataframe()

# Option 2: Capture returned DataFrame (recommended)
cleaned_df = preprocessor.handle_missing_values(strategy='mean', inplace=False)
```

**Migration Guide:**

If you were relying on the implicit `inplace=True` behavior, you have two options:

1. **Add `inplace=True` to all method calls** (quick fix):
   ```python
   preprocessor.handle_missing_values(strategy='mean', inplace=True)
   preprocessor.remove_duplicates(inplace=True)
   ```

2. **Refactor to use returned DataFrames** (recommended, more pandas-like):
   ```python
   df = preprocessor.handle_missing_values(strategy='mean')
   df = preprocessor.remove_duplicates()
   ```

**Affected Classes:**
- `DataPreprocessor` - All transformation methods
- `FeatureEngineer` - All encoding, scaling, and feature creation methods

**Not Affected:**
- `DataAnalyzer` - Read-only, no inplace operations
- `FeatureSelector` - Uses different pattern with `apply_selection()`

See [CHANGELOG.md](CHANGELOG.md) for full list of changes.

## Modules

- **data_analysis.py**: Exploratory data analysis and visualization
- **feature_engineering.py**: Feature transformation and creation
- **preprocessing.py**: Data cleaning and preprocessing
- **feature_selection.py**: Feature selection methods

## Quick Start

```python
import pandas as pd
from feature_engineering_tk import DataAnalyzer, TargetAnalyzer, FeatureEngineer, DataPreprocessor, FeatureSelector, quick_analysis

# Load your data
df = pd.read_csv('your_data.csv')

# Quick analysis
quick_analysis(df)
```

## Usage Examples

### 1. Data Analysis

```python
from feature_engineering_tk import DataAnalyzer

# Initialize analyzer
analyzer = DataAnalyzer(df)

# Get basic information
info = analyzer.get_basic_info()
print(f"Shape: {info['shape']}")
print(f"Memory: {info['memory_usage_mb']:.2f} MB")

# Check missing values
missing = analyzer.get_missing_summary()
print(missing)

# Get numeric summary statistics
numeric_stats = analyzer.get_numeric_summary()
print(numeric_stats)

# Get categorical summary
cat_stats = analyzer.get_categorical_summary()
print(cat_stats)

# Find high correlations
high_corr = analyzer.get_high_correlations(threshold=0.7)
print(high_corr)

# Detect outliers using IQR method
outliers_iqr = analyzer.detect_outliers_iqr(columns=['age', 'salary'], multiplier=1.5)

# Detect outliers using Z-score method
outliers_zscore = analyzer.detect_outliers_zscore(columns=['age', 'salary'], threshold=3.0)

# Visualizations
analyzer.plot_missing_values()
analyzer.plot_correlation_heatmap()
analyzer.plot_distributions(columns=['age', 'salary', 'score'])
```

### 2. Target Analysis

```python
from feature_engineering_tk import TargetAnalyzer

# Initialize with target column (auto-detects classification vs regression)
analyzer = TargetAnalyzer(df, target_column='price', task='auto')

# Or explicitly specify task type
analyzer = TargetAnalyzer(df, target_column='category', task='classification')

# Get task information
task_info = analyzer.get_task_info()
print(f"Task type: {task_info['task']}")

# Classification Analysis
if analyzer.task == 'classification':
    # Class distribution and imbalance analysis
    dist = analyzer.analyze_class_distribution()
    imbalance_info = analyzer.get_class_imbalance_info()

    # Feature-target relationships (Chi-square, ANOVA)
    relationships = analyzer.analyze_feature_target_relationship()

    # Class-wise statistics
    class_stats = analyzer.analyze_class_wise_statistics()

    # Visualizations
    analyzer.plot_class_distribution(show=True)
    analyzer.plot_feature_by_class('age', plot_type='box', show=True)

# Regression Analysis
if analyzer.task == 'regression':
    # Target distribution with normality tests
    target_dist = analyzer.analyze_target_distribution(normality_test=True)

    # Feature correlations with target
    correlations = analyzer.analyze_feature_correlations(method='pearson')

    # Mutual information scores
    mi_scores = analyzer.analyze_mutual_information()

    # Visualizations
    analyzer.plot_target_distribution(show=True)
    analyzer.plot_feature_vs_target(max_features=6, show=True)

    # Residual analysis (requires predictions)
    residuals = analyzer.analyze_residuals(y_pred)
    analyzer.plot_residuals(y_pred, show=True)

# Common Analysis (both tasks)
# Data quality checks
quality = analyzer.analyze_data_quality()

# Multicollinearity detection (VIF)
vif_scores = analyzer.calculate_vif()

# Intelligent feature engineering suggestions
fe_suggestions = analyzer.suggest_feature_engineering()
for sugg in fe_suggestions:
    print(f"{sugg['priority'].upper()}: {sugg['feature']} - {sugg['suggestion']}")

# ML model recommendations
model_recs = analyzer.recommend_models()
for rec in model_recs:
    print(f"{rec['priority'].upper()}: {rec['model']}")
    print(f"  Why: {rec['reason']}")

# Actionable recommendations
recommendations = analyzer.generate_recommendations()

# Generate comprehensive report
report = analyzer.generate_full_report()

# Export report in multiple formats
analyzer.export_report('analysis.html', format='html')
analyzer.export_report('analysis.md', format='markdown')
analyzer.export_report('analysis.json', format='json')
```

### 3. Data Preprocessing

```python
from feature_engineering_tk import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(df)

# Handle missing values
preprocessor.handle_missing_values(strategy='mean', columns=['age', 'salary'])
preprocessor.handle_missing_values(strategy='mode', columns=['category'])
preprocessor.handle_missing_values(strategy='median', columns=['score'])

# Remove duplicates
preprocessor.remove_duplicates()

# Handle outliers
preprocessor.handle_outliers(
    columns=['salary', 'age'],
    method='iqr',
    action='cap',
    multiplier=1.5
)

# Convert data types
preprocessor.convert_dtypes({
    'date': 'datetime',
    'category': 'category',
    'price': 'float64'
})

# Clip values to range
preprocessor.clip_values('age', lower=0, upper=120)

# Remove constant columns
preprocessor.remove_constant_columns()

# Remove high cardinality columns
preprocessor.remove_high_cardinality_columns(threshold=0.95)

# Filter rows based on condition
preprocessor.filter_rows(lambda df: df['age'] > 18)

# Drop columns
preprocessor.drop_columns(['id', 'temp_column'])

# Rename columns
preprocessor.rename_columns({'old_name': 'new_name'})

# Apply custom function
preprocessor.apply_custom_function('text', lambda x: x.lower(), new_column='text_lower')

# Get cleaned dataframe
cleaned_df = preprocessor.get_dataframe()
```

### 4. Feature Engineering

```python
from feature_engineering_tk import FeatureEngineer

# Initialize feature engineer
engineer = FeatureEngineer(df)

# Label encoding
engineer.encode_categorical_label(columns=['gender', 'city'])

# One-hot encoding
engineer.encode_categorical_onehot(
    columns=['country', 'department'],
    drop_first=True,
    prefix={'country': 'cnt', 'department': 'dept'}
)

# Ordinal encoding
engineer.encode_categorical_ordinal(
    column='education',
    categories=['High School', 'Bachelor', 'Master', 'PhD']
)

# Scale features
engineer.scale_features(columns=['age', 'salary'], method='standard')
engineer.scale_features(columns=['price', 'quantity'], method='minmax')
engineer.scale_features(columns=['income'], method='robust')

# Create polynomial features
engineer.create_polynomial_features(
    columns=['feature1', 'feature2'],
    degree=2,
    interaction_only=False
)

# Create binning
engineer.create_binning(
    column='age',
    bins=5,
    strategy='quantile',
    labels=['Very Young', 'Young', 'Middle', 'Senior', 'Very Senior']
)

engineer.create_binning(
    column='salary',
    bins=[0, 30000, 60000, 100000, 200000],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Log transformation
engineer.create_log_transform(columns=['salary', 'revenue'])

# Square root transformation
engineer.create_sqrt_transform(columns=['area', 'population'])

# Extract datetime features
engineer.create_datetime_features(
    column='date',
    features=['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']
)

# Create aggregations
engineer.create_aggregations(
    group_by='city',
    agg_column='salary',
    agg_funcs=['mean', 'median', 'std']
)

engineer.create_aggregations(
    group_by=['department', 'level'],
    agg_column='performance_score',
    agg_funcs=['mean', 'max', 'min']
)

# Create ratio features
engineer.create_ratio_features(
    numerator='profit',
    denominator='revenue',
    name='profit_margin'
)

# Create flag features
engineer.create_flag_features(
    column='age',
    condition=lambda x: x >= 65,
    flag_name='is_senior'
)

engineer.create_flag_features(
    column='status',
    condition='active',
    flag_name='is_active'
)

# Get engineered dataframe
engineered_df = engineer.get_dataframe()
```

### 5. Feature Selection

```python
from feature_engineering_tk import FeatureSelector, select_features_auto

# Initialize feature selector
selector = FeatureSelector(df, target_column='target')

# Select by variance
selected = selector.select_by_variance(threshold=0.01)
print(f"Features with variance > 0.01: {selected}")

# Remove highly correlated features
selected = selector.select_by_correlation(threshold=0.8, method='pearson')
print(f"Features after correlation filter: {selected}")

# Select top k features correlated with target
selected = selector.select_by_target_correlation(k=10, method='pearson')
print(f"Top 10 features correlated with target: {selected}")

# Statistical test selection
selected = selector.select_by_statistical_test(
    k=15,
    task='classification',
    score_func='f_classif'
)
print(f"Top 15 features by statistical test: {selected}")

# Feature importance using Random Forest
selected = selector.select_by_importance(
    k=10,
    task='classification',
    n_estimators=100,
    random_state=42
)
print(f"Top 10 features by importance: {selected}")

# Select by missing values threshold
selected = selector.select_by_missing_values(threshold=0.3)
print(f"Features with < 30% missing: {selected}")

# Get feature importance dataframe
importance_df = selector.get_feature_importance_df()
print(importance_df)

# Apply selection to get new dataframe
selected_df = selector.apply_selection(keep_target=True)

# Automatic feature selection pipeline
auto_selected_df = select_features_auto(
    df,
    target_column='target',
    task='classification',
    max_features=20,
    variance_threshold=0.01,
    correlation_threshold=0.9
)
```

### 6. Complete Pipeline Example

```python
import pandas as pd
from feature_engineering_tk import DataAnalyzer, DataPreprocessor, FeatureEngineer, FeatureSelector

# Load data
df = pd.read_csv('data.csv')

# Step 1: Analyze
print("Analyzing data...")
analyzer = DataAnalyzer(df)
quick_analysis(df)

# Step 2: Preprocess
print("\nPreprocessing data...")
preprocessor = DataPreprocessor(df)
preprocessor.handle_missing_values(strategy='mean', columns=['numeric_col'])
preprocessor.handle_missing_values(strategy='mode', columns=['categorical_col'])
preprocessor.remove_duplicates()
preprocessor.handle_outliers(columns=['salary'], method='iqr', action='cap')
df_clean = preprocessor.get_dataframe()

# Step 3: Feature Engineering
print("\nEngineering features...")
engineer = FeatureEngineer(df_clean)
engineer.encode_categorical_onehot(columns=['category'], drop_first=True)
engineer.scale_features(columns=['age', 'salary'], method='standard')
engineer.create_datetime_features(column='date', features=['year', 'month', 'dayofweek'])
engineer.create_ratio_features('profit', 'revenue', 'profit_margin')
df_engineered = engineer.get_dataframe()

# Step 4: Feature Selection
print("\nSelecting features...")
selector = FeatureSelector(df_engineered, target_column='target')
selected_features = selector.select_by_importance(k=15, task='classification')
df_final = selector.apply_selection(keep_target=True)

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Selected features: {selected_features}")

# Ready for ML!
X = df_final.drop('target', axis=1)
y = df_final['target']
```

## API Reference

### DataAnalyzer

General-purpose exploratory data analysis (no target column required).

**Core Methods:**
- `get_basic_info()`: Get basic dataframe information (shape, dtypes, memory)
- `get_missing_summary()`: Get summary of missing values
- `get_numeric_summary()`: Get statistics for numeric columns
- `get_categorical_summary()`: Get summary for categorical columns
- `detect_outliers_iqr()`: Detect outliers using IQR method
- `detect_outliers_zscore()`: Detect outliers using Z-score
- `get_correlation_matrix()`: Get correlation matrix
- `get_high_correlations()`: Find highly correlated feature pairs
- `calculate_vif()`: Calculate Variance Inflation Factor for multicollinearity detection
- `get_cardinality_info()`: Get cardinality information for categorical features
- `detect_misclassified_categorical()`: Identify numeric columns that should be categorical
- `suggest_binning()`: Get intelligent binning recommendations based on distributions

**Visualization Methods:**
- `plot_missing_values()`: Visualize missing values heatmap
- `plot_correlation_heatmap()`: Plot correlation heatmap
- `plot_distributions()`: Plot feature distributions (histograms/KDE)

### TargetAnalyzer

Advanced target-aware analysis for ML tasks (requires target column).

**Initialization:**
- `TargetAnalyzer(df, target_column, task='auto')`: Auto-detects classification vs regression

**Task Information:**
- `get_task_info()`: Get detected task type and target column information

**Classification Methods:**
- `analyze_class_distribution()`: Class counts, percentages, imbalance ratios
- `get_class_imbalance_info()`: Detailed imbalance analysis with severity levels
- `analyze_feature_target_relationship()`: Chi-square and ANOVA tests
- `analyze_class_wise_statistics()`: Feature statistics per class
- `plot_class_distribution()`: Visualize class distribution
- `plot_feature_by_class()`: Box/violin/histogram plots by class

**Regression Methods:**
- `analyze_target_distribution()`: Target statistics with normality tests (Shapiro-Wilk, Anderson-Darling)
- `analyze_feature_correlations()`: Pearson/Spearman correlations with target
- `analyze_residuals()`: Residual analysis (MAE, RMSE, R², normality)
- `plot_target_distribution()`: Target histogram and Q-Q plot
- `plot_feature_vs_target()`: Scatter plots with regression lines
- `plot_residuals()`: Residual plots (residuals vs predicted, Q-Q plot)

**Common Methods (Both Tasks):**
- `analyze_mutual_information()`: Feature importance via mutual information
- `analyze_data_quality()`: Missing values, constant features, leakage detection
- `calculate_vif()`: Multicollinearity detection (auto-excludes target)
- `suggest_feature_engineering()`: Intelligent feature transformation recommendations
- `recommend_models()`: ML algorithm recommendations based on data characteristics
- `generate_recommendations()`: Actionable recommendations with priority levels
- `generate_full_report()`: Comprehensive analysis dictionary
- `export_report()`: Export to HTML/Markdown/JSON formats

### DataPreprocessor

**Data Cleaning:**
- `handle_missing_values()`: Handle missing values with various strategies
- `remove_duplicates()`: Remove duplicate rows
- `handle_outliers()`: Handle outliers
- `convert_dtypes()`: Convert column data types
- `clip_values()`: Clip values to range
- `remove_constant_columns()`: Remove constant columns
- `remove_high_cardinality_columns()`: Remove high cardinality columns
- `filter_rows()`: Filter rows by condition
- `drop_columns()`: Drop specified columns
- `rename_columns()`: Rename columns
- `apply_custom_function()`: Apply custom transformation

**String Preprocessing:**
- `clean_string_columns()`: Clean string columns with 7 operations (strip, lower, upper, title, remove_punctuation, remove_digits, remove_extra_spaces)
- `handle_whitespace_variants()`: Standardize whitespace variants in categorical columns
- `extract_string_length()`: Create length features from string columns

**Data Validation:**
- `validate_data_quality()`: Comprehensive quality report (missing values, constant columns, infinite values, duplicates)
- `detect_infinite_values()`: Detect np.inf/-np.inf in numeric columns
- `create_missing_indicators()`: Create binary indicator columns for missing values

**Operation Tracking:**
- `get_preprocessing_summary()`: Get formatted text summary of all preprocessing operations
- `export_summary()`: Export preprocessing history to text/markdown/JSON formats

### FeatureEngineer

- `encode_categorical_label()`: Label encoding
- `encode_categorical_onehot()`: One-hot encoding
- `encode_categorical_ordinal()`: Ordinal encoding
- `scale_features()`: Scale features (standard, minmax, robust)
- `create_polynomial_features()`: Create polynomial features
- `create_binning()`: Bin continuous features
- `create_log_transform()`: Apply log transformation
- `create_sqrt_transform()`: Apply square root transformation
- `create_datetime_features()`: Extract datetime features
- `create_aggregations()`: Create aggregation features
- `create_ratio_features()`: Create ratio features
- `create_flag_features()`: Create binary flag features

### FeatureSelector

- `select_by_variance()`: Select by variance threshold
- `select_by_correlation()`: Remove highly correlated features
- `select_by_target_correlation()`: Select by correlation with target
- `select_by_statistical_test()`: Select using statistical tests
- `select_by_importance()`: Select by feature importance
- `select_by_missing_values()`: Select by missing value threshold
- `get_feature_importance_df()`: Get feature scores dataframe
- `apply_selection()`: Apply selection to dataframe

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- **Documentation**: See usage examples above
- **Issues**: [GitHub Issues](https://github.com/bluelion1999/feature_engineering_tk/issues)
- **PyPI**: [feature-engineering-tk](https://pypi.org/project/feature-engineering-tk/)

## Links

- **GitHub Repository**: https://github.com/bluelion1999/feature_engineering_tk
- **PyPI Package**: https://pypi.org/project/feature-engineering-tk/
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## License

MIT License - see [LICENSE](LICENSE) file for details
