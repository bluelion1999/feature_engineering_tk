# MLToolkit

A comprehensive Python toolkit for feature engineering and rudimentary data analysis to prepare dataframes for machine learning.

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/bluelion1999/feature_engineering_tk.git
```

### Install from PyPI (coming soon)

```bash
pip install feature-engineering-tk
```

### Install from source

```bash
git clone https://github.com/bluelion1999/feature_engineering_tk.git
cd feature_engineering_tk
pip install -e .
```

### Development Installation

For development with additional tools:

```bash
git clone https://github.com/bluelion1999/feature_engineering_tk.git
cd feature_engineering_tk
pip install -e ".[dev]"
```

## Modules

- **data_analysis.py**: Exploratory data analysis and visualization
- **feature_engineering.py**: Feature transformation and creation
- **preprocessing.py**: Data cleaning and preprocessing
- **feature_selection.py**: Feature selection methods

## Quick Start

```python
import pandas as pd
from feature_engineering_tk import DataAnalyzer, FeatureEngineer, DataPreprocessor, FeatureSelector, quick_analysis

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

### 2. Data Preprocessing

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

### 3. Feature Engineering

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

### 4. Feature Selection

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

### 5. Complete Pipeline Example

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

- `get_basic_info()`: Get basic dataframe information
- `get_missing_summary()`: Get summary of missing values
- `get_numeric_summary()`: Get statistics for numeric columns
- `get_categorical_summary()`: Get summary for categorical columns
- `detect_outliers_iqr()`: Detect outliers using IQR method
- `detect_outliers_zscore()`: Detect outliers using Z-score
- `get_correlation_matrix()`: Get correlation matrix
- `get_high_correlations()`: Find highly correlated feature pairs
- `get_cardinality_info()`: Get cardinality information
- `plot_missing_values()`: Visualize missing values
- `plot_correlation_heatmap()`: Plot correlation heatmap
- `plot_distributions()`: Plot feature distributions

### DataPreprocessor

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

## License

MIT
