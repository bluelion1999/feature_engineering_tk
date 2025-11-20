# MLToolkit - Claude Code Development Documentation

> **Purpose**: This document provides detailed context for Claude Code to understand, maintain, and extend the MLToolkit repository. It includes implementation patterns, architectural decisions, and specific code structures used throughout the project.

## Project Overview

This is **MLToolkit**, a comprehensive Python library for feature engineering and rudimentary data analysis to prepare dataframes for machine learning. The toolkit was created to provide data scientists with modular, reusable tools for common ML preprocessing tasks.

**Repository**: https://github.com/bluelion1999/feature_engineering_tk
**Default Branch**: master
**Python Version**: 3.8+

## Project Structure

```
mltoolkit/
├── __init__.py                 # Package initialization, exposes main classes
├── data_analysis.py           # Exploratory data analysis and visualization
├── feature_engineering.py     # Feature transformation and creation
├── preprocessing.py           # Data cleaning and preprocessing
├── feature_selection.py       # Feature selection methods
├── requirements.txt           # Project dependencies
├── README.md                  # User-facing documentation
└── claude.md                  # This file - development documentation
```

## Module Descriptions

### 1. data_analysis.py

**Purpose**: Exploratory data analysis and statistical summaries

**Main Class**: `DataAnalyzer`

**Key Features**:
- Basic dataframe information (shape, dtypes, memory usage, duplicates)
- Missing value analysis with counts and percentages
- Numeric and categorical column summaries
- Outlier detection using IQR and Z-score methods
- Correlation analysis (matrix, high correlations)
- Cardinality information for all columns
- Visualization tools (missing values, heatmaps, distributions)

**Helper Function**: `quick_analysis(df)` - Performs comprehensive analysis and prints results

**Design Pattern**: Takes a dataframe in constructor, provides analysis methods that return DataFrames or dictionaries

### 2. feature_engineering.py

**Purpose**: Transform and create new features from existing data

**Main Class**: `FeatureEngineer`

**Key Features**:
- **Encoding**: Label, one-hot, and ordinal encoding for categorical variables
- **Scaling**: Standard, MinMax, and Robust scaling methods
- **Polynomial Features**: Create polynomial and interaction features
- **Binning**: Discretize continuous variables using quantile or equal-width strategies
- **Transformations**: Log and square root transformations
- **Datetime Features**: Extract year, month, day, dayofweek, quarter, weekend flags, etc.
- **Aggregations**: Create grouped aggregation features (mean, sum, std, min, max)
- **Ratio Features**: Create ratio features from two numeric columns
- **Flag Features**: Create binary indicators based on conditions

**Design Pattern**:
- Stores dataframe internally with copy
- All methods have `inplace` parameter (default True)
- Stores fitted encoders/scalers in dictionaries for potential reuse
- Returns dataframe for method chaining

### 3. preprocessing.py

**Purpose**: Clean and prepare data for analysis

**Main Class**: `DataPreprocessor`

**Key Features**:
- **Missing Values**: Multiple strategies (drop, fill, mean, median, mode, forward/backward fill, interpolate)
- **Duplicates**: Remove duplicate rows with flexible subset and keep options
- **Outliers**: Handle using IQR or Z-score methods (remove, cap, replace)
- **Data Types**: Convert column types including datetime and category
- **Clipping**: Clip values to specified ranges
- **Column Operations**: Drop constant/high-cardinality columns, rename, reorder
- **Filtering**: Filter rows by conditions
- **Custom Functions**: Apply custom transformations to columns
- **Sampling**: Sample data with optional random state

**Design Pattern**: Similar to FeatureEngineer with inplace operations and method chaining

### 4. feature_selection.py

**Purpose**: Select most relevant features for modeling

**Main Class**: `FeatureSelector`

**Key Features**:
- **Variance-based**: Remove low-variance features
- **Correlation-based**: Remove highly correlated features
- **Target Correlation**: Select features most correlated with target
- **Statistical Tests**: F-test, mutual information, chi-square
- **Feature Importance**: Tree-based importance using Random Forest
- **Missing Value Filter**: Remove features with too many missing values
- **Auto Selection**: `select_features_auto()` function applies multi-step pipeline

**Helper Function**: `select_features_auto(df, target_column, task, max_features, variance_threshold, correlation_threshold)`

**Design Pattern**:
- Requires target_column for supervised methods
- Stores selected features and scores internally
- `apply_selection()` method returns filtered dataframe
- `get_feature_importance_df()` returns scores as DataFrame

## Detailed Implementation Patterns

### Class Constructor Pattern

All main classes follow this pattern:

```python
class ClassName:
    def __init__(self, df: pd.DataFrame, optional_param: Optional[str] = None):
        self.df = df.copy()  # Always copy to avoid modifying original
        self.optional_param = optional_param
        # Additional state storage (encoders, scalers, selected features, etc.)
        self.state_dict = {}
```

**Key Points**:
- Always use `df.copy()` to prevent unintended modifications
- Store additional state in instance variables (e.g., `self.encoders`, `self.scalers`, `self.selected_features`)
- Use type hints for parameters

### Method Signature Pattern

Most transformation methods follow this structure:

```python
def method_name(self,
                columns: Union[str, List[str]],
                specific_params: Any,
                inplace: bool = True) -> pd.DataFrame:
    """Method docstring."""
    # 1. Handle inplace logic
    df_result = self.df if inplace else self.df.copy()

    # 2. Validate inputs
    if column not in df_result.columns:
        print(f"Warning: Column '{column}' not found in dataframe.")
        return df_result

    # 3. Perform transformation
    # ... actual logic ...

    # 4. Update internal state if inplace
    if inplace:
        self.df = df_result
        return self.df
    return df_result
```

**Key Points**:
- `inplace` parameter defaults to `True`
- Use `print()` for warnings, not exceptions (defensive programming)
- Always return a DataFrame
- Update `self.df` only when `inplace=True`

### Column Validation Pattern

Used consistently across all modules:

```python
# For single column
if column not in df_result.columns:
    print(f"Warning: Column '{column}' not found in dataframe.")
    return df_result

# For multiple columns
valid_cols = [col for col in columns if col in df_result.columns]
if not valid_cols:
    print(f"Warning: None of the specified columns found in dataframe.")
    return df_result
```

### Error Handling Pattern

- **Validation errors**: Use `raise ValueError()` for invalid parameters
- **Missing columns**: Print warning and return unchanged dataframe
- **Computation errors**: Use try-except with helpful error messages

```python
try:
    # risky operation
    result = some_operation(data)
except Exception as e:
    print(f"Error during operation: {e}")
    return df_result  # Return unchanged on error
```

### Sklearn Integration Pattern

When using sklearn transformers:

```python
from sklearn.preprocessing import StandardScaler

# In method:
scaler = StandardScaler()
df_result[columns] = scaler.fit_transform(df_result[columns])
self.scalers[f"{method}_scaler"] = scaler  # Store for potential reuse
```

**Sklearn Classes Used**:
- `StandardScaler`, `MinMaxScaler`, `RobustScaler` (preprocessing)
- `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder` (preprocessing)
- `SelectKBest`, `VarianceThreshold` (feature_selection)
- `f_classif`, `f_regression`, `mutual_info_classif`, `mutual_info_regression`, `chi2` (feature_selection)
- `RandomForestClassifier`, `RandomForestRegressor` (ensemble)

### Visualization Pattern (DataAnalyzer only)

```python
def plot_something(self, figsize: tuple = (12, 6)):
    """Docstring."""
    # 1. Prepare data
    data = self.get_relevant_data()

    if data.empty:
        print("No data available to plot.")
        return

    # 2. Create plot
    fig, ax = plt.subplots(figsize=figsize)
    # plotting logic using seaborn or matplotlib

    # 3. Finalize
    plt.tight_layout()
    plt.show()
```

### Feature Creation Naming Conventions

When creating new features:
- Polynomial: `{column}_squared`, `{column}_cubed`
- Interactions: `{col1}_x_{col2}`
- Transformations: `{column}_log`, `{column}_sqrt`
- Binning: `{column}_binned`
- Datetime: `{column}_{feature}` (e.g., `date_year`, `date_month`)
- Aggregations: `{agg_column}_{groupby}_{func}` (e.g., `salary_department_mean`)
- Ratios: `{numerator}_to_{denominator}_ratio`
- Flags: `{column}_flag` or custom name

## Design Principles & Patterns

### 1. Consistent API Design
- All main classes accept a DataFrame in the constructor
- All transformation methods have an `inplace` parameter (default True)
- Methods return the dataframe for chaining operations
- `get_dataframe()` method available to retrieve current state

### 2. Defensive Programming
- Check if columns exist before operating on them
- Print warnings for missing columns instead of raising errors
- Handle edge cases (negative values in log transform, division by zero)
- Validate parameters and provide helpful error messages

### 3. Flexibility
- Optional parameters with sensible defaults
- Support for both single columns and lists of columns
- Multiple strategies/methods for common operations
- Custom function support where applicable

### 4. Scikit-learn Integration
- Uses sklearn for encoding, scaling, and feature selection
- Stores fitted transformers for potential future use
- Compatible with sklearn pipelines

## Module-Specific Implementation Details

### data_analysis.py

**Imports**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
```

**Class State**:
- `self.df`: Copied dataframe (never modifies original)

**Key Implementation Details**:
- `get_correlation_matrix()`: Uses `df.corr(method=method)` where method can be 'pearson', 'spearman', or 'kendall'
- Outlier detection uses Q1, Q3, IQR calculations: `lower_bound = Q1 - multiplier * IQR`
- `quick_analysis()` is a standalone function (not a class method) that prints formatted analysis
- Visualization methods don't return anything, just display plots using `plt.show()`

### feature_engineering.py

**Imports**:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, OrdinalEncoder
)
from typing import List, Optional, Dict, Union, Any
```

**Class State**:
- `self.df`: Internal dataframe
- `self.encoders`: Dict storing fitted encoders with keys like `"{column}_label"`, `"{column}_ordinal"`
- `self.scalers`: Dict storing fitted scalers with keys like `"{method}_scaler"`

**Key Implementation Details**:
- `encode_categorical_onehot()` uses `pd.get_dummies()` with `dtype=int` and concatenates to df
- Log transform handles negative values by adding offset: `offset = abs(min_val) + 1`
- Datetime extraction uses pandas dt accessor: `df[column].dt.year`, `df[column].dt.dayofweek`, etc.
- Aggregations use `groupby().transform()` to maintain original row count
- Ratio features add small epsilon to denominator: `df[numerator] / (df[denominator] + 1e-8)`

### preprocessing.py

**Imports**:
```python
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any
```

**Class State**:
- `self.df`: Internal dataframe only

**Key Implementation Details**:
- Missing value strategies map to pandas methods:
  - `'forward_fill'` → `fillna(method='ffill')`
  - `'backward_fill'` → `fillna(method='bfill')`
  - `'interpolate'` → `interpolate(method=method or 'linear')`
- Outlier handling with action='cap' directly modifies values using `.loc[]`
- Type conversion uses `pd.to_datetime()` for datetime and `astype()` for others
- After row filtering/outlier removal, always `reset_index(drop=True)`

### feature_selection.py

**Imports**:
```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    mutual_info_regression, chi2, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Optional, Union
```

**Class State**:
- `self.df`: Internal dataframe
- `self.target_column`: Target variable name (can be None)
- `self.selected_features`: List of selected feature names
- `self.feature_scores`: Dict mapping score type to scores dict

**Key Implementation Details**:
- Feature scores stored with keys like `'variance'`, `'correlation_with_target'`, `'statistical_test'`, `'importance'`
- Correlation-based selection uses upper triangle to avoid dropping both correlated features
- Random Forest importance: `n_jobs=-1` for parallel processing
- `select_features_auto()` is a standalone function that creates multiple FeatureSelector instances in sequence
- Always exclude target column from feature lists using list comprehension

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

**Note**: No other dependencies required. Keep it minimal.

### __init__.py

**Purpose**: Package initialization that exposes main classes

**Complete Contents**:
```python
"""
MLToolkit - A comprehensive Python toolkit for feature engineering and data analysis
"""

from .data_analysis import DataAnalyzer, quick_analysis
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .feature_selection import FeatureSelector, select_features_auto

__version__ = '1.0.0'

__all__ = [
    'DataAnalyzer',
    'quick_analysis',
    'FeatureEngineer',
    'DataPreprocessor',
    'FeatureSelector',
    'select_features_auto',
]
```

**Key Points**:
- Uses relative imports (`.data_analysis`, not `data_analysis`)
- Exposes both classes and helper functions
- Includes version number
- `__all__` list controls `from mltoolkit import *`

## Common Implementation Examples

### Example: Outlier Detection (IQR Method)

Used in both `DataAnalyzer.detect_outliers_iqr()` and `DataPreprocessor.handle_outliers()`:

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - multiplier * IQR
upper_bound = Q3 + multiplier * IQR

outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
```

### Example: Column Type Selection

Pattern for selecting numeric vs categorical columns:

```python
# Numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
```

### Example: Creating Feature Names

From `FeatureEngineer.create_aggregations()`:

```python
for func in agg_funcs:
    agg_name = f"{agg_column}_{'_'.join(group_by)}_{func}"
    agg_values = df.groupby(group_by)[agg_column].transform(func)
    df[agg_name] = agg_values
```

### Example: Handling String/List Column Parameters

Many methods accept both single strings and lists:

```python
if isinstance(group_by, str):
    group_by = [group_by]  # Convert to list for uniform handling
```

### Example: Return Pattern with Inplace

Consistent across all transformation methods:

```python
if inplace:
    self.df = df_result
    return self.df
return df_result
```

## Git Configuration

- **Repository**: https://github.com/bluelion1999/feature_engineering_tk
- **Default Branch**: master
- **Remote**: origin
- **Initial Commit**: c146f94 - "Initial commit: MLToolkit for feature engineering and data analysis"

## Usage Patterns

### Typical Workflow

```python
import pandas as pd
from mltoolkit import DataAnalyzer, DataPreprocessor, FeatureEngineer, FeatureSelector

# 1. Load and analyze
df = pd.read_csv('data.csv')
analyzer = DataAnalyzer(df)
quick_analysis(df)

# 2. Preprocess
preprocessor = DataPreprocessor(df)
preprocessor.handle_missing_values(strategy='mean', columns=['numeric_cols'])
preprocessor.remove_duplicates()
preprocessor.handle_outliers(columns=['salary'], method='iqr', action='cap')
df_clean = preprocessor.get_dataframe()

# 3. Engineer features
engineer = FeatureEngineer(df_clean)
engineer.encode_categorical_onehot(columns=['category'], drop_first=True)
engineer.scale_features(columns=['age', 'salary'], method='standard')
engineer.create_datetime_features(column='date')
df_engineered = engineer.get_dataframe()

# 4. Select features
selector = FeatureSelector(df_engineered, target_column='target')
selected_features = selector.select_by_importance(k=15, task='classification')
df_final = selector.apply_selection(keep_target=True)
```

## Future Development Guidelines

### When Adding New Features

1. **Maintain API Consistency**:
   - Use `inplace` parameter
   - Return the dataframe
   - Accept column names as strings or lists

2. **Handle Edge Cases**:
   - Check column existence
   - Handle missing values appropriately
   - Validate input parameters

3. **Document Thoroughly**:
   - Add docstrings to all methods
   - Update README.md with examples
   - Add to appropriate section in this file

4. **Test Edge Cases**:
   - Empty dataframes
   - Single column dataframes
   - All-null columns
   - Different data types

### Code Style

- Use type hints where beneficial
- Keep methods focused on single responsibility
- Prefer explicit over implicit
- Use meaningful variable names
- Add comments for complex logic only

### Common Extension Points

1. **New Encoding Methods** → Add to `FeatureEngineer`
2. **New Outlier Detection** → Add to `DataAnalyzer` and `DataPreprocessor`
3. **New Transformations** → Add to `FeatureEngineer`
4. **New Selection Methods** → Add to `FeatureSelector`
5. **New Visualizations** → Add to `DataAnalyzer`

## Known Limitations

1. **Memory**: All operations create copies of dataframes (can be memory-intensive for large datasets)
2. **Scaling**: Fitted scalers/encoders stored but not easily exportable for production use
3. **Validation**: No built-in train/test split handling
4. **Categorical Encoding**: High-cardinality categorical variables may create many columns with one-hot encoding
5. **Feature Selection**: Tree-based methods can be slow on very large datasets

## How to Reproduce This Project From Scratch

If you need to recreate this entire project, follow these steps:

### 1. Create requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

### 2. Create data_analysis.py
- Import: pandas, numpy, matplotlib.pyplot, seaborn, typing
- Create `DataAnalyzer` class with `__init__(self, df)`
- Add methods: get_basic_info, get_missing_summary, get_numeric_summary, get_categorical_summary
- Add methods: detect_outliers_iqr, detect_outliers_zscore, get_correlation_matrix, get_high_correlations, get_cardinality_info
- Add plotting methods: plot_missing_values, plot_correlation_heatmap, plot_distributions
- Create `quick_analysis(df)` standalone function

### 3. Create feature_engineering.py
- Import: pandas, numpy, sklearn.preprocessing classes, typing
- Create `FeatureEngineer` class with encoders and scalers dicts
- Add encoding methods: encode_categorical_label, encode_categorical_onehot, encode_categorical_ordinal
- Add scaling: scale_features (standard, minmax, robust)
- Add feature creation: create_polynomial_features, create_binning, create_log_transform, create_sqrt_transform
- Add datetime: create_datetime_features
- Add aggregation: create_aggregations, create_ratio_features, create_flag_features
- Add: get_dataframe() method

### 4. Create preprocessing.py
- Import: pandas, numpy, typing
- Create `DataPreprocessor` class
- Add methods: handle_missing_values (8+ strategies), remove_duplicates, handle_outliers
- Add methods: convert_dtypes, clip_values, remove_constant_columns, remove_high_cardinality_columns
- Add methods: filter_rows, drop_columns, rename_columns, reorder_columns
- Add methods: apply_custom_function, reset_index_clean, sample_data, get_dataframe

### 5. Create feature_selection.py
- Import: pandas, numpy, sklearn.feature_selection, sklearn.ensemble, typing
- Create `FeatureSelector` class with target_column, selected_features, feature_scores
- Add methods: select_by_variance, select_by_correlation, select_by_target_correlation
- Add methods: select_by_statistical_test, select_by_importance, select_by_missing_values
- Add methods: get_feature_importance_df, apply_selection, get_selected_features
- Create `select_features_auto()` standalone function (3-step pipeline)

### 6. Create __init__.py
- Add module docstring
- Import all main classes and helper functions using relative imports
- Set `__version__ = '1.0.0'`
- Define `__all__` list

### 7. Create README.md
- Add project description, installation, module overview
- Add usage examples for each module
- Add complete pipeline example
- Add API reference section

### 8. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: MLToolkit for feature engineering and data analysis

Added comprehensive toolkit with four main modules:
- data_analysis.py: Exploratory data analysis and visualization
- feature_engineering.py: Feature transformation and creation
- preprocessing.py: Data cleaning and preprocessing
- feature_selection.py: Feature selection methods

Includes complete documentation and usage examples.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"

git remote add origin https://github.com/bluelion1999/feature_engineering_tk
git push -u origin master
```

## Important Notes for Future Conversations

1. **Don't Create Unnecessary Files**: Only create files when explicitly needed. Prefer editing existing files.

2. **Git Commit Messages**: Use descriptive messages with the Claude Code attribution format (without Co-Authored-By since it's redundant)

3. **Default Branch**: This project uses **master** as the default branch, not main

4. **Inplace Operations**: Most methods default to `inplace=True` to modify the object's internal dataframe

5. **Windows Environment**: Project developed on Windows (note path separators, line endings)

6. **No Emojis**: Avoid using emojis in code unless explicitly requested by user

7. **Error Handling Philosophy**: Print warnings for missing columns, raise ValueError for invalid parameters, use try-except for risky operations

8. **Always Copy DataFrames**: Use `df.copy()` in constructors to avoid modifying originals

9. **Type Hints**: Use typing module for clear function signatures (List, Optional, Dict, Union, Any)

## Contact & Collaboration

**Repository Owner**: bluelion1999
**Created**: 2025-11-20
**Last Updated**: 2025-11-20

---

**For Claude Code Users**: This document serves as project context for maintaining consistency across conversation sessions. When making changes, update this file to reflect new decisions, patterns, or structure changes.
