# Feature Plan: Enhanced Stats Reporting

**Branch**: feature/enhanced-stats-reporting
**Status**: Planning
**Target Module**: data_analysis.py

## Overview

Develop an extensive statistical reporting system that provides comprehensive insights when a target column is specified. The system will automatically detect whether the task is classification or regression and provide task-specific statistical analysis.

## Proposed Features

### 1. Target Analysis Class

Create a new class `TargetAnalyzer` that extends analysis capabilities when a target column is provided.

```python
class TargetAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str, task: str = 'auto')
```

**Parameters**:
- `df`: The dataframe to analyze
- `target_column`: Name of the target variable
- `task`: 'classification', 'regression', or 'auto' (auto-detect based on target column)

### 2. Classification-Specific Statistics

When task is classification:

#### Class Distribution Analysis
- Class counts and percentages
- Class imbalance ratio
- Minority/majority class identification
- Recommendations for handling imbalance (if present)

#### Feature-Target Relationships
- Chi-square test for categorical features vs target
- Point-biserial correlation for numeric features vs binary target
- ANOVA F-statistic for numeric features vs multi-class target
- Cramér's V for categorical-categorical relationships

#### Class-wise Statistics
- Mean, median, std of numeric features per class
- Distribution of categorical features per class
- Class separability metrics

#### Visualization
- Class distribution bar chart
- Feature distributions by class (box plots, violin plots)
- Confusion matrix heatmap (if predictions provided)

### 3. Regression-Specific Statistics

When task is regression:

#### Target Distribution Analysis
- Mean, median, mode, std, min, max
- Skewness and kurtosis
- Normality tests (Shapiro-Wilk, Anderson-Darling)
- Distribution visualization (histogram, Q-Q plot)

#### Feature-Target Relationships
- Pearson/Spearman correlation with target
- Mutual information scores
- Linear relationship strength (R²)
- Non-linear relationship detection

#### Residual Analysis (if predictions provided)
- Residual plots
- Heteroscedasticity detection
- Residual normality tests

#### Visualization
- Target distribution histogram
- Scatter plots: top features vs target
- Correlation heatmap (features + target)
- Pairplot of top correlated features

### 4. Common Statistics (Both Tasks)

#### Data Quality Report
- Missing values in features vs target
- Outliers in relation to target
- Data leakage detection (features perfectly correlated with target)

#### Statistical Tests
- Feature independence tests
- Multicollinearity detection (VIF - Variance Inflation Factor)
- Feature importance ranking

#### Recommendations
- Features to investigate further
- Potential data quality issues
- Suggested preprocessing steps
- Feature engineering opportunities

### 5. Report Generation

#### Summary Report
```python
def generate_report(self, format: str = 'text') -> Union[str, Dict, pd.DataFrame]:
    """Generate comprehensive statistical report"""
```

**Output Formats**:
- `'text'`: Formatted text report to console
- `'dict'`: Structured dictionary
- `'dataframe'`: Multiple DataFrames in a dict

#### Export Options
```python
def export_report(self, filepath: str, format: str = 'html'):
    """Export report to file"""
```

**Formats**:
- HTML report with interactive plots
- PDF report (if possible)
- Markdown report
- JSON for programmatic access

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `TargetAnalyzer` class in data_analysis.py
2. Implement task auto-detection logic
3. Add basic target distribution analysis

### Phase 2: Classification Features
1. Implement class distribution analysis
2. Add statistical tests for classification
3. Create class-wise feature statistics
4. Add classification visualizations

### Phase 3: Regression Features
1. Implement target distribution analysis
2. Add correlation and relationship metrics
3. Create regression-specific visualizations
4. Add residual analysis capabilities

### Phase 4: Common Features
1. Add data quality checks
2. Implement feature independence tests
3. Add multicollinearity detection
4. Create recommendation engine

### Phase 5: Reporting
1. Implement text report generation
2. Add structured output formats
3. Create HTML export with plots
4. Add markdown export

## Technical Considerations

### Dependencies
May need additional packages:
- `scipy.stats` for statistical tests (already in requirements)
- `statsmodels` for advanced statistics (new dependency)
- Consider: `plotly` for interactive HTML reports (optional)

### Performance
- Lazy computation of expensive statistics
- Caching of computed results
- Optional depth parameter (quick, standard, comprehensive)

### API Design
- Maintain consistency with existing DataAnalyzer class
- Support method chaining
- Clear parameter names and defaults

## Example Usage

```python
from mltoolkit import TargetAnalyzer

# Classification example
analyzer = TargetAnalyzer(df, target_column='is_fraud', task='classification')
analyzer.analyze_class_distribution()
analyzer.analyze_feature_relationships()
analyzer.plot_class_distributions()
report = analyzer.generate_report(format='text')
analyzer.export_report('fraud_analysis.html', format='html')

# Regression example
analyzer = TargetAnalyzer(df, target_column='price', task='regression')
analyzer.analyze_target_distribution()
analyzer.analyze_correlations()
analyzer.plot_top_features()
report = analyzer.generate_report(format='dict')
```

## Success Criteria

1. **Comprehensive**: Covers all major statistical analyses for both classification and regression
2. **Actionable**: Provides clear insights and recommendations
3. **User-friendly**: Easy to use with sensible defaults
4. **Fast**: Completes analysis on typical datasets (10K rows, 50 features) in <10 seconds
5. **Well-documented**: Clear docstrings and usage examples
6. **Tested**: Edge cases handled gracefully

## Future Enhancements (Post-MVP)

- Multi-output target support
- Time series specific analysis
- Automated feature engineering suggestions
- Integration with AutoML pipelines
- Comparison mode (compare multiple models/datasets)
- Interactive dashboard generation

## Notes

- Keep backward compatibility with existing DataAnalyzer class
- Consider adding to __init__.py exports when stable
- Update README.md with new features
- Update claude.md with implementation patterns
- Add comprehensive examples to documentation

---

**Created**: 2025-11-20
**Branch**: feature/enhanced-stats-reporting
**Status**: Ready for implementation
