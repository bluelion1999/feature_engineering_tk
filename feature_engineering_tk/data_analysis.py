import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .base import FeatureEngineeringBase
from .utils import (
    validate_columns,
    get_numeric_columns,
    get_feature_columns
)
from . import statistical_utils

# Configure logging
logger = logging.getLogger(__name__)


class DataAnalyzer(FeatureEngineeringBase):
    """
    Data analysis class for exploratory data analysis and visualization.

    Provides statistical summaries, outlier detection, correlation analysis,
    and visualization tools.
    """

    def get_basic_info(self) -> pd.DataFrame:
        """Get basic information about the dataframe."""
        return pd.DataFrame({
            'shape': [self.df.shape],
            'columns': [list(self.df.columns)],
            'dtypes': [self.df.dtypes.to_dict()],
            'memory_usage_mb': [self.df.memory_usage(deep=True).sum() / 1024**2],
            'duplicates': [self.df.duplicated().sum()]
        })

    def get_missing_summary(self) -> pd.DataFrame:
        """Get summary of missing values."""
        missing = pd.DataFrame({
            'column': self.df.columns,
            'missing_count': self.df.isnull().sum().values,
            'missing_percent': (self.df.isnull().sum() / len(self.df) * 100).values
        })
        missing = missing[missing['missing_count'] > 0].sort_values(
            'missing_percent', ascending=False
        )
        return missing.reset_index(drop=True)

    def get_numeric_summary(self, percentiles: Optional[List[float]] = None) -> pd.DataFrame:
        """Get summary statistics for numeric columns."""
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]

        numeric_cols = get_numeric_columns(self.df)
        if len(numeric_cols) == 0:
            return pd.DataFrame()

        return self.df[numeric_cols].describe(percentiles=percentiles)

    def get_categorical_summary(self, max_unique: int = 50) -> pd.DataFrame:
        """Get summary for categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) == 0:
            return pd.DataFrame()

        summary = []
        for col in cat_cols:
            unique_count = self.df[col].nunique()
            if unique_count <= max_unique:
                top_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else None
                # Check if value_counts is empty before accessing .iloc[0]
                value_counts = self.df[col].value_counts()
                summary.append({
                    'column': col,
                    'unique_count': unique_count,
                    'top_value': top_value,
                    'top_value_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_value_percent': (value_counts.iloc[0] / len(self.df) * 100) if len(value_counts) > 0 else 0
                })

        return pd.DataFrame(summary)

    def detect_outliers_iqr(self, columns: Optional[List[str]] = None, multiplier: float = 1.5) -> Dict[str, pd.Series]:
        """Detect outliers using IQR method."""
        if columns is None:
            columns = get_numeric_columns(self.df)
        else:
            columns = validate_columns(self.df, columns)

        outliers = {}
        for col in columns:

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            if outlier_mask.sum() > 0:
                outliers[col] = outlier_mask

        return outliers

    def detect_outliers_zscore(self, columns: Optional[List[str]] = None, threshold: float = 3.0) -> Dict[str, pd.Series]:
        """Detect outliers using Z-score method."""
        if columns is None:
            columns = get_numeric_columns(self.df)
        else:
            columns = validate_columns(self.df, columns)

        outliers = {}
        for col in columns:

            # Fixed: Add division by zero check
            col_std = self.df[col].std()
            if col_std == 0:
                logger.warning(f"Column '{col}' has zero standard deviation, skipping outlier detection")
                continue

            z_scores = np.abs((self.df[col] - self.df[col].mean()) / col_std)
            outlier_mask = z_scores > threshold

            if outlier_mask.sum() > 0:
                outliers[col] = outlier_mask

        return outliers

    def get_correlation_matrix(self, method: str = 'pearson', min_correlation: float = 0.0) -> pd.DataFrame:
        """Get correlation matrix for numeric columns."""
        numeric_cols = get_numeric_columns(self.df)

        if len(numeric_cols) < 2:
            return pd.DataFrame()

        corr_matrix = self.df[numeric_cols].corr(method=method)

        if min_correlation > 0:
            mask = np.abs(corr_matrix) >= min_correlation
            corr_matrix = corr_matrix.where(mask)

        return corr_matrix

    def get_high_correlations(self, threshold: float = 0.7, method: str = 'pearson') -> pd.DataFrame:
        """Find pairs of highly correlated features."""
        corr_matrix = self.get_correlation_matrix(method=method)

        if corr_matrix.empty:
            return pd.DataFrame()

        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        df_high_corr = pd.DataFrame(high_corr)
        if not df_high_corr.empty:
            df_high_corr = df_high_corr.sort_values('correlation', key=abs, ascending=False)

        return df_high_corr.reset_index(drop=True)

    def get_cardinality_info(self) -> pd.DataFrame:
        """Get cardinality information for all columns."""
        cardinality = []
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            cardinality.append({
                'column': col,
                'unique_count': unique_count,
                'cardinality_ratio': unique_count / len(self.df),
                'dtype': str(self.df[col].dtype)
            })

        df_cardinality = pd.DataFrame(cardinality)
        return df_cardinality.sort_values('unique_count', ascending=False).reset_index(drop=True)

    def calculate_vif(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF) for multicollinearity detection.

        VIF measures how much the variance of a regression coefficient is inflated
        due to multicollinearity. Values > 10 indicate high multicollinearity.

        Args:
            columns: List of numeric columns to analyze. If None, uses all numeric columns.

        Returns:
            DataFrame with columns: feature, VIF (sorted by VIF descending)
            Returns empty DataFrame if insufficient columns or calculation fails.

        Note:
            - VIF > 10: High multicollinearity (consider removing feature)
            - VIF > 5: Moderate multicollinearity
            - VIF < 5: Low multicollinearity
        """
        if columns is None:
            columns = get_numeric_columns(self.df)
        else:
            columns = validate_columns(self.df, columns)

        if len(columns) < 2:
            logger.warning("Need at least 2 numeric columns for VIF calculation")
            return pd.DataFrame()

        # Prepare data
        df_vif = self.df[columns].fillna(self.df[columns].mean())

        # Remove constant columns
        df_vif = df_vif.loc[:, df_vif.std() > 0]

        if df_vif.shape[1] < 2:
            logger.warning("Insufficient non-constant features for VIF calculation")
            return pd.DataFrame()

        try:
            vif_data = []
            for i, col in enumerate(df_vif.columns):
                vif = variance_inflation_factor(df_vif.values, i)
                vif_data.append({'feature': col, 'VIF': vif})

            vif_df = pd.DataFrame(vif_data)
            vif_df = vif_df.sort_values('VIF', ascending=False)

            logger.info(f"Calculated VIF for {len(vif_df)} features")
            return vif_df

        except Exception as e:
            logger.warning(f"Could not calculate VIF: {e}")
            return pd.DataFrame()

    def detect_misclassified_categorical(self, max_unique: int = 10,
                                         min_unique_ratio: float = 0.05) -> pd.DataFrame:
        """
        Detect numeric columns that should likely be categorical.

        Identifies numeric columns that may be flags, binary indicators, or low-cardinality
        categorical variables incorrectly stored as numeric types.

        Args:
            max_unique: Maximum unique values for a column to be considered categorical.
                       Default 10.
            min_unique_ratio: Minimum ratio of unique values to total rows. Columns with
                            lower ratios are candidates for categorical encoding. Default 0.05.

        Returns:
            DataFrame with columns: column, unique_count, unique_ratio, dtype, suggestion
            Sorted by unique_count ascending (most likely categorical first)
        """
        numeric_cols = get_numeric_columns(self.df)

        if len(numeric_cols) == 0:
            return pd.DataFrame()

        candidates = []
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue

            unique_count = col_data.nunique()
            unique_ratio = unique_count / len(col_data)

            # Check if column should be categorical
            is_candidate = False
            suggestion = ""

            # Binary/flag columns (exactly 2 unique values)
            if unique_count == 2:
                is_candidate = True
                values = sorted(col_data.unique())
                suggestion = f"Binary flag ({values[0]}, {values[1]}) - consider converting to categorical or boolean"

            # Low cardinality numeric columns
            elif unique_count <= max_unique:
                is_candidate = True
                suggestion = f"Low cardinality ({unique_count} categories) - likely ordinal or nominal"

            # Very low unique ratio (many repeated values)
            elif unique_ratio < min_unique_ratio:
                is_candidate = True
                suggestion = f"Very low unique ratio ({unique_ratio:.1%}) - possibly categorical with {unique_count} categories"

            # Integer-only columns with low cardinality
            elif col_data.dtype in ['int64', 'int32', 'int16', 'int8'] and unique_count <= 20:
                # Check if all values are integers (some int columns can have floats after operations)
                if (col_data == col_data.astype(int)).all():
                    is_candidate = True
                    suggestion = f"Integer column with {unique_count} values - likely categorical/ordinal"

            if is_candidate:
                candidates.append({
                    'column': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'dtype': str(self.df[col].dtype),
                    'suggestion': suggestion
                })

        df_candidates = pd.DataFrame(candidates)
        if not df_candidates.empty:
            df_candidates = df_candidates.sort_values('unique_count')

        return df_candidates.reset_index(drop=True)

    def suggest_binning(self, max_bins: int = 10, min_unique: int = 20) -> pd.DataFrame:
        """
        Suggest binning strategies for numeric columns.

        Analyzes numeric columns to recommend binning approaches based on:
        - Distribution characteristics (skewness, outliers)
        - Number of unique values
        - Value ranges

        Args:
            max_bins: Maximum number of bins to suggest. Default 10.
            min_unique: Minimum unique values required to suggest binning. Default 20.

        Returns:
            DataFrame with columns: column, strategy, num_bins, reason
            Sorted by priority (columns that would benefit most from binning first)
        """
        numeric_cols = get_numeric_columns(self.df)

        if len(numeric_cols) == 0:
            return pd.DataFrame()

        suggestions = []
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 10:
                continue

            unique_count = col_data.nunique()

            # Skip columns with very few unique values (already categorical-like)
            if unique_count < min_unique:
                continue

            # Calculate statistics
            skewness = col_data.skew()
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            # Detect outliers
            outlier_mask = (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
            outlier_pct = (outlier_mask.sum() / len(col_data)) * 100

            # Determine binning strategy and number of bins
            strategy = ""
            num_bins = 0
            reason = ""
            priority = 0

            # Strategy 1: Quantile binning for skewed distributions
            if abs(skewness) > 1.0:
                strategy = "quantile"
                num_bins = min(max_bins, max(5, unique_count // 20))
                reason = f"Skewed distribution (skewness={skewness:.2f})"
                priority = 3 if abs(skewness) > 2 else 2

            # Strategy 2: Equal-width for uniform distributions
            elif abs(skewness) < 0.5 and outlier_pct < 5:
                strategy = "uniform"
                num_bins = min(max_bins, max(5, unique_count // 20))
                reason = f"Relatively uniform distribution (skewness={skewness:.2f})"
                priority = 1

            # Strategy 3: Custom bins for outlier-heavy columns
            elif outlier_pct > 5:
                strategy = "quantile"
                num_bins = min(max_bins, max(4, unique_count // 30))
                reason = f"{outlier_pct:.1f}% outliers - quantile binning handles outliers better"
                priority = 3

            # Strategy 4: Default to uniform for normal-ish distributions
            else:
                strategy = "uniform"
                num_bins = min(max_bins, max(5, unique_count // 20))
                reason = "General purpose binning recommended"
                priority = 1

            suggestions.append({
                'column': col,
                'strategy': strategy,
                'num_bins': num_bins,
                'reason': reason,
                '_priority': priority
            })

        df_suggestions = pd.DataFrame(suggestions)
        if not df_suggestions.empty:
            # Sort by priority (descending) then by column name
            df_suggestions = df_suggestions.sort_values(['_priority', 'column'], ascending=[False, True])
            df_suggestions = df_suggestions.drop(columns=['_priority'])

        return df_suggestions.reset_index(drop=True)

    def plot_missing_values(self, figsize: tuple = (12, 6), show: bool = True) -> Optional[Figure]:
        """
        Visualize missing values.

        Args:
            figsize: Figure size as (width, height)
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if no missing values
        """
        missing_summary = self.get_missing_summary()

        if missing_summary.empty:
            logger.info("No missing values found")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=missing_summary, x='column', y='missing_percent', ax=ax)
        ax.set_xlabel('Column')
        ax.set_ylabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_correlation_heatmap(self, figsize: tuple = (10, 8), method: str = 'pearson',
                                  annot: bool = True, show: bool = True) -> Optional[Figure]:
        """
        Plot correlation heatmap.

        Args:
            figsize: Figure size as (width, height)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: If True, annotate cells with correlation values
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if insufficient data
        """
        corr_matrix = self.get_correlation_matrix(method=method)

        if corr_matrix.empty:
            logger.warning("Not enough numeric columns for correlation analysis")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0,
                    square=True, linewidths=1, ax=ax, fmt='.2f')
        ax.set_title(f'Correlation Heatmap ({method.capitalize()})')
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_distributions(self, columns: Optional[List[str]] = None, figsize: tuple = (15, 10),
                          show: bool = True) -> Optional[Figure]:
        """
        Plot distributions for numeric columns.

        Args:
            columns: List of columns to plot. If None, plots all numeric columns.
            figsize: Figure size as (width, height)
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if no columns to plot
        """
        if columns is None:
            columns = get_numeric_columns(self.df)
        else:
            columns = validate_columns(self.df, columns)

        if not columns:
            logger.warning("No numeric columns to plot")
            return None

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for idx, col in enumerate(columns):
            if idx < len(axes):
                self.df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')

        # Remove empty subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if show:
            plt.show()

        return fig


# ============================================================================
# TargetAnalyzer has been moved to feature_engineering_tk/target_analyzer/
# For backward compatibility, it can still be imported via __getattr__ below
# ============================================================================


def quick_analysis(df: pd.DataFrame) -> None:
    """
    Perform a quick comprehensive analysis of a dataframe.

    Prints a formatted report with:
    - Summary of data shape, memory, and issues found
    - Data quality analysis (missing values, duplicates)
    - Feature summaries (numeric and categorical)
    - Correlation analysis
    - Recommendations (misclassified columns, binning suggestions)

    Args:
        df: DataFrame to analyze
    """
    analyzer = DataAnalyzer(df)

    # Gather all data first
    basic_info_df = analyzer.get_basic_info()
    # Extract values from single-row DataFrame
    shape = basic_info_df['shape'].iloc[0]
    memory_mb = basic_info_df['memory_usage_mb'].iloc[0]
    duplicates = basic_info_df['duplicates'].iloc[0]

    missing = analyzer.get_missing_summary()
    numeric_summary = analyzer.get_numeric_summary()
    cat_summary = analyzer.get_categorical_summary()
    cardinality = analyzer.get_cardinality_info()
    high_corr = analyzer.get_high_correlations(threshold=0.7)
    misclassified = analyzer.detect_misclassified_categorical()
    binning = analyzer.suggest_binning()

    # Count issues for summary
    issues = []
    if not missing.empty:
        issues.append(f"{len(missing)} columns with missing values")
    if duplicates > 0:
        issues.append(f"{duplicates} duplicate rows")
    if not high_corr.empty:
        issues.append(f"{len(high_corr)} high correlation pairs")
    if not misclassified.empty:
        issues.append(f"{len(misclassified)} potentially misclassified columns")

    # Count column types
    n_numeric = len(numeric_summary.columns) if not numeric_summary.empty else 0
    n_categorical = len(cat_summary) if not cat_summary.empty else 0

    # Extract shape
    rows, cols = shape

    # Print header
    print("=" * 80)
    print("                         DATA ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Summary section
    print("SUMMARY")
    print(f"  Shape: {rows:,} rows x {cols} columns | Memory: {memory_mb:.2f} MB")
    if issues:
        print(f"  Issues: {len(issues)} found ({', '.join(issues[:2])}{'...' if len(issues) > 2 else ''})")
    else:
        print("  Issues: None detected")
    print()
    print("-" * 80)

    # Basic Information
    print()
    print(">>> BASIC INFORMATION")
    print(f"  Rows:        {rows:,}")
    print(f"  Columns:     {cols} ({n_numeric} numeric, {n_categorical} categorical)")
    print(f"  Memory:      {memory_mb:.2f} MB")
    print(f"  Duplicates:  {duplicates} rows ({duplicates/rows*100:.1f}%)")
    print()

    # Data Quality
    print(">>> DATA QUALITY")
    if not missing.empty:
        print("  Missing Values:")
        for _, row in missing.head(5).iterrows():
            print(f"    - {row['column']}: {row['missing_count']} ({row['missing_percent']:.1f}%)")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more columns")
    else:
        print("  Missing Values: None")

    if not high_corr.empty:
        print()
        print("  High Correlations (|r| >= 0.7):")
        for _, row in high_corr.head(3).iterrows():
            print(f"    - {row['feature_1']} <-> {row['feature_2']}: r = {row['correlation']:.3f}")
        if len(high_corr) > 3:
            print(f"    ... and {len(high_corr) - 3} more pairs")
    print()

    # Numeric Features
    print(">>> NUMERIC FEATURES" + (f" ({n_numeric} columns)" if n_numeric > 0 else ""))
    if not numeric_summary.empty:
        # Show condensed summary
        print(numeric_summary.to_string())
    else:
        print("  No numeric columns found.")
    print()

    # Categorical Features
    print(">>> CATEGORICAL FEATURES" + (f" ({n_categorical} columns)" if n_categorical > 0 else ""))
    if not cat_summary.empty:
        print(cat_summary.to_string(index=False))
    else:
        print("  No categorical columns found.")
    print()

    # Cardinality (condensed)
    print(">>> CARDINALITY")
    high_card = cardinality[cardinality['cardinality_ratio'] > 0.5]
    low_card = cardinality[cardinality['unique_count'] <= 10]
    if not high_card.empty:
        print("  High cardinality (>50% unique):")
        for _, row in high_card.head(3).iterrows():
            print(f"    - {row['column']}: {row['unique_count']} unique ({row['cardinality_ratio']*100:.1f}%)")
    if not low_card.empty and len(low_card) != len(cardinality):
        print("  Low cardinality (<=10 unique):")
        for _, row in low_card.head(3).iterrows():
            print(f"    - {row['column']}: {row['unique_count']} unique")
    if high_card.empty and (low_card.empty or len(low_card) == len(cardinality)):
        print("  All columns have moderate cardinality.")
    print()

    # Recommendations
    if not misclassified.empty or not binning.empty:
        print(">>> RECOMMENDATIONS")

        if not misclassified.empty:
            print()
            print("  Potential Misclassified Categoricals:")
            for _, row in misclassified.iterrows():
                print(f"    - {row['column']}: {row['suggestion']}")
            print("    Tip: Consider converting these to categorical type")

        if not binning.empty:
            print()
            print("  Binning Suggestions:")
            for _, row in binning.iterrows():
                print(f"    - {row['column']}: {row['strategy']} ({row['num_bins']} bins) - {row['reason']}")
            print("    Tip: Use FeatureEngineer.create_binning(column, bins, strategy)")

        print()

    print("=" * 80)


# Backward compatibility: Import TargetAnalyzer from new location
# This allows `from feature_engineering_tk.data_analysis import TargetAnalyzer` to still work
def __getattr__(name):
    """Provide backward compatibility for TargetAnalyzer import."""
    if name == 'TargetAnalyzer':
        import warnings
        warnings.warn(
            "Importing TargetAnalyzer from data_analysis is deprecated. "
            "Use 'from feature_engineering_tk import TargetAnalyzer' or "
            "'from feature_engineering_tk.target_analyzer import TargetAnalyzer' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .target_analyzer import TargetAnalyzer
        return TargetAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
