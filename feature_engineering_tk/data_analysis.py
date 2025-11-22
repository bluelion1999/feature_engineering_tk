import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Data analysis class for exploratory data analysis and visualization.

    Provides statistical summaries, outlier detection, correlation analysis,
    and visualization tools.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataAnalyzer with a dataframe.

        Args:
            df: Input pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            logger.warning("Initializing with empty DataFrame")
        self.df = df.copy()

    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataframe."""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': self.df.duplicated().sum()
        }

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

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
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
                summary.append({
                    'column': col,
                    'unique_count': unique_count,
                    'top_value': top_value,
                    'top_value_freq': self.df[col].value_counts().iloc[0] if unique_count > 0 else 0,
                    'top_value_percent': (self.df[col].value_counts().iloc[0] / len(self.df) * 100) if unique_count > 0 else 0
                })

        return pd.DataFrame(summary)

    def detect_outliers_iqr(self, columns: Optional[List[str]] = None, multiplier: float = 1.5) -> Dict[str, pd.Series]:
        """Detect outliers using IQR method."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}
        for col in columns:
            if col not in self.df.columns:
                continue

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
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}
        for col in columns:
            if col not in self.df.columns:
                continue

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
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return pd.DataFrame()

        corr_matrix = numeric_df.corr(method=method)

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

    def plot_missing_values(self, figsize: tuple = (12, 6), show: bool = True):
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
                                  annot: bool = True, show: bool = True):
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
                          show: bool = True):
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
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

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


def quick_analysis(df: pd.DataFrame) -> None:
    """Perform a quick comprehensive analysis of a dataframe."""
    analyzer = DataAnalyzer(df)

    print("=" * 80)
    print("BASIC INFORMATION")
    print("=" * 80)
    basic_info = analyzer.get_basic_info()
    print(f"Shape: {basic_info['shape']}")
    print(f"Memory Usage: {basic_info['memory_usage_mb']:.2f} MB")
    print(f"Duplicates: {basic_info['duplicates']}")
    print()

    print("=" * 80)
    print("MISSING VALUES")
    print("=" * 80)
    missing = analyzer.get_missing_summary()
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string(index=False))
    print()

    print("=" * 80)
    print("NUMERIC SUMMARY")
    print("=" * 80)
    numeric_summary = analyzer.get_numeric_summary()
    if not numeric_summary.empty:
        print(numeric_summary)
    else:
        print("No numeric columns found.")
    print()

    print("=" * 80)
    print("CATEGORICAL SUMMARY")
    print("=" * 80)
    cat_summary = analyzer.get_categorical_summary()
    if not cat_summary.empty:
        print(cat_summary.to_string(index=False))
    else:
        print("No categorical columns found.")
    print()

    print("=" * 80)
    print("CARDINALITY INFORMATION")
    print("=" * 80)
    cardinality = analyzer.get_cardinality_info()
    print(cardinality.to_string(index=False))
    print()

    print("=" * 80)
    print("HIGH CORRELATIONS (|r| >= 0.7)")
    print("=" * 80)
    high_corr = analyzer.get_high_correlations(threshold=0.7)
    if not high_corr.empty:
        print(high_corr.to_string(index=False))
    else:
        print("No high correlations found.")
