import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Optional, Dict, Any
from scipy import stats

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


class TargetAnalyzer:
    """
    Target-aware analysis class for classification and regression tasks.

    Provides comprehensive statistical analysis when a target column is specified,
    including task-specific metrics, distributions, and visualizations.

    Automatically detects task type (classification vs regression) based on target
    column characteristics, or accepts explicit task specification.
    """

    def __init__(self, df: pd.DataFrame, target_column: str, task: str = 'auto'):
        """
        Initialize TargetAnalyzer with a dataframe and target column.

        Args:
            df: Input pandas DataFrame
            target_column: Name of the target variable column
            task: Task type ('classification', 'regression', or 'auto').
                  Default 'auto' will detect based on target characteristics.

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If target_column not in dataframe or invalid task specified
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            logger.warning("Initializing with empty DataFrame")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        if task not in ['auto', 'classification', 'regression']:
            raise ValueError("task must be 'auto', 'classification', or 'regression'")

        self.df = df.copy()
        self.target_column = target_column
        self._analysis_cache = {}

        # Detect or set task type
        if task == 'auto':
            self.task = self._detect_task()
            logger.info(f"Auto-detected task type: {self.task}")
        else:
            self.task = task
            logger.info(f"Task type set to: {self.task}")

    def _detect_task(self) -> str:
        """
        Automatically detect whether this is a classification or regression task.

        Returns:
            str: 'classification' or 'regression'
        """
        target = self.df[self.target_column].dropna()

        if len(target) == 0:
            logger.warning("Target column is empty, defaulting to classification")
            return 'classification'

        # Check if numeric
        if pd.api.types.is_numeric_dtype(target):
            unique_count = target.nunique()
            unique_ratio = unique_count / len(target)

            # Heuristics for classification vs regression
            if unique_count == 2:
                return 'classification'
            elif unique_count <= 20 or unique_ratio < 0.05:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get information about the detected/specified task.

        Returns:
            Dict containing task type, target column info, and class information
            for classification tasks
        """
        target = self.df[self.target_column]

        info = {
            'task': self.task,
            'target_column': self.target_column,
            'target_dtype': str(target.dtype),
            'unique_values': target.nunique(),
            'missing_count': target.isnull().sum(),
            'missing_percent': (target.isnull().sum() / len(target) * 100)
        }

        if self.task == 'classification':
            info['classes'] = sorted(target.dropna().unique().tolist())
            info['class_count'] = len(info['classes'])

        return info

    def analyze_class_distribution(self) -> pd.DataFrame:
        """
        Analyze class distribution for classification tasks.

        Returns:
            DataFrame with columns: class, count, percentage, imbalance_ratio
            Empty DataFrame if task is not classification
        """
        if self.task != 'classification':
            logger.warning("analyze_class_distribution() is only available for classification tasks")
            return pd.DataFrame()

        if 'class_distribution' in self._analysis_cache:
            return self._analysis_cache['class_distribution']

        target = self.df[self.target_column].dropna()
        value_counts = target.value_counts()

        distribution = pd.DataFrame({
            'class': value_counts.index,
            'count': value_counts.values,
            'percentage': (value_counts.values / len(target) * 100)
        }).reset_index(drop=True)

        majority_count = distribution['count'].max()
        distribution['imbalance_ratio'] = majority_count / distribution['count']

        self._analysis_cache['class_distribution'] = distribution
        return distribution

    def get_class_imbalance_info(self) -> Dict[str, Any]:
        """
        Get detailed class imbalance information for classification tasks.

        Returns:
            Dict containing imbalance metrics, severity, and recommendations
            Empty dict if task is not classification
        """
        if self.task != 'classification':
            logger.warning("get_class_imbalance_info() is only available for classification tasks")
            return {}

        dist = self.analyze_class_distribution()

        if dist.empty:
            return {}

        majority_class = dist.loc[dist['count'].idxmax(), 'class']
        minority_class = dist.loc[dist['count'].idxmin(), 'class']
        imbalance_ratio = dist['count'].max() / dist['count'].min()

        info = {
            'is_balanced': imbalance_ratio <= 1.5,
            'imbalance_ratio': imbalance_ratio,
            'majority_class': majority_class,
            'majority_count': int(dist['count'].max()),
            'minority_class': minority_class,
            'minority_count': int(dist['count'].min()),
        }

        # Severity and recommendations
        if imbalance_ratio > 3:
            info['severity'] = 'severe'
            info['recommendation'] = 'Consider using SMOTE, class weights, or stratified sampling'
        elif imbalance_ratio > 1.5:
            info['severity'] = 'moderate'
            info['recommendation'] = 'Consider using class weights or stratified sampling'
        else:
            info['severity'] = 'none'
            info['recommendation'] = 'Classes are well balanced'

        return info

    def analyze_target_distribution(self) -> Dict[str, Any]:
        """
        Analyze target distribution for regression tasks.

        Returns:
            Dict containing comprehensive statistics including mean, median, std,
            skewness, kurtosis, and normality test results
            Empty dict if task is not regression
        """
        if self.task != 'regression':
            logger.warning("analyze_target_distribution() is only available for regression tasks")
            return {}

        if 'target_distribution' in self._analysis_cache:
            return self._analysis_cache['target_distribution']

        target = self.df[self.target_column].dropna()

        distribution = {
            'count': len(target),
            'mean': target.mean(),
            'median': target.median(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'range': target.max() - target.min(),
            'q25': target.quantile(0.25),
            'q75': target.quantile(0.75),
            'iqr': target.quantile(0.75) - target.quantile(0.25),
            'skewness': target.skew(),
            'kurtosis': target.kurtosis()
        }

        # Normality test (sample if dataset is large)
        if len(target) >= 3:
            try:
                sample_size = min(5000, len(target))
                sample = target.sample(sample_size, random_state=42)
                shapiro_stat, shapiro_p = stats.shapiro(sample)
                distribution['shapiro_stat'] = shapiro_stat
                distribution['shapiro_pvalue'] = shapiro_p
                distribution['is_normal'] = shapiro_p > 0.05
            except Exception as e:
                logger.warning(f"Could not compute normality test: {e}")

        self._analysis_cache['target_distribution'] = distribution
        return distribution

    def plot_class_distribution(self, figsize: tuple = (10, 6), show: bool = True):
        """
        Plot class distribution for classification tasks.

        Args:
            figsize: Figure size as (width, height)
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if not classification
        """
        if self.task != 'classification':
            logger.warning("plot_class_distribution() is only available for classification tasks")
            return None

        dist = self.analyze_class_distribution()

        if dist.empty:
            logger.warning("No data available for plotting")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar chart
        ax1.bar(dist['class'].astype(str), dist['count'], edgecolor='black')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution (Counts)')
        ax1.tick_params(axis='x', rotation=45)

        # Pie chart
        colors = plt.cm.Set3(range(len(dist)))
        ax2.pie(dist['percentage'], labels=dist['class'], autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Class Distribution (Percentage)')

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_target_distribution(self, figsize: tuple = (12, 5), show: bool = True):
        """
        Plot target distribution for regression tasks.

        Args:
            figsize: Figure size as (width, height)
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if not regression
        """
        if self.task != 'regression':
            logger.warning("plot_target_distribution() is only available for regression tasks")
            return None

        target = self.df[self.target_column].dropna()

        if len(target) == 0:
            logger.warning("No data available for plotting")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Histogram with mean/median lines
        axes[0].hist(target, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(target.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {target.mean():.2f}')
        axes[0].axvline(target.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {target.median():.2f}')
        axes[0].set_xlabel(self.target_column)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {self.target_column}')
        axes[0].legend()

        # Q-Q plot
        stats.probplot(target, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive text summary report.

        Returns:
            str: Formatted text report with task-specific statistics
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TARGET ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        task_info = self.get_task_info()
        lines.append(f"Task Type: {task_info['task'].upper()}")
        lines.append(f"Target Column: {task_info['target_column']}")
        lines.append(f"Target Data Type: {task_info['target_dtype']}")
        lines.append(f"Unique Values: {task_info['unique_values']}")
        lines.append(f"Missing Values: {task_info['missing_count']} ({task_info['missing_percent']:.2f}%)")
        lines.append("")

        if self.task == 'classification':
            lines.append("=" * 80)
            lines.append("CLASS DISTRIBUTION")
            lines.append("=" * 80)

            dist = self.analyze_class_distribution()
            lines.append(dist.to_string(index=False))
            lines.append("")

            imbalance_info = self.get_class_imbalance_info()
            lines.append("=" * 80)
            lines.append("CLASS IMBALANCE ANALYSIS")
            lines.append("=" * 80)
            lines.append(f"Balanced: {'Yes' if imbalance_info['is_balanced'] else 'No'}")
            lines.append(f"Imbalance Ratio: {imbalance_info['imbalance_ratio']:.2f}")
            lines.append(f"Severity: {imbalance_info['severity'].upper()}")
            lines.append(f"Majority Class: {imbalance_info['majority_class']} ({imbalance_info['majority_count']} samples)")
            lines.append(f"Minority Class: {imbalance_info['minority_class']} ({imbalance_info['minority_count']} samples)")
            lines.append(f"Recommendation: {imbalance_info['recommendation']}")

        elif self.task == 'regression':
            lines.append("=" * 80)
            lines.append("TARGET DISTRIBUTION")
            lines.append("=" * 80)

            dist = self.analyze_target_distribution()
            lines.append(f"Count: {dist['count']}")
            lines.append(f"Mean: {dist['mean']:.4f}")
            lines.append(f"Median: {dist['median']:.4f}")
            lines.append(f"Std Dev: {dist['std']:.4f}")
            lines.append(f"Min: {dist['min']:.4f}")
            lines.append(f"Max: {dist['max']:.4f}")
            lines.append(f"Range: {dist['range']:.4f}")
            lines.append(f"IQR: {dist['iqr']:.4f}")
            lines.append(f"Skewness: {dist['skewness']:.4f}")
            lines.append(f"Kurtosis: {dist['kurtosis']:.4f}")

            if 'is_normal' in dist:
                lines.append(f"Normality (Shapiro-Wilk p-value): {dist['shapiro_pvalue']:.4f}")
                lines.append(f"Appears Normal: {'Yes' if dist['is_normal'] else 'No'}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


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
