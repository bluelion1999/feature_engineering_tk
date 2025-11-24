import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

    # ============================================================================
    # PHASE 2: Classification-Specific Statistical Tests
    # ============================================================================

    def analyze_feature_target_relationship(self, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze relationship between features and target using appropriate statistical tests.

        For classification:
        - Chi-square test for categorical features
        - ANOVA F-test for numeric features

        For regression:
        - Pearson correlation for numeric features
        - ANOVA F-test for categorical features

        Args:
            feature_columns: List of feature columns to analyze. If None, uses all columns except target.

        Returns:
            DataFrame with columns: feature, test_type, statistic, pvalue, significant (at α=0.05)
        """
        if feature_columns is None:
            feature_columns = [col for col in self.df.columns if col != self.target_column]

        results = []
        target = self.df[self.target_column].dropna()

        for feature in feature_columns:
            if feature == self.target_column:
                continue

            feature_data = self.df[feature].dropna()

            # Skip if too many missing values
            if len(feature_data) < 10:
                logger.warning(f"Skipping feature '{feature}' due to insufficient data")
                continue

            try:
                if self.task == 'classification':
                    if pd.api.types.is_numeric_dtype(self.df[feature]):
                        # ANOVA F-test for numeric feature vs categorical target
                        groups = [self.df[self.df[self.target_column] == cls][feature].dropna()
                                  for cls in target.unique()]
                        groups = [g for g in groups if len(g) > 0]
                        if len(groups) >= 2:
                            statistic, pvalue = f_oneway(*groups)
                            results.append({
                                'feature': feature,
                                'test_type': 'ANOVA F-test',
                                'statistic': statistic,
                                'pvalue': pvalue,
                                'significant': pvalue < 0.05
                            })
                    else:
                        # Chi-square test for categorical feature vs categorical target
                        contingency_table = pd.crosstab(self.df[feature], self.df[self.target_column])
                        chi2, pvalue, dof, expected = chi2_contingency(contingency_table)
                        results.append({
                            'feature': feature,
                            'test_type': 'Chi-square test',
                            'statistic': chi2,
                            'pvalue': pvalue,
                            'significant': pvalue < 0.05
                        })

                elif self.task == 'regression':
                    if pd.api.types.is_numeric_dtype(self.df[feature]):
                        # Pearson correlation for numeric feature vs numeric target
                        valid_idx = self.df[[feature, self.target_column]].dropna().index
                        if len(valid_idx) > 2:
                            corr, pvalue = pearsonr(self.df.loc[valid_idx, feature],
                                                    self.df.loc[valid_idx, self.target_column])
                            results.append({
                                'feature': feature,
                                'test_type': 'Pearson correlation',
                                'statistic': corr,
                                'pvalue': pvalue,
                                'significant': pvalue < 0.05
                            })
                    else:
                        # ANOVA F-test for categorical feature vs numeric target
                        groups = [self.df[self.df[feature] == cat][self.target_column].dropna()
                                  for cat in self.df[feature].unique()]
                        groups = [g for g in groups if len(g) > 0]
                        if len(groups) >= 2:
                            statistic, pvalue = f_oneway(*groups)
                            results.append({
                                'feature': feature,
                                'test_type': 'ANOVA F-test',
                                'statistic': statistic,
                                'pvalue': pvalue,
                                'significant': pvalue < 0.05
                            })

            except Exception as e:
                logger.warning(f"Could not analyze feature '{feature}': {e}")
                continue

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('pvalue')

        return df_results

    def analyze_class_wise_statistics(self, feature_columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Compute statistics for each feature broken down by target class (classification only).

        Args:
            feature_columns: List of numeric feature columns. If None, uses all numeric columns.

        Returns:
            Dict mapping feature names to DataFrames with class-wise statistics
        """
        if self.task != 'classification':
            logger.warning("analyze_class_wise_statistics() is only available for classification tasks")
            return {}

        if feature_columns is None:
            feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col != self.target_column]

        results = {}
        classes = sorted(self.df[self.target_column].dropna().unique())

        for feature in feature_columns:
            if feature == self.target_column:
                continue

            class_stats = []
            for cls in classes:
                class_data = self.df[self.df[self.target_column] == cls][feature].dropna()
                if len(class_data) > 0:
                    class_stats.append({
                        'class': cls,
                        'count': len(class_data),
                        'mean': class_data.mean(),
                        'median': class_data.median(),
                        'std': class_data.std(),
                        'min': class_data.min(),
                        'max': class_data.max()
                    })

            if class_stats:
                results[feature] = pd.DataFrame(class_stats)

        return results

    def plot_feature_by_class(self, feature: str, plot_type: str = 'box',
                             figsize: tuple = (10, 6), show: bool = True):
        """
        Plot feature distribution by class (classification only).

        Args:
            feature: Feature column name
            plot_type: Type of plot ('box', 'violin', 'hist')
            figsize: Figure size as (width, height)
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if not classification
        """
        if self.task != 'classification':
            logger.warning("plot_feature_by_class() is only available for classification tasks")
            return None

        if feature not in self.df.columns:
            logger.warning(f"Feature '{feature}' not found in dataframe")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        if plot_type == 'box':
            self.df.boxplot(column=feature, by=self.target_column, ax=ax)
            ax.set_title(f'Box Plot: {feature} by {self.target_column}')
        elif plot_type == 'violin':
            sns.violinplot(data=self.df, x=self.target_column, y=feature, ax=ax)
            ax.set_title(f'Violin Plot: {feature} by {self.target_column}')
        elif plot_type == 'hist':
            for cls in sorted(self.df[self.target_column].dropna().unique()):
                class_data = self.df[self.df[self.target_column] == cls][feature].dropna()
                ax.hist(class_data, alpha=0.5, label=f'Class {cls}', bins=20)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram: {feature} by {self.target_column}')
            ax.legend()

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    # ============================================================================
    # PHASE 3: Regression-Specific Analysis
    # ============================================================================

    def analyze_feature_correlations(self, feature_columns: Optional[List[str]] = None,
                                     method: str = 'pearson') -> pd.DataFrame:
        """
        Analyze correlations between numeric features and target (regression only).

        Args:
            feature_columns: List of numeric features. If None, uses all numeric columns.
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            DataFrame with columns: feature, correlation, abs_correlation, pvalue, significant
        """
        if self.task != 'regression':
            logger.warning("analyze_feature_correlations() is only available for regression tasks")
            return pd.DataFrame()

        if feature_columns is None:
            feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col != self.target_column]

        results = []
        for feature in feature_columns:
            if feature == self.target_column:
                continue

            valid_idx = self.df[[feature, self.target_column]].dropna().index
            if len(valid_idx) < 3:
                continue

            try:
                if method == 'pearson':
                    corr, pvalue = pearsonr(self.df.loc[valid_idx, feature],
                                           self.df.loc[valid_idx, self.target_column])
                elif method == 'spearman':
                    corr, pvalue = spearmanr(self.df.loc[valid_idx, feature],
                                            self.df.loc[valid_idx, self.target_column])
                else:
                    logger.warning(f"Unknown correlation method: {method}")
                    continue

                results.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'pvalue': pvalue,
                    'significant': pvalue < 0.05
                })
            except Exception as e:
                logger.warning(f"Could not compute correlation for '{feature}': {e}")
                continue

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('abs_correlation', ascending=False)

        return df_results

    def analyze_mutual_information(self, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate mutual information between features and target.

        Args:
            feature_columns: List of features. If None, uses all columns except target.

        Returns:
            DataFrame with columns: feature, mutual_info, normalized_mi
        """
        if feature_columns is None:
            feature_columns = [col for col in self.df.columns if col != self.target_column]

        # Prepare data - only numeric features for MI
        numeric_features = [col for col in feature_columns
                           if pd.api.types.is_numeric_dtype(self.df[col])]

        if not numeric_features:
            logger.warning("No numeric features found for mutual information analysis")
            return pd.DataFrame()

        X = self.df[numeric_features].fillna(0)
        y = self.df[self.target_column].dropna()

        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(y) < 10:
            logger.warning("Insufficient data for mutual information analysis")
            return pd.DataFrame()

        try:
            if self.task == 'classification':
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)

            # Normalize by entropy
            max_mi = np.log(len(np.unique(y))) if self.task == 'classification' else np.max(mi_scores)
            if max_mi > 0:
                normalized_mi = mi_scores / max_mi
            else:
                normalized_mi = mi_scores

            results = pd.DataFrame({
                'feature': numeric_features,
                'mutual_info': mi_scores,
                'normalized_mi': normalized_mi
            })

            results = results.sort_values('mutual_info', ascending=False)
            return results

        except Exception as e:
            logger.warning(f"Could not compute mutual information: {e}")
            return pd.DataFrame()

    def plot_feature_vs_target(self, features: Optional[List[str]] = None,
                               max_features: int = 6, figsize: tuple = (15, 10), show: bool = True):
        """
        Create scatter plots of features vs target (regression only).

        Args:
            features: List of features to plot. If None, uses top correlated features.
            max_features: Maximum number of features to plot
            figsize: Figure size as (width, height)
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if not regression
        """
        if self.task != 'regression':
            logger.warning("plot_feature_vs_target() is only available for regression tasks")
            return None

        if features is None:
            # Use top correlated features
            corr_df = self.analyze_feature_correlations()
            if corr_df.empty:
                logger.warning("No features available for plotting")
                return None
            features = corr_df.head(max_features)['feature'].tolist()

        features = features[:max_features]
        n_features = len(features)

        if n_features == 0:
            logger.warning("No features to plot")
            return None

        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, feature in enumerate(features):
            ax = axes[idx]
            valid_idx = self.df[[feature, self.target_column]].dropna().index

            if len(valid_idx) > 0:
                ax.scatter(self.df.loc[valid_idx, feature],
                          self.df.loc[valid_idx, self.target_column],
                          alpha=0.5)
                ax.set_xlabel(feature)
                ax.set_ylabel(self.target_column)

                # Add regression line
                try:
                    z = np.polyfit(self.df.loc[valid_idx, feature],
                                  self.df.loc[valid_idx, self.target_column], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(self.df.loc[valid_idx, feature].min(),
                                        self.df.loc[valid_idx, feature].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8)
                except:
                    pass

                ax.set_title(f'{feature} vs {self.target_column}')

        # Remove empty subplots
        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def analyze_residuals(self, predictions: pd.Series) -> Dict[str, Any]:
        """
        Analyze residuals for regression tasks.

        Args:
            predictions: Predicted target values (must have same index as self.df)

        Returns:
            Dict containing residual statistics and test results
        """
        if self.task != 'regression':
            logger.warning("analyze_residuals() is only available for regression tasks")
            return {}

        # Align predictions with actual values
        common_idx = self.df[self.target_column].dropna().index.intersection(predictions.index)
        actual = self.df.loc[common_idx, self.target_column]
        pred = predictions.loc[common_idx]

        if len(actual) == 0:
            logger.warning("No valid data for residual analysis")
            return {}

        residuals = actual - pred

        results = {
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_min': residuals.min(),
            'residual_max': residuals.max(),
            'mae': np.abs(residuals).mean(),
            'rmse': np.sqrt((residuals ** 2).mean()),
            'r2_score': r2_score(actual, pred)
        }

        # Normality test on residuals
        if len(residuals) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals.sample(min(5000, len(residuals))))
                results['shapiro_stat'] = shapiro_stat
                results['shapiro_pvalue'] = shapiro_p
                results['residuals_normal'] = shapiro_p > 0.05
            except Exception as e:
                logger.warning(f"Could not compute normality test on residuals: {e}")

        return results

    def plot_residuals(self, predictions: pd.Series, figsize: tuple = (12, 5), show: bool = True):
        """
        Plot residual analysis (regression only).

        Args:
            predictions: Predicted target values
            figsize: Figure size as (width, height)
            show: If True, display the plot. Default True.

        Returns:
            matplotlib.figure.Figure: The figure object, or None if not regression
        """
        if self.task != 'regression':
            logger.warning("plot_residuals() is only available for regression tasks")
            return None

        common_idx = self.df[self.target_column].dropna().index.intersection(predictions.index)
        actual = self.df.loc[common_idx, self.target_column]
        pred = predictions.loc[common_idx]
        residuals = actual - pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')

        # Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot of Residuals')

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    # ============================================================================
    # PHASE 4: Common Data Quality Checks
    # ============================================================================

    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Analyze data quality issues including missing values, outliers, and potential leakage.

        Returns:
            Dict containing data quality metrics
        """
        results = {}

        # Missing values analysis
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        missing_by_feature = {}
        for col in feature_cols:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_by_feature[col] = {
                    'count': missing_count,
                    'percent': missing_count / len(self.df) * 100
                }

        results['missing_values'] = missing_by_feature
        results['target_missing'] = {
            'count': self.df[self.target_column].isnull().sum(),
            'percent': self.df[self.target_column].isnull().sum() / len(self.df) * 100
        }

        # Potential data leakage detection
        leakage_suspects = []

        if self.task == 'regression':
            # Check for perfect or near-perfect correlations
            corr_df = self.analyze_feature_correlations()
            if not corr_df.empty:
                perfect_corr = corr_df[corr_df['abs_correlation'] > 0.99]
                for _, row in perfect_corr.iterrows():
                    leakage_suspects.append({
                        'feature': row['feature'],
                        'reason': f'Near-perfect correlation ({row["correlation"]:.4f})',
                        'severity': 'high'
                    })

        elif self.task == 'classification':
            # Check for features with very low p-values and high test statistics
            rel_df = self.analyze_feature_target_relationship()
            if not rel_df.empty:
                suspicious = rel_df[rel_df['pvalue'] < 1e-10]
                for _, row in suspicious.iterrows():
                    leakage_suspects.append({
                        'feature': row['feature'],
                        'reason': f'Extremely significant relationship (p={row["pvalue"]:.2e})',
                        'severity': 'medium'
                    })

        results['leakage_suspects'] = leakage_suspects

        # Constant features
        constant_features = []
        for col in feature_cols:
            if self.df[col].nunique() == 1:
                constant_features.append(col)

        results['constant_features'] = constant_features

        return results

    def calculate_vif(self, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for multicollinearity detection.

        Args:
            feature_columns: List of numeric features. If None, uses all numeric columns.

        Returns:
            DataFrame with columns: feature, VIF
        """
        if feature_columns is None:
            feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col != self.target_column]

        if len(feature_columns) < 2:
            logger.warning("Need at least 2 features for VIF calculation")
            return pd.DataFrame()

        # Prepare data
        df_vif = self.df[feature_columns].fillna(self.df[feature_columns].mean())

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

            return vif_df

        except Exception as e:
            logger.warning(f"Could not calculate VIF: {e}")
            return pd.DataFrame()

    def generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Data quality recommendations
        quality = self.analyze_data_quality()

        if quality['missing_values']:
            high_missing = [k for k, v in quality['missing_values'].items() if v['percent'] > 50]
            if high_missing:
                recommendations.append(
                    f"⚠ {len(high_missing)} features have >50% missing values: "
                    f"{', '.join(high_missing[:3])}{'...' if len(high_missing) > 3 else ''}. "
                    "Consider dropping or imputing."
                )

        if quality['target_missing']['percent'] > 0:
            recommendations.append(
                f"⚠ Target column has {quality['target_missing']['percent']:.1f}% missing values. "
                "These rows cannot be used for supervised learning."
            )

        if quality['constant_features']:
            recommendations.append(
                f"⚠ {len(quality['constant_features'])} constant features provide no information. "
                f"Consider dropping: {', '.join(quality['constant_features'][:3])}"
            )

        if quality['leakage_suspects']:
            high_severity = [s for s in quality['leakage_suspects'] if s['severity'] == 'high']
            if high_severity:
                recommendations.append(
                    f"🚨 {len(high_severity)} features show signs of potential data leakage. "
                    "Review these features carefully!"
                )

        # Task-specific recommendations
        if self.task == 'classification':
            imbalance = self.get_class_imbalance_info()
            if imbalance and imbalance['severity'] != 'none':
                recommendations.append(f"⚙ {imbalance['recommendation']}")

        elif self.task == 'regression':
            dist = self.analyze_target_distribution()
            if 'is_normal' in dist and not dist['is_normal']:
                if abs(dist['skewness']) > 1:
                    recommendations.append(
                        "⚙ Target is highly skewed. Consider log transformation or robust regression methods."
                    )

        # Feature selection recommendations
        try:
            mi_df = self.analyze_mutual_information()
            if not mi_df.empty:
                low_mi = mi_df[mi_df['normalized_mi'] < 0.01]
                if len(low_mi) > 0:
                    recommendations.append(
                        f"📊 {len(low_mi)} features have very low mutual information with target. "
                        "Consider feature selection."
                    )
        except:
            pass

        # Multicollinearity check
        try:
            vif_df = self.calculate_vif()
            if not vif_df.empty:
                high_vif = vif_df[vif_df['VIF'] > 10]
                if len(high_vif) > 0:
                    recommendations.append(
                        f"📉 {len(high_vif)} features have high multicollinearity (VIF>10). "
                        f"Consider removing: {', '.join(high_vif.head(3)['feature'].tolist())}"
                    )
        except:
            pass

        if not recommendations:
            recommendations.append("✓ No major issues detected. Data quality looks good!")

        return recommendations


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
