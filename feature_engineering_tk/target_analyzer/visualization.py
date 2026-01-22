"""
Visualization mixin for TargetAnalyzer.

Contains all plotting methods for classification and regression tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import logging
from typing import List, Optional
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


class VisualizationMixin:
    """
    Mixin providing visualization methods for TargetAnalyzer.

    Requires: self.df, self.target_column, self.task,
              self.analyze_class_distribution, self.analyze_target_distribution,
              self.analyze_feature_correlations
    """

    def plot_class_distribution(self, figsize: tuple = (10, 6), show: bool = True) -> Optional[Figure]:
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

    def plot_target_distribution(self, figsize: tuple = (12, 5), show: bool = True) -> Optional[Figure]:
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

    def plot_feature_by_class(self, feature: str, plot_type: str = 'box',
                             figsize: tuple = (10, 6), show: bool = True) -> Optional[Figure]:
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

    def plot_feature_vs_target(self, features: Optional[List[str]] = None,
                               max_features: int = 6, figsize: tuple = (15, 10), show: bool = True) -> Optional[Figure]:
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

    def plot_residuals(self, predictions: 'pd.Series', figsize: tuple = (12, 5), show: bool = True) -> Optional[Figure]:
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
