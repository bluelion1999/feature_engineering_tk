"""
Tests for plotting functionality.

Tests that plotting methods return figure objects and handle edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from feature_engineering_tk import DataAnalyzer


class TestPlottingReturns:
    """Test suite for plotting methods returning figures."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'numeric2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'with_nulls': [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'categorical': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C']
        })

    def test_plot_missing_values_returns_figure(self, sample_df):
        """Test that plot_missing_values returns a Figure object."""
        analyzer = DataAnalyzer(sample_df)

        # Don't show the plot during tests
        fig = analyzer.plot_missing_values(show=False)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up

    def test_plot_missing_values_no_missing(self):
        """Test that plot returns None when no missing values."""
        df_no_missing = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        analyzer = DataAnalyzer(df_no_missing)

        fig = analyzer.plot_missing_values(show=False)

        assert fig is None

    def test_plot_correlation_heatmap_returns_figure(self, sample_df):
        """Test that plot_correlation_heatmap returns a Figure object."""
        analyzer = DataAnalyzer(sample_df)

        fig = analyzer.plot_correlation_heatmap(show=False)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_heatmap_insufficient_data(self):
        """Test that plot returns None with insufficient numeric columns."""
        df_no_numeric = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        analyzer = DataAnalyzer(df_no_numeric)

        fig = analyzer.plot_correlation_heatmap(show=False)

        assert fig is None

    def test_plot_distributions_returns_figure(self, sample_df):
        """Test that plot_distributions returns a Figure object."""
        analyzer = DataAnalyzer(sample_df)

        fig = analyzer.plot_distributions(show=False)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_distributions_no_numeric(self):
        """Test that plot returns None when no numeric columns."""
        df_no_numeric = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        analyzer = DataAnalyzer(df_no_numeric)

        fig = analyzer.plot_distributions(show=False)

        assert fig is None

    def test_plot_can_be_saved(self, sample_df, tmp_path):
        """Test that returned figure can be saved to file."""
        analyzer = DataAnalyzer(sample_df)

        fig = analyzer.plot_missing_values(show=False)

        # Save to temporary file
        save_path = tmp_path / "test_plot.png"
        fig.savefig(save_path)

        assert save_path.exists()
        plt.close(fig)

    def test_plot_can_be_customized(self, sample_df):
        """Test that returned figure can be customized."""
        analyzer = DataAnalyzer(sample_df)

        fig = analyzer.plot_missing_values(show=False)

        # Customize the figure
        fig.suptitle("Custom Title", fontsize=16)
        axes = fig.get_axes()
        assert len(axes) > 0

        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
