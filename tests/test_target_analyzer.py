"""
Tests for TargetAnalyzer class in data_analysis module
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from feature_engineering_tk import TargetAnalyzer


@pytest.fixture
def classification_df():
    """Create sample classification dataset"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })


@pytest.fixture
def regression_df():
    """Create sample regression dataset"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randn(1000) * 10 + 50
    })


@pytest.fixture
def multi_class_df():
    """Create multi-class classification dataset"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(500),
        'feature2': np.random.randn(500),
        'target': np.random.choice(['low', 'medium', 'high'], 500)
    })


class TestTargetAnalyzerInitialization:
    """Test TargetAnalyzer initialization"""

    def test_init_with_valid_classification_data(self, classification_df):
        """Test initialization with classification data"""
        analyzer = TargetAnalyzer(classification_df, target_column='target', task='auto')
        assert analyzer.task == 'classification'
        assert analyzer.target_column == 'target'
        assert len(analyzer.df) == 1000

    def test_init_with_valid_regression_data(self, regression_df):
        """Test initialization with regression data"""
        analyzer = TargetAnalyzer(regression_df, target_column='target', task='auto')
        assert analyzer.task == 'regression'
        assert analyzer.target_column == 'target'

    def test_init_with_explicit_task(self, classification_df):
        """Test initialization with explicit task specification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target', task='classification')
        assert analyzer.task == 'classification'

    def test_init_with_invalid_dataframe(self):
        """Test initialization with invalid input"""
        with pytest.raises(TypeError):
            TargetAnalyzer([1, 2, 3], target_column='target')

    def test_init_with_missing_target_column(self, classification_df):
        """Test initialization with missing target column"""
        with pytest.raises(ValueError, match="not found"):
            TargetAnalyzer(classification_df, target_column='nonexistent')

    def test_init_with_invalid_task(self, classification_df):
        """Test initialization with invalid task type"""
        with pytest.raises(ValueError, match="must be"):
            TargetAnalyzer(classification_df, target_column='target', task='invalid')

    def test_copy_dataframe(self, classification_df):
        """Test that dataframe is copied during initialization"""
        original_df = classification_df.copy()
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        analyzer.df.loc[0, 'feature1'] = 999
        assert original_df.loc[0, 'feature1'] != 999


class TestTaskDetection:
    """Test automatic task detection"""

    def test_detect_binary_classification(self, classification_df):
        """Test detection of binary classification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target', task='auto')
        assert analyzer.task == 'classification'

    def test_detect_multi_class_classification(self, multi_class_df):
        """Test detection of multi-class classification"""
        analyzer = TargetAnalyzer(multi_class_df, target_column='target', task='auto')
        assert analyzer.task == 'classification'

    def test_detect_regression(self, regression_df):
        """Test detection of regression"""
        analyzer = TargetAnalyzer(regression_df, target_column='target', task='auto')
        assert analyzer.task == 'regression'

    def test_detect_with_many_unique_values(self):
        """Test detection with many unique numeric values"""
        df = pd.DataFrame({
            'feature': range(100),
            'target': np.random.rand(100) * 100
        })
        analyzer = TargetAnalyzer(df, target_column='target', task='auto')
        assert analyzer.task == 'regression'

    def test_detect_with_categorical_target(self):
        """Test detection with categorical target"""
        df = pd.DataFrame({
            'feature': range(100),
            'target': np.random.choice(['cat', 'dog', 'bird'], 100)
        })
        analyzer = TargetAnalyzer(df, target_column='target', task='auto')
        assert analyzer.task == 'classification'


class TestClassificationAnalysis:
    """Test classification-specific methods"""

    def test_get_task_info_classification(self, classification_df):
        """Test get_task_info for classification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        info = analyzer.get_task_info()

        assert info['task'] == 'classification'
        assert info['target_column'] == 'target'
        assert 'classes' in info
        assert 'class_count' in info
        assert info['class_count'] == 2

    def test_analyze_class_distribution(self, classification_df):
        """Test class distribution analysis"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        dist = analyzer.analyze_class_distribution()

        assert isinstance(dist, pd.DataFrame)
        assert 'class' in dist.columns
        assert 'count' in dist.columns
        assert 'percentage' in dist.columns
        assert 'imbalance_ratio' in dist.columns
        assert len(dist) == 2

    def test_analyze_class_distribution_wrong_task(self, regression_df):
        """Test class distribution on regression task returns empty"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        dist = analyzer.analyze_class_distribution()
        assert dist.empty

    def test_get_class_imbalance_info(self, classification_df):
        """Test class imbalance information"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        imbalance = analyzer.get_class_imbalance_info()

        assert 'is_balanced' in imbalance
        assert 'imbalance_ratio' in imbalance
        assert 'majority_class' in imbalance
        assert 'minority_class' in imbalance
        assert 'severity' in imbalance
        assert 'recommendation' in imbalance

    def test_class_imbalance_severity_levels(self):
        """Test different imbalance severity levels"""
        # Balanced dataset
        df_balanced = pd.DataFrame({'target': [0]*50 + [1]*50})
        analyzer = TargetAnalyzer(df_balanced, target_column='target')
        imbalance = analyzer.get_class_imbalance_info()
        assert imbalance['severity'] == 'none'

        # Moderate imbalance
        df_moderate = pd.DataFrame({'target': [0]*60 + [1]*40})
        analyzer = TargetAnalyzer(df_moderate, target_column='target')
        imbalance = analyzer.get_class_imbalance_info()
        assert imbalance['severity'] == 'none'  # 60/40 = 1.5, at boundary

        # Severe imbalance
        df_severe = pd.DataFrame({'target': [0]*80 + [1]*20})
        analyzer = TargetAnalyzer(df_severe, target_column='target')
        imbalance = analyzer.get_class_imbalance_info()
        assert imbalance['severity'] == 'severe'

    def test_plot_class_distribution(self, classification_df):
        """Test class distribution plotting returns Figure"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        fig = analyzer.plot_class_distribution(show=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_class_distribution_wrong_task(self, regression_df):
        """Test plotting on wrong task returns None"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        fig = analyzer.plot_class_distribution(show=False)
        assert fig is None


class TestRegressionAnalysis:
    """Test regression-specific methods"""

    def test_get_task_info_regression(self, regression_df):
        """Test get_task_info for regression"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        info = analyzer.get_task_info()

        assert info['task'] == 'regression'
        assert info['target_column'] == 'target'
        assert 'classes' not in info

    def test_analyze_target_distribution(self, regression_df):
        """Test target distribution analysis for regression"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        dist = analyzer.analyze_target_distribution()

        assert isinstance(dist, dict)
        assert 'count' in dist
        assert 'mean' in dist
        assert 'median' in dist
        assert 'std' in dist
        assert 'min' in dist
        assert 'max' in dist
        assert 'range' in dist
        assert 'iqr' in dist
        assert 'skewness' in dist
        assert 'kurtosis' in dist

    def test_analyze_target_distribution_with_normality_test(self, regression_df):
        """Test that normality test is included"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        dist = analyzer.analyze_target_distribution()

        assert 'shapiro_stat' in dist
        assert 'shapiro_pvalue' in dist
        assert 'is_normal' in dist

    def test_analyze_target_distribution_wrong_task(self, classification_df):
        """Test target distribution on classification task returns empty"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        dist = analyzer.analyze_target_distribution()
        assert dist == {}

    def test_plot_target_distribution(self, regression_df):
        """Test target distribution plotting returns Figure"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        fig = analyzer.plot_target_distribution(show=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_target_distribution_wrong_task(self, classification_df):
        """Test plotting on wrong task returns None"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        fig = analyzer.plot_target_distribution(show=False)
        assert fig is None


class TestSummaryReport:
    """Test summary report generation"""

    def test_generate_summary_report_classification(self, classification_df):
        """Test summary report for classification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        report = analyzer.generate_summary_report()

        assert isinstance(report, str)
        assert 'TARGET ANALYSIS REPORT' in report
        assert 'CLASSIFICATION' in report
        assert 'CLASS DISTRIBUTION' in report
        assert 'CLASS IMBALANCE ANALYSIS' in report

    def test_generate_summary_report_regression(self, regression_df):
        """Test summary report for regression"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        report = analyzer.generate_summary_report()

        assert isinstance(report, str)
        assert 'TARGET ANALYSIS REPORT' in report
        assert 'REGRESSION' in report
        assert 'TARGET DISTRIBUTION' in report
        assert 'Mean:' in report
        assert 'Skewness:' in report

    def test_summary_report_contains_all_metrics(self, classification_df):
        """Test that summary report contains all expected metrics"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        report = analyzer.generate_summary_report()

        # Check basic info
        assert 'Task Type:' in report
        assert 'Target Column:' in report
        assert 'Unique Values:' in report

        # Check classification metrics
        assert 'Imbalance Ratio:' in report
        assert 'Recommendation:' in report


class TestCaching:
    """Test caching mechanism"""

    def test_class_distribution_caching(self, classification_df):
        """Test that class distribution is cached"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')

        # First call
        dist1 = analyzer.analyze_class_distribution()

        # Second call should use cache
        dist2 = analyzer.analyze_class_distribution()

        # Should be the same object (from cache)
        assert dist1 is dist2

    def test_target_distribution_caching(self, regression_df):
        """Test that target distribution is cached"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')

        # First call
        dist1 = analyzer.analyze_target_distribution()

        # Second call should use cache
        dist2 = analyzer.analyze_target_distribution()

        # Should be the same object (from cache)
        assert dist1 is dist2


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_target_column(self):
        """Test with all-null target column"""
        df = pd.DataFrame({
            'feature': [1, 2, 3],
            'target': [None, None, None]
        })
        analyzer = TargetAnalyzer(df, target_column='target')
        dist = analyzer.analyze_class_distribution()
        assert dist.empty

    def test_single_class(self):
        """Test classification with single class"""
        df = pd.DataFrame({
            'feature': range(100),
            'target': [1] * 100
        })
        analyzer = TargetAnalyzer(df, target_column='target')
        dist = analyzer.analyze_class_distribution()
        assert len(dist) == 1

    def test_very_small_dataset(self):
        """Test with very small dataset"""
        df = pd.DataFrame({
            'feature': [1, 2],
            'target': [0, 1]
        })
        analyzer = TargetAnalyzer(df, target_column='target', task='classification')
        info = analyzer.get_task_info()
        assert info['class_count'] == 2
