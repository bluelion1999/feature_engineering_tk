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


# ============================================================================
# PHASE 2-4 TESTS
# ============================================================================

class TestPhase2ClassificationFeatures:
    """Test Phase 2: Classification-specific features"""

    def test_analyze_feature_target_relationship_classification(self, classification_df):
        """Test feature-target relationship analysis for classification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        results = analyzer.analyze_feature_target_relationship()

        assert isinstance(results, pd.DataFrame)
        assert 'feature' in results.columns
        assert 'test_type' in results.columns
        assert 'statistic' in results.columns
        assert 'pvalue' in results.columns
        assert 'significant' in results.columns

    def test_analyze_class_wise_statistics(self, classification_df):
        """Test class-wise statistics computation"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        stats = analyzer.analyze_class_wise_statistics()

        assert isinstance(stats, dict)
        assert len(stats) > 0
        for feature, df_stats in stats.items():
            assert isinstance(df_stats, pd.DataFrame)
            assert 'class' in df_stats.columns
            assert 'mean' in df_stats.columns
            assert 'std' in df_stats.columns

    def test_plot_feature_by_class_box(self, classification_df):
        """Test box plot for feature by class"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        fig = analyzer.plot_feature_by_class('feature1', plot_type='box', show=False)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_feature_by_class_violin(self, classification_df):
        """Test violin plot for feature by class"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        fig = analyzer.plot_feature_by_class('feature1', plot_type='violin', show=False)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_feature_by_class_hist(self, classification_df):
        """Test histogram for feature by class"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        fig = analyzer.plot_feature_by_class('feature1', plot_type='hist', show=False)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_class_wise_stats_wrong_task(self, regression_df):
        """Test that class-wise stats returns empty dict for regression"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        stats = analyzer.analyze_class_wise_statistics()
        assert stats == {}


class TestPhase3RegressionFeatures:
    """Test Phase 3: Regression-specific features"""

    def test_analyze_feature_correlations_pearson(self, regression_df):
        """Test Pearson correlation analysis"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        correlations = analyzer.analyze_feature_correlations(method='pearson')

        assert isinstance(correlations, pd.DataFrame)
        assert 'feature' in correlations.columns
        assert 'correlation' in correlations.columns
        assert 'abs_correlation' in correlations.columns
        assert 'pvalue' in correlations.columns
        assert 'significant' in correlations.columns

    def test_analyze_feature_correlations_spearman(self, regression_df):
        """Test Spearman correlation analysis"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        correlations = analyzer.analyze_feature_correlations(method='spearman')

        assert isinstance(correlations, pd.DataFrame)
        assert len(correlations) > 0

    def test_analyze_mutual_information_regression(self, regression_df):
        """Test mutual information for regression"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        mi_results = analyzer.analyze_mutual_information()

        assert isinstance(mi_results, pd.DataFrame)
        if not mi_results.empty:
            assert 'feature' in mi_results.columns
            assert 'mutual_info' in mi_results.columns
            assert 'normalized_mi' in mi_results.columns

    def test_analyze_mutual_information_classification(self, classification_df):
        """Test mutual information for classification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        mi_results = analyzer.analyze_mutual_information()

        assert isinstance(mi_results, pd.DataFrame)
        if not mi_results.empty:
            assert 'feature' in mi_results.columns

    def test_plot_feature_vs_target(self, regression_df):
        """Test scatter plots of features vs target"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        fig = analyzer.plot_feature_vs_target(features=['feature1', 'feature2'], show=False)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_feature_vs_target_auto_select(self, regression_df):
        """Test auto-selection of top features for plotting"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        fig = analyzer.plot_feature_vs_target(max_features=2, show=False)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_analyze_residuals(self, regression_df):
        """Test residual analysis"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')

        # Create mock predictions
        predictions = regression_df['target'] + np.random.randn(len(regression_df)) * 5
        residuals = analyzer.analyze_residuals(predictions)

        assert isinstance(residuals, dict)
        assert 'residual_mean' in residuals
        assert 'residual_std' in residuals
        assert 'mae' in residuals
        assert 'rmse' in residuals
        assert 'r2_score' in residuals

    def test_plot_residuals(self, regression_df):
        """Test residual plotting"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')

        # Create mock predictions
        predictions = regression_df['target'] + np.random.randn(len(regression_df)) * 5
        fig = analyzer.plot_residuals(predictions, show=False)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_regression_methods_wrong_task(self, classification_df):
        """Test regression methods return empty on classification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')

        correlations = analyzer.analyze_feature_correlations()
        assert correlations.empty

        fig = analyzer.plot_feature_vs_target(show=False)
        assert fig is None


class TestPhase4CommonFeatures:
    """Test Phase 4: Common features for both tasks"""

    def test_analyze_data_quality(self, classification_df):
        """Test data quality analysis"""
        # Add some issues to test
        df = classification_df.copy()
        df.loc[0:50, 'feature1'] = None  # Add missing values
        df['constant_feature'] = 1  # Add constant feature

        analyzer = TargetAnalyzer(df, target_column='target')
        quality = analyzer.analyze_data_quality()

        assert isinstance(quality, dict)
        assert 'missing_values' in quality
        assert 'target_missing' in quality
        assert 'leakage_suspects' in quality
        assert 'constant_features' in quality

    def test_calculate_vif(self, regression_df):
        """Test VIF calculation"""
        # Add correlated features
        df = regression_df.copy()
        df['feature4'] = df['feature1'] * 2 + np.random.randn(len(df)) * 0.1

        analyzer = TargetAnalyzer(df, target_column='target')
        vif = analyzer.calculate_vif()

        assert isinstance(vif, pd.DataFrame)
        if not vif.empty:
            assert 'feature' in vif.columns
            assert 'VIF' in vif.columns

    def test_vif_insufficient_features(self):
        """Test VIF with insufficient features"""
        df = pd.DataFrame({
            'feature1': range(100),
            'target': np.random.rand(100)
        })
        analyzer = TargetAnalyzer(df, target_column='target')
        vif = analyzer.calculate_vif()

        assert vif.empty

    def test_generate_recommendations_classification(self, classification_df):
        """Test recommendation generation for classification"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')
        recommendations = analyzer.generate_recommendations()

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        for rec in recommendations:
            assert isinstance(rec, str)

    def test_generate_recommendations_regression(self, regression_df):
        """Test recommendation generation for regression"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')
        recommendations = analyzer.generate_recommendations()

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_generate_recommendations_with_issues(self):
        """Test recommendations with data quality issues"""
        df = pd.DataFrame({
            'feature1': [1] * 100,  # Constant feature
            'feature2': [None] * 80 + list(range(20)),  # High missing
            'feature3': range(100),
            'target': np.random.choice([0, 1], 100)
        })
        analyzer = TargetAnalyzer(df, target_column='target')
        recommendations = analyzer.generate_recommendations()

        # Should have recommendations about constant feature and high missing values
        rec_text = ' '.join(recommendations)
        assert 'constant' in rec_text.lower() or 'missing' in rec_text.lower()

    def test_data_quality_leakage_detection_regression(self):
        """Test leakage detection for regression"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        # Create near-perfect correlation (potential leakage)
        df['leaky_feature'] = df['target'] + np.random.randn(100) * 0.01

        analyzer = TargetAnalyzer(df, target_column='target')
        quality = analyzer.analyze_data_quality()

        assert len(quality['leakage_suspects']) > 0

    def test_data_quality_missing_values(self):
        """Test missing value detection"""
        df = pd.DataFrame({
            'feature1': [1, None, 3, 4, 5],
            'feature2': [None] * 5,
            'target': [0, 1, 0, 1, 0]
        })
        analyzer = TargetAnalyzer(df, target_column='target')
        quality = analyzer.analyze_data_quality()

        assert 'feature1' in quality['missing_values']
        assert 'feature2' in quality['missing_values']
        assert quality['missing_values']['feature2']['percent'] == 100


class TestPhase2to4Integration:
    """Integration tests for Phase 2-4 features"""

    def test_full_classification_workflow(self, classification_df):
        """Test complete classification analysis workflow"""
        analyzer = TargetAnalyzer(classification_df, target_column='target')

        # Phase 1
        dist = analyzer.analyze_class_distribution()
        assert not dist.empty

        # Phase 2
        relationships = analyzer.analyze_feature_target_relationship()
        assert not relationships.empty

        class_stats = analyzer.analyze_class_wise_statistics()
        assert len(class_stats) > 0

        # Phase 3 (should work for both)
        mi = analyzer.analyze_mutual_information()

        # Phase 4
        quality = analyzer.analyze_data_quality()
        assert isinstance(quality, dict)

        recommendations = analyzer.generate_recommendations()
        assert len(recommendations) > 0

    def test_full_regression_workflow(self, regression_df):
        """Test complete regression analysis workflow"""
        analyzer = TargetAnalyzer(regression_df, target_column='target')

        # Phase 1
        dist = analyzer.analyze_target_distribution()
        assert 'mean' in dist

        # Phase 2 (relationship tests work for regression too)
        relationships = analyzer.analyze_feature_target_relationship()

        # Phase 3
        correlations = analyzer.analyze_feature_correlations()
        assert not correlations.empty

        mi = analyzer.analyze_mutual_information()

        # Create predictions for residual analysis
        predictions = regression_df['target'] + np.random.randn(len(regression_df)) * 5
        residuals = analyzer.analyze_residuals(predictions)
        assert 'r2_score' in residuals

        # Phase 4
        quality = analyzer.analyze_data_quality()
        vif = analyzer.calculate_vif()
        recommendations = analyzer.generate_recommendations()
        assert len(recommendations) > 0
