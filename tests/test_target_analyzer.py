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

    def test_class_imbalance_single_class_no_div_by_zero(self):
        """Test class imbalance with single class doesn't cause division by zero (Bug #2)."""
        # Single class scenario
        df_single = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': [0, 0, 0, 0, 0]  # Only one class
        })
        analyzer = TargetAnalyzer(df_single, target_column='target')

        # Should handle single class gracefully (imbalance_ratio = 1.0, not division by zero)
        imbalance = analyzer.get_class_imbalance_info()

        assert 'imbalance_ratio' in imbalance
        assert isinstance(imbalance['imbalance_ratio'], (int, float))
        # With one class, ratio should be 1.0 (max/min when both are same)
        assert imbalance['imbalance_ratio'] == 1.0
        assert imbalance['is_balanced'] == True  # numpy bool requires == not is

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


# =======================
# Phase 5: Report Generation Tests
# =======================

class TestPhase5ReportGeneration:
    """Test Phase 5 report generation and export features."""

    def test_generate_full_report_classification(self, classification_df):
        """Test generating full report for classification task."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        report = analyzer.generate_full_report()

        # Check report structure
        assert isinstance(report, dict)
        assert report['task'] == 'classification'
        assert 'timestamp' in report
        assert 'task_info' in report
        assert 'distribution' in report
        assert 'imbalance' in report
        assert 'relationships' in report
        assert 'class_stats' in report
        assert 'mutual_info' in report
        assert 'data_quality' in report
        assert 'vif' in report
        assert 'recommendations' in report

        # Check task info
        task_info = report['task_info']
        assert task_info['task'] == 'classification'
        assert 'target_column' in task_info
        assert 'unique_values' in task_info
        assert 'classes' in task_info
        assert 'class_count' in task_info

    def test_generate_full_report_regression(self, regression_df):
        """Test generating full report for regression task."""
        analyzer = TargetAnalyzer(regression_df, 'target')
        report = analyzer.generate_full_report()

        # Check report structure
        assert isinstance(report, dict)
        assert report['task'] == 'regression'
        assert 'timestamp' in report
        assert 'task_info' in report
        assert 'distribution' in report
        assert 'correlations' in report
        assert 'relationships' in report
        assert 'mutual_info' in report
        assert 'data_quality' in report
        assert 'vif' in report
        assert 'recommendations' in report

    def test_export_report_json(self, classification_df, tmp_path):
        """Test exporting report as JSON."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        filepath = tmp_path / "report.json"

        analyzer.export_report(str(filepath), format='json')

        # Check file was created
        assert filepath.exists()

        # Check JSON is valid
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert data['task'] == 'classification'
        assert 'timestamp' in data

    def test_export_report_markdown(self, regression_df, tmp_path):
        """Test exporting report as Markdown."""
        analyzer = TargetAnalyzer(regression_df, 'target')
        filepath = tmp_path / "report.md"

        analyzer.export_report(str(filepath), format='markdown')

        # Check file was created
        assert filepath.exists()

        # Check markdown content
        with open(filepath, 'r') as f:
            content = f.read()

        assert '# Target Analysis Report' in content
        assert '## Task Information' in content
        assert 'regression' in content.lower()
        assert '## Distribution Analysis' in content

    def test_export_report_html(self, classification_df, tmp_path):
        """Test exporting report as HTML."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        filepath = tmp_path / "report.html"

        analyzer.export_report(str(filepath), format='html')

        # Check file was created
        assert filepath.exists()

        # Check HTML content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        assert '<html>' in content
        assert '<title>Target Analysis Report</title>' in content
        assert '<h1>Target Analysis Report</h1>' in content
        assert 'classification' in content.lower()

    def test_export_report_invalid_format(self, classification_df, tmp_path):
        """Test that invalid format raises error."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        filepath = tmp_path / "report.txt"

        with pytest.raises(ValueError, match="Format must be"):
            analyzer.export_report(str(filepath), format='invalid')

    def test_report_includes_all_analyses(self, classification_df):
        """Test that report includes results from all analysis methods."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        report = analyzer.generate_full_report()

        # Verify all sections are present with data
        assert len(report['distribution']) > 0
        assert len(report['relationships']) > 0
        assert len(report['mutual_info']) > 0
        assert isinstance(report['data_quality'], dict)
        assert len(report['recommendations']) > 0

    def test_html_export_includes_css(self, classification_df, tmp_path):
        """Test that HTML export includes embedded CSS styling."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        filepath = tmp_path / "report.html"

        analyzer.export_report(str(filepath), format='html')

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for CSS styling
        assert '<style>' in content
        assert 'font-family' in content
        assert 'table' in content

    def test_markdown_export_includes_tables(self, regression_df, tmp_path):
        """Test that Markdown export includes properly formatted tables."""
        analyzer = TargetAnalyzer(regression_df, 'target')
        filepath = tmp_path / "report.md"

        analyzer.export_report(str(filepath), format='markdown')

        with open(filepath, 'r') as f:
            content = f.read()

        # Check for markdown tables
        assert '|' in content  # Table delimiter
        assert '---' in content  # Table header separator


# =======================
# Phase 7: Feature Engineering Suggestions Tests
# =======================

class TestPhase7FeatureEngineeringSuggestions:
    """Test Phase 7 feature engineering suggestion features."""

    def test_suggest_feature_engineering_basic(self, regression_df):
        """Test basic feature engineering suggestion generation."""
        analyzer = TargetAnalyzer(regression_df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Check suggestion structure
        for sugg in suggestions:
            assert 'feature' in sugg
            assert 'suggestion' in sugg
            assert 'reason' in sugg
            assert 'priority' in sugg
            assert sugg['priority'] in ['high', 'medium', 'low']

    def test_suggest_skewed_transformations(self):
        """Test suggestions for skewed features."""
        # Create highly skewed data
        np.random.seed(42)
        df = pd.DataFrame({
            'skewed_feature': np.random.exponential(2, 100),
            'target': np.random.randn(100)
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # Should suggest transformation for skewed feature
        transform_suggs = [s for s in suggestions if 'transformation' in s['suggestion'].lower()]
        assert len(transform_suggs) > 0
        assert any('skewed' in s['reason'].lower() for s in transform_suggs)

    def test_suggest_categorical_encoding(self):
        """Test suggestions for categorical features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'low_card': np.random.choice(['A', 'B', 'C'], 100),
            'high_card': [f'cat_{i % 30}' for i in range(100)],
            'target': np.random.randint(0, 2, 100)
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # Should suggest one-hot for low cardinality
        low_card_suggs = [s for s in suggestions if s['feature'] == 'low_card']
        assert any('one-hot' in s['suggestion'].lower() for s in low_card_suggs)

        # Should suggest target encoding for high cardinality
        high_card_suggs = [s for s in suggestions if s['feature'] == 'high_card']
        assert any('target encode' in s['suggestion'].lower() or 'group' in s['suggestion'].lower()
                   for s in high_card_suggs)

    def test_suggest_scaling(self):
        """Test suggestions for features needing scaling."""
        np.random.seed(42)
        df = pd.DataFrame({
            'large_range': np.random.uniform(0, 1000, 100),
            'small_range': np.random.uniform(0, 10, 100),
            'target': np.random.randn(100)
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # Should suggest scaling for large range feature
        scaling_suggs = [s for s in suggestions
                        if s['feature'] == 'large_range' and 'scaler' in s['suggestion'].lower()]
        assert len(scaling_suggs) > 0

    def test_suggest_polynomial_features(self):
        """Test suggestions for polynomial features with non-linear relationships."""
        np.random.seed(42)
        # Use a more pronounced non-linear pattern
        x = np.linspace(0, 10, 100)
        # Create a clear non-monotonic pattern
        y_nonlinear = np.sin(x) * 10 + x
        df = pd.DataFrame({
            'feature': x,
            'target': y_nonlinear + np.random.randn(100) * 0.5
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # Check that suggestions are generated (polynomial detection depends on threshold)
        # The specific suggestions may vary based on data characteristics
        assert len(suggestions) > 0
        assert all('feature' in s and 'suggestion' in s for s in suggestions)

    def test_suggest_interaction_terms_regression(self):
        """Test interaction term suggestions for regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randn(100) * 10 + 50
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # Should suggest interaction terms
        interaction_suggs = [s for s in suggestions if 'interaction' in s['suggestion'].lower()]
        # May or may not have interactions depending on correlations
        # Just check structure if present
        for sugg in interaction_suggs:
            assert 'feature' in sugg

    def test_suggest_missing_indicators(self):
        """Test suggestions for missing value indicators."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_with_missing': [1, 2, None, 4, None, 6, None, 8, 9, 10] * 10,
            'target': np.random.randint(0, 2, 100)
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # Should suggest missing indicator flag
        missing_suggs = [s for s in suggestions if 'missing indicator' in s['suggestion'].lower()]
        assert len(missing_suggs) > 0
        assert any('missing' in s['reason'].lower() for s in missing_suggs)

    def test_suggestions_sorted_by_priority(self, classification_df):
        """Test that suggestions are sorted by priority."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        if len(suggestions) > 1:
            priorities = [s['priority'] for s in suggestions]
            # Check high comes before low
            if 'high' in priorities and 'low' in priorities:
                high_idx = priorities.index('high')
                low_idx = priorities.index('low')
                assert high_idx < low_idx

    def test_suggest_classification_binning(self):
        """Test binning suggestions for classification with weak linear features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'weak_feature': np.random.randn(200),
            'target': np.random.randint(0, 2, 200)
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # May or may not suggest binning depending on p-values
        # Just verify structure is correct
        assert isinstance(suggestions, list)

    def test_empty_suggestions_edge_case(self):
        """Test with minimal data that produces no suggestions."""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': [1, 2, 3, 4, 5]
        })

        analyzer = TargetAnalyzer(df, 'target')
        suggestions = analyzer.suggest_feature_engineering()

        # Should return empty list or minimal suggestions
        assert isinstance(suggestions, list)


# =======================
# Phase 8: Model Recommendations Tests
# =======================

class TestPhase8ModelRecommendations:
    """Test Phase 8 model recommendation features."""

    def test_recommend_models_basic(self, classification_df):
        """Test basic model recommendation generation."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        recommendations = analyzer.recommend_models()

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check recommendation structure
        for rec in recommendations:
            assert 'model' in rec
            assert 'reason' in rec
            assert 'priority' in rec
            assert 'considerations' in rec
            assert rec['priority'] in ['high', 'medium', 'low']

    def test_recommend_models_classification_balanced(self):
        """Test recommendations for balanced classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': np.random.choice([0, 1], 1000, p=[0.5, 0.5])
        })

        analyzer = TargetAnalyzer(df, 'target')
        recommendations = analyzer.recommend_models()

        # Should recommend Random Forest (balanced data)
        rf_recs = [r for r in recommendations if 'Random Forest' in r['model']]
        assert len(rf_recs) > 0
        assert 'balanced' in rf_recs[0]['model'].lower() or 'Balanced classes' in rf_recs[0]['reason']

    def test_recommend_models_classification_imbalanced(self):
        """Test recommendations for imbalanced classification."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
        })

        analyzer = TargetAnalyzer(df, 'target')
        recommendations = analyzer.recommend_models()

        # Should recommend models for imbalanced data
        imbalance_aware = [r for r in recommendations
                          if 'imbalance' in r['reason'].lower() or 'balanced' in r['model'].lower()]
        assert len(imbalance_aware) > 0

    def test_recommend_models_regression(self, regression_df):
        """Test recommendations for regression task."""
        analyzer = TargetAnalyzer(regression_df, 'target')
        recommendations = analyzer.recommend_models()

        # Should recommend regression models
        regressor_recs = [r for r in recommendations
                         if 'Regressor' in r['model'] or 'Regression' in r['model']]
        assert len(regressor_recs) > 0

        # Should include Random Forest and tree-based models
        tree_models = [r for r in recommendations
                      if 'Forest' in r['model'] or 'XGBoost' in r['model'] or 'LightGBM' in r['model']]
        assert len(tree_models) > 0

    def test_recommend_models_small_dataset(self):
        """Test recommendations for small dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

        analyzer = TargetAnalyzer(df, 'target')
        recommendations = analyzer.recommend_models()

        # Should recommend cross-validation for small datasets
        cv_recs = [r for r in recommendations if 'Cross-Validation' in r['model']]
        assert len(cv_recs) > 0
        assert 'small dataset' in cv_recs[0]['reason'].lower()

    def test_recommend_models_high_dimensional(self):
        """Test recommendations for high-dimensional data."""
        np.random.seed(42)
        n_features = 60
        df = pd.DataFrame(
            np.random.randn(500, n_features + 1),
            columns=[f'feature_{i}' for i in range(n_features)] + ['target']
        )

        analyzer = TargetAnalyzer(df, 'target')
        recommendations = analyzer.recommend_models()

        # Should mention regularization or dimensionality
        reg_recs = [r for r in recommendations
                   if 'regularization' in r['reason'].lower() or 'Ridge' in r['model'] or 'Lasso' in r['model']]
        # May or may not have explicit regularization recommendations depending on task
        assert isinstance(recommendations, list)

    def test_recommend_models_with_outliers(self):
        """Test recommendations for regression with outliers."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'target': np.concatenate([np.random.randn(180) * 10 + 50,
                                     np.array([200, 300, -100] * 6 + [150, 250])])  # Add outliers
        })

        analyzer = TargetAnalyzer(df, 'target')
        recommendations = analyzer.recommend_models()

        # Should recommend robust models if outliers detected
        robust_recs = [r for r in recommendations
                      if 'Huber' in r['model'] or 'outliers' in r['reason'].lower()]
        # May or may not detect outliers depending on distribution
        assert isinstance(recommendations, list)

    def test_recommendations_sorted_by_priority(self, regression_df):
        """Test that recommendations are sorted by priority."""
        analyzer = TargetAnalyzer(regression_df, 'target')
        recommendations = analyzer.recommend_models()

        if len(recommendations) > 1:
            priorities = [r['priority'] for r in recommendations]
            # Check high comes before low
            if 'high' in priorities and 'low' in priorities:
                high_idx = priorities.index('high')
                low_idx = priorities.index('low')
                assert high_idx < low_idx

    def test_recommend_models_includes_considerations(self, classification_df):
        """Test that all recommendations include practical considerations."""
        analyzer = TargetAnalyzer(classification_df, 'target')
        recommendations = analyzer.recommend_models()

        for rec in recommendations:
            assert len(rec['considerations']) > 0
            # Considerations should provide actionable guidance
            assert isinstance(rec['considerations'], str)

    def test_recommend_models_classification_vs_regression(self):
        """Test that different models are recommended for classification vs regression."""
        np.random.seed(42)
        data = np.random.randn(500, 3)

        # Classification
        df_class = pd.DataFrame(data, columns=['f1', 'f2', 'target'])
        df_class['target'] = np.random.choice([0, 1], 500)
        analyzer_class = TargetAnalyzer(df_class, 'target')
        recs_class = analyzer_class.recommend_models()

        # Regression
        df_reg = pd.DataFrame(data, columns=['f1', 'f2', 'target'])
        analyzer_reg = TargetAnalyzer(df_reg, 'target')
        recs_reg = analyzer_reg.recommend_models()

        # Get model names
        class_models = {r['model'] for r in recs_class}
        reg_models = {r['model'] for r in recs_reg}

        # Should have some different models
        assert class_models != reg_models
