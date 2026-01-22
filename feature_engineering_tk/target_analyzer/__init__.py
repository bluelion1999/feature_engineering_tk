"""
Target Analyzer Package

This package provides target-aware analysis for classification and regression tasks.
The TargetAnalyzer class is composed of several mixins, each providing specific functionality:

- TargetAnalyzerCore: Initialization, task detection, and distribution analysis
- StatisticalMixin: Feature-target relationships, correlations, mutual information
- QualityMixin: Data quality analysis, VIF calculation, recommendations
- VisualizationMixin: All plotting methods
- ReportingMixin: Report generation and export
- SuggestionsMixin: Feature engineering and model suggestions

Usage:
    from feature_engineering_tk import TargetAnalyzer
    # or
    from feature_engineering_tk.target_analyzer import TargetAnalyzer

    analyzer = TargetAnalyzer(df, target_column='target')
    report = analyzer.generate_full_report()
"""

from .core import TargetAnalyzerCore
from .statistical import StatisticalMixin
from .quality import QualityMixin
from .visualization import VisualizationMixin
from .reporting import ReportingMixin
from .suggestions import SuggestionsMixin


class TargetAnalyzer(
    SuggestionsMixin,
    ReportingMixin,
    VisualizationMixin,
    QualityMixin,
    StatisticalMixin,
    TargetAnalyzerCore
):
    """
    Target-aware analysis class for classification and regression tasks.

    Provides comprehensive statistical analysis when a target column is specified,
    including task-specific metrics, distributions, and visualizations.

    Automatically detects task type (classification vs regression) based on target
    column characteristics, or accepts explicit task specification.

    This class is composed of several mixins:
    - TargetAnalyzerCore: Core functionality (init, task detection, distributions)
    - StatisticalMixin: Statistical tests, correlations, mutual information
    - QualityMixin: Data quality, VIF, recommendations
    - VisualizationMixin: All plotting methods
    - ReportingMixin: Report generation and export
    - SuggestionsMixin: Feature engineering and model suggestions

    Example:
        >>> analyzer = TargetAnalyzer(df, target_column='price', task='regression')
        >>> # Core distribution analysis
        >>> dist = analyzer.analyze_target_distribution()
        >>> # Statistical tests
        >>> relationships = analyzer.analyze_feature_target_relationship()
        >>> # Generate full report
        >>> report = analyzer.generate_full_report()
        >>> # Export to file
        >>> analyzer.export_report('analysis.html', format='html')
    """
    pass


__all__ = ['TargetAnalyzer']
