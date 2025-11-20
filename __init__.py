"""
MLToolkit - A comprehensive Python toolkit for feature engineering and data analysis
"""

from .data_analysis import DataAnalyzer, TargetAnalyzer, quick_analysis
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .feature_selection import FeatureSelector, select_features_auto

__version__ = '1.0.0'

__all__ = [
    'DataAnalyzer',
    'TargetAnalyzer',
    'quick_analysis',
    'FeatureEngineer',
    'DataPreprocessor',
    'FeatureSelector',
    'select_features_auto',
]
