"""
Feature Engineering Toolkit - A comprehensive Python library for feature engineering and data analysis

Provides intelligent automation for ML workflows including preprocessing, feature engineering,
statistical analysis, and model recommendations.
"""

from .data_analysis import DataAnalyzer, TargetAnalyzer, quick_analysis
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .feature_selection import FeatureSelector, select_features_auto
from .base import FeatureEngineeringBase
from .exceptions import (
    MLToolkitError,
    ValidationError,
    ColumnNotFoundError,
    InvalidStrategyError,
    InvalidMethodError,
    DataTypeError,
    EmptyDataFrameError,
    InsufficientDataError,
    TransformerNotFittedError,
    ConstantColumnError,
)
from . import statistical_utils

__version__ = '2.4.2'

__all__ = [
    # Main Classes
    'DataAnalyzer',
    'TargetAnalyzer',
    'FeatureEngineer',
    'DataPreprocessor',
    'FeatureSelector',
    'FeatureEngineeringBase',
    # Helper Functions
    'quick_analysis',
    'select_features_auto',
    # Exceptions
    'MLToolkitError',
    'ValidationError',
    'ColumnNotFoundError',
    'InvalidStrategyError',
    'InvalidMethodError',
    'DataTypeError',
    'EmptyDataFrameError',
    'InsufficientDataError',
    'TransformerNotFittedError',
    'ConstantColumnError',
    # Statistical Utilities
    'statistical_utils',
]
