"""
MLToolkit - A comprehensive Python toolkit for feature engineering and data analysis
"""

from .data_analysis import DataAnalyzer, quick_analysis
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .feature_selection import FeatureSelector, select_features_auto
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

__version__ = '1.0.0'

__all__ = [
    'DataAnalyzer',
    'quick_analysis',
    'FeatureEngineer',
    'DataPreprocessor',
    'FeatureSelector',
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
]
