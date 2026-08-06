"""
Base classes for feature engineering toolkit.

This module provides the base class used across all
feature engineering toolkit classes.
"""

import logging
import pandas as pd

from .utils import validate_and_copy_dataframe

logger = logging.getLogger(__name__)


class FeatureEngineeringBase:
    """
    Base class for all feature engineering toolkit classes.

    Provides shared initialization logic and common methods for DataFrame
    manipulation classes including DataPreprocessor, FeatureEngineer,
    DataAnalyzer, TargetAnalyzer, and FeatureSelector.

    Attributes:
        df: Internal pandas DataFrame
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with DataFrame validation and copying.

        Args:
            df: Input pandas DataFrame

        Raises:
            TypeError: If df is not a pandas DataFrame
        """
        self.df = validate_and_copy_dataframe(df)
        logger.debug(f"{self.__class__.__name__} initialized with DataFrame shape {self.df.shape}")

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the current DataFrame.

        Returns:
            Copy of internal DataFrame
        """
        return self.df.copy()
