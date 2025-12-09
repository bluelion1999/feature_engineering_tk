"""
Base classes and decorators for feature engineering toolkit.

This module provides the base class and decorators used across all
feature engineering toolkit classes.
"""

import logging
from functools import wraps
from typing import Callable, Union
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


def inplace_transform(method: Callable) -> Callable:
    """
    Decorator to handle inplace DataFrame transformation pattern.

    This decorator automatically handles the common pattern of:
    1. Creating a copy of self.df if inplace=False
    2. Performing transformation on the appropriate DataFrame
    3. Updating self.df and returning self if inplace=True
    4. Returning the transformed DataFrame if inplace=False

    The decorated method should:
    - Accept an 'inplace' parameter (default False)
    - Perform transformations on 'df_result' variable
    - Return the transformed DataFrame

    Usage:
        @inplace_transform
        def some_method(self, columns, inplace=False):
            # Method will receive df_result variable automatically
            df_result[columns] = df_result[columns].transform(...)
            return df_result

    Args:
        method: Method to decorate (must have 'inplace' parameter)

    Returns:
        Decorated method with automatic inplace handling
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Extract inplace parameter (default to False)
        inplace = kwargs.get('inplace', False)

        # Create working DataFrame
        df_result = self.df if inplace else self.df.copy()

        # Inject df_result into method's local scope by adding it to kwargs
        # This is a workaround - the method should reference df_result
        original_df = self.df
        if not inplace:
            # Temporarily replace self.df with copy for the method
            self.df = df_result

        try:
            # Call the original method
            result = method(self, *args, **kwargs)

            # Handle return based on inplace
            if inplace:
                # Update self.df with the result and return self for chaining
                if result is not None and isinstance(result, pd.DataFrame):
                    self.df = result
                return self
            else:
                # Restore original df and return the result DataFrame
                self.df = original_df
                return result if result is not None else df_result

        except Exception as e:
            # Restore original df in case of error
            if not inplace:
                self.df = original_df
            raise e

    return wrapper
