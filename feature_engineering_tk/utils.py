"""
Utility functions for feature engineering toolkit.

This module provides shared validation and helper functions used across
all feature engineering toolkit classes.
"""

import logging
from typing import List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_and_copy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and create a copy of a DataFrame.

    Args:
        df: Input DataFrame to validate and copy

    Returns:
        Copy of the validated DataFrame

    Raises:
        TypeError: If input is not a pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        logger.warning("Initializing with empty DataFrame")

    return df.copy()


def validate_columns(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    raise_on_missing: bool = False
) -> List[str]:
    """
    Validate that columns exist in DataFrame and return valid column list.

    Args:
        df: DataFrame to validate against
        columns: Single column name or list of column names to validate
        raise_on_missing: If True, raise error for missing columns.
                         If False, log warning and return only valid columns.

    Returns:
        List of valid column names that exist in the DataFrame

    Raises:
        ValueError: If raise_on_missing=True and columns are missing
        TypeError: If columns is not str or list
    """
    # Normalize to list
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        raise TypeError("columns must be a string or list of strings")

    # Find invalid columns
    invalid_cols = [col for col in columns if col not in df.columns]

    if invalid_cols:
        msg = f"Columns not found in DataFrame: {invalid_cols}"
        if raise_on_missing:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # Return only valid columns
    valid_cols = [col for col in columns if col in df.columns]

    if not valid_cols:
        logger.warning("No valid columns found after validation")

    return valid_cols


def get_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> List[str]:
    """
    Get numeric columns from DataFrame.

    Args:
        df: DataFrame to extract numeric columns from
        columns: Optional subset of columns to filter. If None, checks all columns.

    Returns:
        List of column names with numeric dtypes
    """
    if columns is None:
        # Get all numeric columns
        return df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Get numeric columns from specified subset
        subset_df = df[columns]
        return subset_df.select_dtypes(include=[np.number]).columns.tolist()


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: List[str]
) -> List[str]:
    """
    Validate that columns are numeric and return only valid numeric columns.

    Args:
        df: DataFrame containing the columns
        columns: List of column names to validate as numeric

    Returns:
        List of column names that are numeric
    """
    numeric_cols = []

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue

        if not np.issubdtype(df[col].dtype, np.number):
            logger.warning(f"Column '{col}' is not numeric type, skipping")
            continue

        numeric_cols.append(col)

    if not numeric_cols:
        logger.warning("No valid numeric columns found")

    return numeric_cols


def get_string_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> List[str]:
    """
    Get string/object columns from DataFrame.

    Args:
        df: DataFrame to extract string columns from
        columns: Optional subset of columns to filter. If None, checks all columns.

    Returns:
        List of column names with object/string dtypes
    """
    if columns is None:
        # Get all string columns
        return df.select_dtypes(include=['object']).columns.tolist()
    else:
        # Optimized validation using sets
        columns_set = set(df.columns)
        valid_cols = [col for col in columns if col in columns_set]
        missing_cols = set(columns) - columns_set

        if missing_cols:
            logger.warning(f"Columns not found in DataFrame: {missing_cols}")

        if not valid_cols:
            return []

        # Check dtypes without copying DataFrame
        string_cols = [col for col in valid_cols if df[col].dtype == 'object']
        non_string = set(valid_cols) - set(string_cols)

        if non_string:
            logger.warning(f"Non-string columns skipped: {non_string}")

        return string_cols


def get_feature_columns(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    numeric_only: bool = True
) -> List[str]:
    """
    Get feature columns with optional exclusions.

    Useful for filtering out target columns or other non-feature columns.

    Args:
        df: DataFrame to extract feature columns from
        exclude_columns: List of column names to exclude (e.g., target column)
        numeric_only: If True, return only numeric columns. If False, return all columns.

    Returns:
        List of feature column names after applying filters
    """
    if exclude_columns is None:
        exclude_columns = []

    # Select columns based on type
    if numeric_only:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = df.columns.tolist()

    # Filter out excluded columns
    feature_cols = [col for col in cols if col not in exclude_columns]

    return feature_cols
