import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union, Dict, Any

from .exceptions import (
    ValidationError,
    InvalidStrategyError,
    InvalidMethodError,
    DataTypeError,
    EmptyDataFrameError,
)

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing class for cleaning and preparing data.

    All transformation methods support inplace=False by default to avoid
    unintended mutations. Set inplace=True to modify the internal dataframe.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor with a dataframe.

        Args:
            df: Input pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            logger.warning("Initializing with empty DataFrame")
        self.df = df.copy()

    def handle_missing_values(self, strategy: str = 'drop',
                               columns: Optional[List[str]] = None,
                               fill_value: Any = None,
                               method: Optional[str] = None,
                               inplace: bool = False) -> pd.DataFrame:
        """
        Handle missing values using various strategies.

        Args:
            strategy: Strategy to use ('drop', 'fill_value', 'mean', 'median',
                     'mode', 'forward_fill', 'backward_fill', 'interpolate')
            columns: List of columns to process. If None, processes all columns.
            fill_value: Value to fill when strategy='fill_value'
            method: Interpolation method when strategy='interpolate'
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            Modified DataFrame

        Raises:
            ValueError: If invalid strategy or missing required parameters
        """
        valid_strategies = ['drop', 'fill_value', 'mean', 'median', 'mode',
                           'forward_fill', 'backward_fill', 'interpolate']
        if strategy not in valid_strategies:
            raise InvalidStrategyError(strategy, valid_strategies)

        df_result = self.df if inplace else self.df.copy()

        if columns is None:
            columns = df_result.columns.tolist()

        # Validate columns exist
        invalid_cols = [col for col in columns if col not in df_result.columns]
        if invalid_cols:
            logger.warning(f"Columns not found in dataframe: {invalid_cols}")
            columns = [col for col in columns if col in df_result.columns]

        if not columns:
            logger.warning("No valid columns to process")
            return df_result

        if strategy == 'drop':
            df_result = df_result.dropna(subset=columns)

        elif strategy == 'fill_value':
            if fill_value is None:
                raise ValueError("fill_value must be provided when using 'fill_value' strategy")
            df_result[columns] = df_result[columns].fillna(fill_value)

        elif strategy == 'mean':
            numeric_cols = df_result[columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for mean strategy")
            else:
                df_result[numeric_cols] = df_result[numeric_cols].fillna(df_result[numeric_cols].mean())

        elif strategy == 'median':
            numeric_cols = df_result[columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for median strategy")
            else:
                df_result[numeric_cols] = df_result[numeric_cols].fillna(df_result[numeric_cols].median())

        elif strategy == 'mode':
            for col in columns:
                mode_val = df_result[col].mode()
                if len(mode_val) > 0:
                    df_result[col] = df_result[col].fillna(mode_val[0])

        elif strategy == 'forward_fill':
            # Fixed: Use ffill() instead of deprecated fillna(method='ffill')
            df_result[columns] = df_result[columns].ffill()

        elif strategy == 'backward_fill':
            # Fixed: Use bfill() instead of deprecated fillna(method='bfill')
            df_result[columns] = df_result[columns].bfill()

        elif strategy == 'interpolate':
            numeric_cols = df_result[columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for interpolate strategy")
            else:
                df_result[numeric_cols] = df_result[numeric_cols].interpolate(method=method or 'linear')

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def remove_duplicates(self, subset: Optional[List[str]] = None,
                          keep: str = 'first',
                          inplace: bool = False) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            subset: Columns to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame without duplicates
        """
        if keep not in ['first', 'last', False]:
            raise ValueError("keep must be 'first', 'last', or False")

        df_result = self.df if inplace else self.df.copy()

        if subset is not None:
            invalid_cols = [col for col in subset if col not in df_result.columns]
            if invalid_cols:
                logger.warning(f"Subset columns not found: {invalid_cols}")
                subset = [col for col in subset if col in df_result.columns] or None

        df_result = df_result.drop_duplicates(subset=subset, keep=keep)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def handle_outliers(self, columns: List[str],
                        method: str = 'iqr',
                        action: str = 'remove',
                        multiplier: float = 1.5,
                        threshold: float = 3.0,
                        replace_with: str = 'median',
                        inplace: bool = False) -> pd.DataFrame:
        """
        Handle outliers using various methods.

        Args:
            columns: List of columns to process
            method: Detection method ('iqr' or 'zscore')
            action: Action to take ('remove', 'cap', or 'replace')
            multiplier: IQR multiplier for IQR method
            threshold: Z-score threshold for zscore method
            replace_with: Value to use when action='replace' ('median', 'mean', or 'nan')
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with outliers handled

        Raises:
            ValueError: If invalid method, action, or replace_with value
        """
        if method not in ['iqr', 'zscore']:
            raise InvalidMethodError(method, ['iqr', 'zscore'])
        if action not in ['remove', 'cap', 'replace']:
            raise InvalidMethodError(action, ['remove', 'cap', 'replace'])
        if replace_with not in ['median', 'mean', 'nan']:
            raise ValidationError(f"replace_with must be 'median', 'mean', or 'nan', got '{replace_with}'")
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue

            # Ensure column is numeric
            if not np.issubdtype(df_result[col].dtype, np.number):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue

            if method == 'iqr':
                Q1 = df_result[col].quantile(0.25)
                Q3 = df_result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outlier_mask = (df_result[col] < lower_bound) | (df_result[col] > upper_bound)

            elif method == 'zscore':
                # Fixed: Add division by zero check
                col_std = df_result[col].std()
                if col_std == 0:
                    logger.warning(f"Column '{col}' has zero standard deviation, skipping")
                    continue
                z_scores = np.abs((df_result[col] - df_result[col].mean()) / col_std)
                outlier_mask = z_scores > threshold

            if action == 'remove':
                df_result = df_result[~outlier_mask]

            elif action == 'cap':
                if method == 'iqr':
                    df_result.loc[df_result[col] < lower_bound, col] = lower_bound
                    df_result.loc[df_result[col] > upper_bound, col] = upper_bound
                else:
                    logger.warning("Capping is only supported for IQR method")

            elif action == 'replace':
                if replace_with == 'median':
                    df_result.loc[outlier_mask, col] = df_result[col].median()
                elif replace_with == 'mean':
                    df_result.loc[outlier_mask, col] = df_result[col].mean()
                elif replace_with == 'nan':
                    df_result.loc[outlier_mask, col] = np.nan

        if inplace:
            self.df = df_result.reset_index(drop=True)
            return self.df
        return df_result.reset_index(drop=True)

    def convert_dtypes(self, dtype_map: Dict[str, str], inplace: bool = False) -> pd.DataFrame:
        """
        Convert column data types.

        Args:
            dtype_map: Dictionary mapping column names to target dtypes
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with converted dtypes
        """
        if not isinstance(dtype_map, dict):
            raise TypeError("dtype_map must be a dictionary")

        df_result = self.df if inplace else self.df.copy()

        for col, dtype in dtype_map.items():
            if col not in df_result.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue

            try:
                if dtype == 'datetime':
                    df_result[col] = pd.to_datetime(df_result[col])
                elif dtype == 'category':
                    df_result[col] = df_result[col].astype('category')
                else:
                    df_result[col] = df_result[col].astype(dtype)
            except Exception as e:
                logger.error(f"Error converting column '{col}' to {dtype}: {e}")

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def clip_values(self, column: str, lower: Optional[float] = None,
                    upper: Optional[float] = None,
                    inplace: bool = False) -> pd.DataFrame:
        """
        Clip values in a column to specified range.

        Args:
            column: Column name to clip
            lower: Lower bound (None for no lower bound)
            upper: Upper bound (None for no upper bound)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with clipped values
        """
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df_result if not inplace else self.df

        if not np.issubdtype(df_result[column].dtype, np.number):
            logger.warning(f"Column '{column}' is not numeric")
            return df_result if not inplace else self.df

        df_result[column] = df_result[column].clip(lower=lower, upper=upper)

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def remove_constant_columns(self, inplace: bool = False) -> pd.DataFrame:
        """
        Remove columns with constant values.

        Args:
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame without constant columns
        """
        df_result = self.df if inplace else self.df.copy()

        constant_cols = [col for col in df_result.columns if df_result[col].nunique() <= 1]

        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            df_result = df_result.drop(columns=constant_cols)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def remove_high_cardinality_columns(self, threshold: float = 0.95,
                                         inplace: bool = False) -> pd.DataFrame:
        """
        Remove columns with very high cardinality.

        Args:
            threshold: Cardinality ratio threshold (unique/total)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame without high cardinality columns

        Raises:
            ValueError: If threshold not in (0, 1]
        """
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in range (0, 1]")

        df_result = self.df if inplace else self.df.copy()

        high_card_cols = []
        for col in df_result.columns:
            if len(df_result) > 0:  # Avoid division by zero
                cardinality_ratio = df_result[col].nunique() / len(df_result)
                if cardinality_ratio >= threshold:
                    high_card_cols.append(col)

        if high_card_cols:
            logger.info(f"Removing {len(high_card_cols)} high cardinality columns: {high_card_cols}")
            df_result = df_result.drop(columns=high_card_cols)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def filter_rows(self, condition: Union[pd.Series, callable],
                    inplace: bool = False) -> pd.DataFrame:
        """
        Filter rows based on a condition.

        Args:
            condition: Boolean mask or callable that takes DataFrame and returns mask
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            Filtered DataFrame
        """
        df_result = self.df if inplace else self.df.copy()

        try:
            if callable(condition):
                mask = condition(df_result)
            else:
                mask = condition

            if not isinstance(mask, pd.Series):
                raise TypeError("Condition must evaluate to a boolean Series")

            df_result = df_result[mask]
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            raise

        if inplace:
            self.df = df_result.reset_index(drop=True)
            return self.df
        return df_result.reset_index(drop=True)

    def drop_columns(self, columns: List[str], inplace: bool = False) -> pd.DataFrame:
        """
        Drop specified columns.

        Args:
            columns: List of column names to drop
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame without specified columns
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        existing_cols = [col for col in columns if col in df_result.columns]
        if len(existing_cols) < len(columns):
            missing = set(columns) - set(existing_cols)
            logger.warning(f"Columns not found: {missing}")

        if existing_cols:
            df_result = df_result.drop(columns=existing_cols)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def rename_columns(self, rename_map: Dict[str, str], inplace: bool = False) -> pd.DataFrame:
        """
        Rename columns.

        Args:
            rename_map: Dictionary mapping old names to new names
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with renamed columns
        """
        if not isinstance(rename_map, dict):
            raise TypeError("rename_map must be a dictionary")

        df_result = self.df if inplace else self.df.copy()

        df_result = df_result.rename(columns=rename_map)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def reorder_columns(self, column_order: List[str], inplace: bool = False) -> pd.DataFrame:
        """
        Reorder columns.

        Args:
            column_order: Desired column order (columns not listed will be appended)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with reordered columns
        """
        if not isinstance(column_order, list):
            raise TypeError("column_order must be a list")

        df_result = self.df if inplace else self.df.copy()

        missing_cols = [col for col in column_order if col not in df_result.columns]
        if missing_cols:
            logger.warning(f"Columns not found in dataframe: {missing_cols}")
            column_order = [col for col in column_order if col in df_result.columns]

        other_cols = [col for col in df_result.columns if col not in column_order]
        final_order = column_order + other_cols

        df_result = df_result[final_order]

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def apply_custom_function(self, column: str, func: callable,
                               new_column: Optional[str] = None,
                               inplace: bool = False) -> pd.DataFrame:
        """
        Apply a custom function to a column.

        Args:
            column: Column to apply function to
            func: Function to apply (must be callable)
            new_column: Name for new column (None to replace original)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with function applied

        Raises:
            TypeError: If func is not callable
        """
        if not callable(func):
            raise TypeError("func must be callable")

        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df_result if not inplace else self.df

        target_col = new_column if new_column else column
        try:
            df_result[target_col] = df_result[column].apply(func)
        except Exception as e:
            logger.error(f"Error applying function to column '{column}': {e}")
            raise

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def reset_index_clean(self, inplace: bool = False) -> pd.DataFrame:
        """
        Reset index and drop the old index.

        Args:
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with reset index
        """
        df_result = self.df if inplace else self.df.copy()

        df_result = df_result.reset_index(drop=True)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def sample_data(self, n: Optional[int] = None, frac: Optional[float] = None,
                    random_state: Optional[int] = None,
                    inplace: bool = False) -> pd.DataFrame:
        """
        Sample data from the dataframe.

        Args:
            n: Number of samples (mutually exclusive with frac)
            frac: Fraction of samples (mutually exclusive with n)
            random_state: Random seed for reproducibility
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            Sampled DataFrame

        Raises:
            ValueError: If both n and frac are specified
        """
        if n is not None and frac is not None:
            raise ValueError("Cannot specify both n and frac")
        if n is not None and n <= 0:
            raise ValueError("n must be positive")
        if frac is not None and not 0 < frac <= 1:
            raise ValueError("frac must be in range (0, 1]")

        df_result = self.df if inplace else self.df.copy()

        df_result = df_result.sample(n=n, frac=frac, random_state=random_state)

        if inplace:
            self.df = df_result.reset_index(drop=True)
            return self.df
        return df_result.reset_index(drop=True)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the current dataframe.

        Returns:
            Copy of internal DataFrame
        """
        return self.df.copy()
