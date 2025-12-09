import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union, Dict, Any
from datetime import datetime

from .base import FeatureEngineeringBase
from .utils import (
    validate_columns,
    get_numeric_columns,
    validate_numeric_columns,
    get_string_columns
)
from .exceptions import (
    ValidationError,
    InvalidStrategyError,
    InvalidMethodError,
    DataTypeError,
    EmptyDataFrameError,
)

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessor(FeatureEngineeringBase):
    """
    Data preprocessing class for cleaning and preparing data.

    Provides comprehensive preprocessing capabilities including:
    - Missing value handling (8 strategies)
    - Outlier detection and handling
    - String/text preprocessing and cleaning
    - Data quality validation and reporting
    - Duplicate and constant column removal
    - Type conversions and data filtering

    All transformation methods support inplace=False by default to avoid
    unintended mutations. Set inplace=True to modify the internal dataframe.

    New in v2.2.0:
        - String preprocessing methods (clean_string_columns, handle_whitespace_variants)
        - Data validation methods (validate_data_quality, detect_infinite_values)
        - Missing value indicators (create_missing_indicators)
        - Enhanced error handling and logging across all methods
    """

    # Class constants for thresholds
    DESTRUCTIVE_OPERATION_THRESHOLD = 0.3  # Warn if >30% of data will be removed
    HIGH_CARDINALITY_THRESHOLD = 0.95      # Threshold for high cardinality detection

    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor with a dataframe.

        Args:
            df: Input pandas DataFrame
        """
        super().__init__(df)
        self._operation_history: List[Dict[str, Any]] = []

    def _log_operation(self, method_name: str, parameters: Dict[str, Any],
                       rows_before: int, cols_before: int,
                       rows_after: int, cols_after: int,
                       additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a preprocessing operation to the history.

        Args:
            method_name: Name of the method that was called
            parameters: Dictionary of parameters passed to the method
            rows_before: Number of rows before operation
            cols_before: Number of columns before operation
            rows_after: Number of rows after operation
            cols_after: Number of columns after operation
            additional_info: Optional dictionary with method-specific info
        """
        operation = {
            'timestamp': datetime.now().isoformat(),
            'method': method_name,
            'parameters': parameters,
            'shape_before': (rows_before, cols_before),
            'shape_after': (rows_after, cols_after),
            'rows_changed': rows_after - rows_before,
            'cols_changed': cols_after - cols_before,
        }

        if additional_info:
            operation['details'] = additional_info

        self._operation_history.append(operation)
        logger.debug(f"Logged operation: {method_name}")

    def handle_missing_values(self, strategy: str = 'drop',
                               columns: Optional[List[str]] = None,
                               fill_value: Any = None,
                               method: Optional[str] = None,
                               inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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
            Modified DataFrame if inplace=False, else self for chaining

        Raises:
            ValueError: If invalid strategy or missing required parameters

        Examples:
            >>> preprocessor = DataPreprocessor(df)
            >>> # Fill numeric columns with mean
            >>> df_clean = preprocessor.handle_missing_values(strategy='mean')
            >>>
            >>> # Method chaining with inplace=True
            >>> preprocessor.handle_missing_values(strategy='mean', inplace=True)\
            ...            .remove_duplicates(inplace=True)
            >>>
            >>> # Fill specific columns with a value
            >>> df_clean = preprocessor.handle_missing_values(
            ...     strategy='fill_value',
            ...     columns=['age', 'income'],
            ...     fill_value=0
            ... )
        """
        valid_strategies = ['drop', 'fill_value', 'mean', 'median', 'mode',
                           'forward_fill', 'backward_fill', 'interpolate']
        if strategy not in valid_strategies:
            raise InvalidStrategyError(strategy, valid_strategies)

        # Capture shape before operation
        rows_before, cols_before = self.df.shape

        df_result = self.df if inplace else self.df.copy()

        if columns is None:
            columns = df_result.columns.tolist()
        else:
            columns = validate_columns(df_result, columns)

        if not columns:
            logger.warning("No valid columns to process")
            return df_result

        if strategy == 'drop':
            rows_before = len(df_result)
            df_result = df_result.dropna(subset=columns)
            rows_removed = rows_before - len(df_result)
            if rows_removed > rows_before * self.DESTRUCTIVE_OPERATION_THRESHOLD:
                logger.warning(
                    f"Dropping missing values removed {rows_removed} rows "
                    f"({rows_removed/rows_before*100:.1f}% of data). "
                    f"Consider using an imputation strategy instead."
                )
            elif rows_removed > 0:
                logger.info(f"Dropped {rows_removed} rows with missing values ({rows_removed/rows_before*100:.1f}%)")

        elif strategy == 'fill_value':
            if fill_value is None:
                raise ValueError("fill_value must be provided when using 'fill_value' strategy")
            df_result[columns] = df_result[columns].fillna(fill_value)

        elif strategy == 'mean':
            numeric_cols = get_numeric_columns(df_result, columns)
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for mean strategy")
            else:
                df_result[numeric_cols] = df_result[numeric_cols].fillna(df_result[numeric_cols].mean())

        elif strategy == 'median':
            numeric_cols = get_numeric_columns(df_result, columns)
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
            numeric_cols = get_numeric_columns(df_result, columns)
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for interpolate strategy")
            else:
                df_result[numeric_cols] = df_result[numeric_cols].interpolate(method=method or 'linear')

        if inplace:
            rows_after, cols_after = df_result.shape
            self._log_operation(
                method_name='handle_missing_values',
                parameters={'strategy': strategy, 'columns': columns, 'fill_value': fill_value, 'method': method},
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after
            )
            self.df = df_result
            return self
        return df_result

    def remove_duplicates(self, subset: Optional[List[str]] = None,
                          keep: str = 'first',
                          inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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

        # Capture shape before operation
        rows_before, cols_before = self.df.shape

        df_result = self.df if inplace else self.df.copy()

        if subset is not None:
            subset = validate_columns(df_result, subset)
            if not subset:
                subset = None

        rows_before_drop = len(df_result)
        df_result = df_result.drop_duplicates(subset=subset, keep=keep)
        rows_removed = rows_before_drop - len(df_result)

        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} duplicate rows ({rows_removed/rows_before_drop*100:.1f}%)")
        else:
            logger.info("No duplicate rows found")

        if inplace:
            rows_after, cols_after = df_result.shape
            self._log_operation(
                method_name='remove_duplicates',
                parameters={'subset': subset, 'keep': keep},
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after,
                additional_info={'rows_removed': rows_removed}
            )
            self.df = df_result
            return self
        return df_result

    def handle_outliers(self, columns: List[str],
                        method: str = 'iqr',
                        action: str = 'remove',
                        multiplier: float = 1.5,
                        threshold: float = 3.0,
                        replace_with: str = 'median',
                        inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        # Capture shape before operation
        rows_before, cols_before = self.df.shape

        df_result = self.df if inplace else self.df.copy()

        # Validate columns are numeric
        columns = validate_numeric_columns(df_result, columns)
        if not columns:
            logger.warning("No valid numeric columns to process")
            return df_result if not inplace else self

        for col in columns:

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

            # Log outlier detection
            outlier_count = outlier_mask.sum()
            outlier_pct = outlier_count / len(df_result) * 100
            logger.info(f"Detected {outlier_count} outliers in '{col}' ({outlier_pct:.1f}%) using {method} method")

            if action == 'remove':
                # Warn if removing too many rows
                if outlier_count > len(df_result) * self.DESTRUCTIVE_OPERATION_THRESHOLD:
                    logger.warning(f"Removing outliers from '{col}' would remove {outlier_pct:.1f}% of data. "
                                 f"Consider using action='cap' or 'replace' instead.")
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
            df_result = df_result.reset_index(drop=True)
            rows_after, cols_after = df_result.shape
            self._log_operation(
                method_name='handle_outliers',
                parameters={'columns': columns, 'method': method, 'action': action,
                           'multiplier': multiplier, 'threshold': threshold, 'replace_with': replace_with},
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after
            )
            self.df = df_result
            return self
        return df_result.reset_index(drop=True)

    def convert_dtypes(self, dtype_map: Dict[str, str], inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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

        # Validate columns exist
        valid_cols = validate_columns(df_result, list(dtype_map.keys()))

        for col in valid_cols:
            dtype = dtype_map[col]

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
            return self
        return df_result

    def clip_values(self, column: str, lower: Optional[float] = None,
                    upper: Optional[float] = None,
                    inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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

        # Validate column exists and is numeric
        valid_cols = validate_numeric_columns(df_result, [column])
        if not valid_cols:
            return df_result if not inplace else self

        # Validate lower < upper
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"lower bound ({lower}) must be less than upper bound ({upper})")

        df_result[column] = df_result[column].clip(lower=lower, upper=upper)
        logger.info(f"Clipped '{column}' to range [{lower}, {upper}]")

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self
        return df_result

    def remove_constant_columns(self, inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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
            return self
        return df_result

    def remove_high_cardinality_columns(self, threshold: float = 0.95,
                                         inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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
            return self
        return df_result

    def filter_rows(self, condition: Union[pd.Series, callable],
                    inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
        """
        Filter rows based on a condition.

        Args:
            condition: Boolean mask or callable that takes DataFrame and returns mask
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            Filtered DataFrame
        """
        # Capture shape before operation
        rows_before, cols_before = self.df.shape

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
            df_result = df_result.reset_index(drop=True)
            rows_after, cols_after = df_result.shape
            self._log_operation(
                method_name='filter_rows',
                parameters={'condition': 'custom_condition'},
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after
            )
            self.df = df_result
            return self
        return df_result.reset_index(drop=True)

    def drop_columns(self, columns: List[str], inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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

        # Capture shape before operation
        rows_before, cols_before = self.df.shape

        df_result = self.df if inplace else self.df.copy()

        # Validate columns exist
        existing_cols = validate_columns(df_result, columns)

        if existing_cols:
            df_result = df_result.drop(columns=existing_cols)

        if inplace:
            rows_after, cols_after = df_result.shape
            self._log_operation(
                method_name='drop_columns',
                parameters={'columns': columns},
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after,
                additional_info={'columns_dropped': existing_cols}
            )
            self.df = df_result
            return self
        return df_result

    def rename_columns(self, rename_map: Dict[str, str], inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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
            return self
        return df_result

    def reorder_columns(self, column_order: List[str], inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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

        # Validate columns exist
        column_order = validate_columns(df_result, column_order)

        other_cols = [col for col in df_result.columns if col not in column_order]
        final_order = column_order + other_cols

        df_result = df_result[final_order]

        if inplace:
            self.df = df_result
            return self
        return df_result

    def apply_custom_function(self, column: str, func: callable,
                               new_column: Optional[str] = None,
                               inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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

        # Validate column exists
        valid_cols = validate_columns(df_result, [column])
        if not valid_cols:
            return df_result if not inplace else self

        target_col = new_column if new_column else column
        try:
            df_result[target_col] = df_result[column].apply(func)
        except Exception as e:
            logger.error(f"Error applying function to column '{column}': {e}")
            raise

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self
        return df_result

    def reset_index_clean(self, inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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
            return self
        return df_result

    def sample_data(self, n: Optional[int] = None, frac: Optional[float] = None,
                    random_state: Optional[int] = None,
                    inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
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

        # Validate n <= len(df)
        if n is not None and n > len(df_result):
            raise ValidationError(f"Cannot sample {n} rows from DataFrame with {len(df_result)} rows. "
                                f"Use n <= {len(df_result)} or use frac parameter instead.")

        df_result = df_result.sample(n=n, frac=frac, random_state=random_state)
        logger.info(f"Sampled {len(df_result)} rows from DataFrame")

        if inplace:
            self.df = df_result.reset_index(drop=True)
            return self
        return df_result.reset_index(drop=True)

    # ==================== String/Text Preprocessing ====================

    def clean_string_columns(self, columns: List[str],
                             operations: List[str] = ['strip', 'lower'],
                             inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
        """
        Clean string columns with common operations.

        Args:
            columns: String columns to clean
            operations: List of operations to apply:
                - 'strip': Remove leading/trailing whitespace
                - 'lower': Convert to lowercase
                - 'upper': Convert to uppercase
                - 'title': Title case
                - 'remove_punctuation': Remove punctuation
                - 'remove_digits': Remove numeric characters
                - 'remove_extra_spaces': Collapse multiple spaces
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with cleaned strings

        Raises:
            TypeError: If columns or operations is not a list
            InvalidMethodError: If invalid operation specified

        Example:
            >>> preprocessor.clean_string_columns(['name', 'city'], ['strip', 'lower'])
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")
        if not isinstance(operations, list):
            raise TypeError("operations must be a list")

        valid_ops = ['strip', 'lower', 'upper', 'title',
                     'remove_punctuation', 'remove_digits', 'remove_extra_spaces']
        invalid_ops = [op for op in operations if op not in valid_ops]
        if invalid_ops:
            raise InvalidMethodError(f"Invalid operations: {invalid_ops}", valid_ops)

        # Capture shape before operation
        rows_before, cols_before = self.df.shape

        df_result = self.df if inplace else self.df.copy()

        # Validate columns are string type
        columns = get_string_columns(df_result, columns)
        if not columns:
            logger.warning("No valid string columns to process")
            return df_result if not inplace else self

        for col in columns:

            # Apply operations in order
            for op in operations:
                if op == 'strip':
                    df_result[col] = df_result[col].str.strip()
                elif op == 'lower':
                    df_result[col] = df_result[col].str.lower()
                elif op == 'upper':
                    df_result[col] = df_result[col].str.upper()
                elif op == 'title':
                    df_result[col] = df_result[col].str.title()
                elif op == 'remove_punctuation':
                    df_result[col] = df_result[col].str.replace(r'[^\w\s]', '', regex=True)
                elif op == 'remove_digits':
                    df_result[col] = df_result[col].str.replace(r'\d+', '', regex=True)
                elif op == 'remove_extra_spaces':
                    df_result[col] = df_result[col].str.replace(r'\s+', ' ', regex=True)

            logger.info(f"Applied {len(operations)} string operations to column '{col}'")

        if inplace:
            rows_after, cols_after = df_result.shape
            self._log_operation(
                method_name='clean_string_columns',
                parameters={'columns': columns, 'operations': operations},
                rows_before=rows_before,
                cols_before=cols_before,
                rows_after=rows_after,
                cols_after=cols_after
            )
            self.df = df_result
            return self
        return df_result

    def handle_whitespace_variants(self, columns: List[str],
                                    inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
        """
        Standardize whitespace variants in categorical columns.

        Handles cases like 'New York', ' New York', 'New York ', '  New York  '
        All become 'New York'

        Args:
            columns: Columns to standardize
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with standardized strings

        Example:
            >>> preprocessor.handle_whitespace_variants(['city', 'category'])
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        # Validate columns are string type
        columns = get_string_columns(df_result, columns)
        if not columns:
            logger.warning("No valid string columns to process")
            return df_result if not inplace else self

        for col in columns:

            before_unique = df_result[col].nunique()
            df_result[col] = df_result[col].str.strip()
            df_result[col] = df_result[col].str.replace(r'\s+', ' ', regex=True)
            after_unique = df_result[col].nunique()

            if before_unique != after_unique:
                logger.info(f"Standardized '{col}': {before_unique} -> {after_unique} unique values")

        if inplace:
            self.df = df_result
            return self
        return df_result

    def extract_string_length(self, columns: List[str],
                              suffix: str = '_length',
                              inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
        """
        Create length features from string columns.

        Args:
            columns: String columns to measure
            suffix: Suffix for length column names
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with added length columns

        Example:
            >>> preprocessor.extract_string_length(['name', 'description'])
            # Creates 'name_length' and 'description_length' columns
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        # Validate columns are string type
        columns = get_string_columns(df_result, columns)
        if not columns:
            logger.warning("No valid string columns to process")
            return df_result if not inplace else self

        for col in columns:

            new_col = f"{col}{suffix}"
            df_result[new_col] = df_result[col].str.len()
            logger.info(f"Created length feature '{new_col}'")

        if inplace:
            self.df = df_result
            return self
        return df_result

    # ==================== Data Validation Methods ====================

    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality validation report.

        Returns:
            Dictionary with validation results:
                - shape: DataFrame dimensions
                - missing_values: Columns with missing data
                - duplicate_rows: Count of duplicates
                - constant_columns: Columns with single value
                - high_cardinality_columns: Columns with >95% unique values
                - infinite_values: Columns with inf/-inf values
                - issues_found: List of detected issues

        Example:
            >>> quality = preprocessor.validate_data_quality()
            >>> print(quality['issues_found'])
        """
        validation = {
            'shape': self.df.shape,
            'missing_values': {},
            'duplicate_rows': 0,
            'constant_columns': [],
            'high_cardinality_columns': [],
            'infinite_values': {},
            'issues_found': []
        }

        # Missing values
        missing = self.df.isnull().sum()
        validation['missing_values'] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        if validation['missing_values']:
            validation['issues_found'].append(
                f"{len(validation['missing_values'])} columns have missing values"
            )

        # Duplicates
        validation['duplicate_rows'] = int(self.df.duplicated().sum())
        if validation['duplicate_rows'] > 0:
            validation['issues_found'].append(
                f"{validation['duplicate_rows']} duplicate rows found"
            )

        # Constant columns
        validation['constant_columns'] = [
            col for col in self.df.columns
            if self.df[col].nunique(dropna=False) <= 1
        ]
        if validation['constant_columns']:
            validation['issues_found'].append(
                f"{len(validation['constant_columns'])} constant columns found"
            )

        # High cardinality columns
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > self.HIGH_CARDINALITY_THRESHOLD and len(self.df) > 10:
                validation['high_cardinality_columns'].append(col)
        if validation['high_cardinality_columns']:
            validation['issues_found'].append(
                f"{len(validation['high_cardinality_columns'])} high cardinality columns (>95% unique)"
            )

        # Infinite values
        numeric_cols = get_numeric_columns(self.df)
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                validation['infinite_values'][col] = int(inf_count)
        if validation['infinite_values']:
            validation['issues_found'].append(
                f"{len(validation['infinite_values'])} columns have infinite values"
            )

        if not validation['issues_found']:
            validation['issues_found'].append("No major data quality issues detected")

        logger.info(f"Data quality validation complete: {len(validation['issues_found'])} findings")
        return validation

    def detect_infinite_values(self, columns: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Detect infinite values (np.inf, -np.inf) in numeric columns.

        Args:
            columns: Specific columns to check (None = all numeric columns)

        Returns:
            Dictionary mapping column names to count of infinite values

        Example:
            >>> inf_counts = preprocessor.detect_infinite_values()
            >>> print(inf_counts)
        """
        if columns is None:
            columns = get_numeric_columns(self.df)
        else:
            if not isinstance(columns, list):
                raise TypeError("columns must be a list")
            # Validate columns are numeric
            columns = validate_numeric_columns(self.df, columns)

        infinite_counts = {}
        for col in columns:

            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                infinite_counts[col] = int(inf_count)
                logger.info(f"Column '{col}' has {inf_count} infinite values")

        if not infinite_counts:
            logger.info("No infinite values detected")

        return infinite_counts

    def create_missing_indicators(self, columns: List[str],
                                   suffix: str = '_was_missing',
                                   inplace: bool = False) -> Union[pd.DataFrame, 'DataPreprocessor']:
        """
        Create binary indicator columns for missing values.

        Args:
            columns: Columns to create indicators for
            suffix: Suffix for indicator column names
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with added indicator columns (0/1)

        Example:
            >>> preprocessor.create_missing_indicators(['age', 'income'])
            # Creates 'age_was_missing' and 'income_was_missing' columns
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        # Validate columns exist
        columns = validate_columns(df_result, columns)
        if not columns:
            logger.warning("No valid columns to process")
            return df_result if not inplace else self

        for col in columns:

            new_col = f"{col}{suffix}"
            df_result[new_col] = self.df[col].isnull().astype(int)

            missing_count = df_result[new_col].sum()
            if missing_count > 0:
                logger.info(f"Created indicator '{new_col}' ({missing_count} missing values)")
            else:
                logger.info(f"Created indicator '{new_col}' (no missing values)")

        if inplace:
            self.df = df_result
            return self
        return df_result

    # ==================== Operation History & Summary ====================

    def get_preprocessing_summary(self) -> str:
        """
        Get a formatted summary of all preprocessing operations.

        Returns:
            Formatted string summary of preprocessing history

        Example:
            >>> summary = preprocessor.get_preprocessing_summary()
            >>> print(summary)
        """
        if not self._operation_history:
            return "No preprocessing operations have been performed yet."

        lines = ["=" * 80, "PREPROCESSING SUMMARY", "=" * 80, ""]

        for i, op in enumerate(self._operation_history, 1):
            lines.append(f"{i}. {op['method'].upper()}")
            lines.append(f"   Timestamp: {op['timestamp']}")
            lines.append(f"   Shape: {op['shape_before']} → {op['shape_after']}")
            lines.append(f"   Rows changed: {op['rows_changed']:+d}, Columns changed: {op['cols_changed']:+d}")

            # Add parameters
            if op['parameters']:
                lines.append(f"   Parameters:")
                for key, value in op['parameters'].items():
                    if value is not None:
                        # Truncate long lists for readability
                        if isinstance(value, list) and len(value) > 5:
                            value_str = f"{value[:5]}... ({len(value)} total)"
                        else:
                            value_str = str(value)
                        lines.append(f"      - {key}: {value_str}")

            # Add additional details if present
            if 'details' in op:
                lines.append(f"   Details:")
                for key, value in op['details'].items():
                    lines.append(f"      - {key}: {value}")

            lines.append("")  # Blank line between operations

        # Summary statistics
        total_ops = len(self._operation_history)
        initial_shape = self._operation_history[0]['shape_before']
        final_shape = self._operation_history[-1]['shape_after']
        total_rows_changed = final_shape[0] - initial_shape[0]
        total_cols_changed = final_shape[1] - initial_shape[1]

        lines.append("=" * 80)
        lines.append(f"TOTAL OPERATIONS: {total_ops}")
        lines.append(f"Initial shape: {initial_shape}")
        lines.append(f"Final shape: {final_shape}")
        lines.append(f"Total rows changed: {total_rows_changed:+d}")
        lines.append(f"Total columns changed: {total_cols_changed:+d}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_summary(self, filepath: str, format: str = 'markdown') -> None:
        """
        Export preprocessing summary to a file.

        Args:
            filepath: Path to save the summary file
            format: Export format ('text', 'markdown', or 'json')

        Raises:
            ValueError: If invalid format specified
            ValidationError: If no operations to export

        Example:
            >>> preprocessor.export_summary('preprocessing_report.md', format='markdown')
        """
        valid_formats = ['text', 'markdown', 'json']
        if format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}, got '{format}'")

        if not self._operation_history:
            raise ValidationError("No preprocessing operations to export")

        if format == 'json':
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'operations': self._operation_history,
                    'summary': {
                        'total_operations': len(self._operation_history),
                        'initial_shape': self._operation_history[0]['shape_before'],
                        'final_shape': self._operation_history[-1]['shape_after'],
                    }
                }, f, indent=2)
            logger.info(f"Exported preprocessing summary to {filepath} (JSON format)")

        elif format == 'markdown':
            lines = ["# Preprocessing Summary\n"]
            lines.append(f"**Total Operations:** {len(self._operation_history)}\n")
            lines.append(f"**Initial Shape:** {self._operation_history[0]['shape_before']}\n")
            lines.append(f"**Final Shape:** {self._operation_history[-1]['shape_after']}\n")
            lines.append("\n## Operations\n")

            for i, op in enumerate(self._operation_history, 1):
                lines.append(f"### {i}. {op['method']}\n")
                lines.append(f"- **Timestamp:** {op['timestamp']}")
                lines.append(f"- **Shape:** {op['shape_before']} → {op['shape_after']}")
                lines.append(f"- **Rows changed:** {op['rows_changed']:+d}")
                lines.append(f"- **Columns changed:** {op['cols_changed']:+d}")

                if op['parameters']:
                    lines.append(f"- **Parameters:**")
                    for key, value in op['parameters'].items():
                        if value is not None:
                            lines.append(f"  - `{key}`: {value}")

                if 'details' in op:
                    lines.append(f"- **Details:**")
                    for key, value in op['details'].items():
                        lines.append(f"  - {key}: {value}")

                lines.append("")  # Blank line

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            logger.info(f"Exported preprocessing summary to {filepath} (Markdown format)")

        else:  # text format
            summary_text = self.get_preprocessing_summary()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            logger.info(f"Exported preprocessing summary to {filepath} (text format)")
