import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any


class DataPreprocessor:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_missing_values(self, strategy: str = 'drop',
                               columns: Optional[List[str]] = None,
                               fill_value: Any = None,
                               method: Optional[str] = None,
                               inplace: bool = True) -> pd.DataFrame:
        """Handle missing values using various strategies."""
        df_result = self.df if inplace else self.df.copy()

        if columns is None:
            columns = df_result.columns.tolist()

        if strategy == 'drop':
            df_result = df_result.dropna(subset=columns)

        elif strategy == 'fill_value':
            if fill_value is None:
                raise ValueError("fill_value must be provided when using 'fill_value' strategy")
            df_result[columns] = df_result[columns].fillna(fill_value)

        elif strategy == 'mean':
            numeric_cols = df_result[columns].select_dtypes(include=[np.number]).columns
            df_result[numeric_cols] = df_result[numeric_cols].fillna(df_result[numeric_cols].mean())

        elif strategy == 'median':
            numeric_cols = df_result[columns].select_dtypes(include=[np.number]).columns
            df_result[numeric_cols] = df_result[numeric_cols].fillna(df_result[numeric_cols].median())

        elif strategy == 'mode':
            for col in columns:
                mode_val = df_result[col].mode()
                if len(mode_val) > 0:
                    df_result[col] = df_result[col].fillna(mode_val[0])

        elif strategy == 'forward_fill':
            df_result[columns] = df_result[columns].fillna(method='ffill')

        elif strategy == 'backward_fill':
            df_result[columns] = df_result[columns].fillna(method='bfill')

        elif strategy == 'interpolate':
            numeric_cols = df_result[columns].select_dtypes(include=[np.number]).columns
            df_result[numeric_cols] = df_result[numeric_cols].interpolate(method=method or 'linear')

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def remove_duplicates(self, subset: Optional[List[str]] = None,
                          keep: str = 'first',
                          inplace: bool = True) -> pd.DataFrame:
        """Remove duplicate rows."""
        df_result = self.df if inplace else self.df.copy()

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
                        inplace: bool = True) -> pd.DataFrame:
        """Handle outliers using various methods."""
        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found in dataframe.")
                continue

            if method == 'iqr':
                Q1 = df_result[col].quantile(0.25)
                Q3 = df_result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outlier_mask = (df_result[col] < lower_bound) | (df_result[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df_result[col] - df_result[col].mean()) / df_result[col].std())
                outlier_mask = z_scores > threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            if action == 'remove':
                df_result = df_result[~outlier_mask]

            elif action == 'cap':
                if method == 'iqr':
                    df_result.loc[df_result[col] < lower_bound, col] = lower_bound
                    df_result.loc[df_result[col] > upper_bound, col] = upper_bound

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

    def convert_dtypes(self, dtype_map: Dict[str, str], inplace: bool = True) -> pd.DataFrame:
        """Convert column data types."""
        df_result = self.df if inplace else self.df.copy()

        for col, dtype in dtype_map.items():
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found in dataframe.")
                continue

            try:
                if dtype == 'datetime':
                    df_result[col] = pd.to_datetime(df_result[col])
                elif dtype == 'category':
                    df_result[col] = df_result[col].astype('category')
                else:
                    df_result[col] = df_result[col].astype(dtype)
            except Exception as e:
                print(f"Error converting column '{col}' to {dtype}: {e}")

        if not inplace:
            return df_result
        return self.df

    def clip_values(self, column: str, lower: Optional[float] = None,
                    upper: Optional[float] = None,
                    inplace: bool = True) -> pd.DataFrame:
        """Clip values in a column to specified range."""
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            print(f"Warning: Column '{column}' not found in dataframe.")
            return df_result

        df_result[column] = df_result[column].clip(lower=lower, upper=upper)

        if not inplace:
            return df_result
        return self.df

    def remove_constant_columns(self, inplace: bool = True) -> pd.DataFrame:
        """Remove columns with constant values."""
        df_result = self.df if inplace else self.df.copy()

        constant_cols = [col for col in df_result.columns if df_result[col].nunique() <= 1]

        if constant_cols:
            print(f"Removing constant columns: {constant_cols}")
            df_result = df_result.drop(columns=constant_cols)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def remove_high_cardinality_columns(self, threshold: float = 0.95,
                                         inplace: bool = True) -> pd.DataFrame:
        """Remove columns with very high cardinality."""
        df_result = self.df if inplace else self.df.copy()

        high_card_cols = []
        for col in df_result.columns:
            cardinality_ratio = df_result[col].nunique() / len(df_result)
            if cardinality_ratio >= threshold:
                high_card_cols.append(col)

        if high_card_cols:
            print(f"Removing high cardinality columns: {high_card_cols}")
            df_result = df_result.drop(columns=high_card_cols)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def filter_rows(self, condition: Union[pd.Series, callable],
                    inplace: bool = True) -> pd.DataFrame:
        """Filter rows based on a condition."""
        df_result = self.df if inplace else self.df.copy()

        if callable(condition):
            mask = condition(df_result)
        else:
            mask = condition

        df_result = df_result[mask]

        if inplace:
            self.df = df_result.reset_index(drop=True)
            return self.df
        return df_result.reset_index(drop=True)

    def drop_columns(self, columns: List[str], inplace: bool = True) -> pd.DataFrame:
        """Drop specified columns."""
        df_result = self.df if inplace else self.df.copy()

        existing_cols = [col for col in columns if col in df_result.columns]
        if existing_cols:
            df_result = df_result.drop(columns=existing_cols)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def rename_columns(self, rename_map: Dict[str, str], inplace: bool = True) -> pd.DataFrame:
        """Rename columns."""
        df_result = self.df if inplace else self.df.copy()

        df_result = df_result.rename(columns=rename_map)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def reorder_columns(self, column_order: List[str], inplace: bool = True) -> pd.DataFrame:
        """Reorder columns."""
        df_result = self.df if inplace else self.df.copy()

        missing_cols = [col for col in column_order if col not in df_result.columns]
        if missing_cols:
            print(f"Warning: Columns not found in dataframe: {missing_cols}")
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
                               inplace: bool = True) -> pd.DataFrame:
        """Apply a custom function to a column."""
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            print(f"Warning: Column '{column}' not found in dataframe.")
            return df_result

        target_col = new_column if new_column else column
        df_result[target_col] = df_result[column].apply(func)

        if not inplace:
            return df_result
        return self.df

    def reset_index_clean(self, inplace: bool = True) -> pd.DataFrame:
        """Reset index and drop the old index."""
        df_result = self.df if inplace else self.df.copy()

        df_result = df_result.reset_index(drop=True)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def sample_data(self, n: Optional[int] = None, frac: Optional[float] = None,
                    random_state: Optional[int] = None,
                    inplace: bool = True) -> pd.DataFrame:
        """Sample data from the dataframe."""
        df_result = self.df if inplace else self.df.copy()

        df_result = df_result.sample(n=n, frac=frac, random_state=random_state)

        if inplace:
            self.df = df_result.reset_index(drop=True)
            return self.df
        return df_result.reset_index(drop=True)

    def get_dataframe(self) -> pd.DataFrame:
        """Return the current dataframe."""
        return self.df.copy()
