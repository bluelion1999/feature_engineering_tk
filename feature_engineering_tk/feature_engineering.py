import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, OrdinalEncoder
)
from typing import List, Optional, Dict, Union, Any

from .exceptions import (
    ValidationError,
    InvalidMethodError,
    TransformerNotFittedError,
)

# Configure logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for transforming and creating features.

    All transformation methods support inplace=False by default to avoid
    unintended mutations. Set inplace=True to modify the internal dataframe.

    Fitted encoders and scalers are stored and can be exported for production use.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer with a dataframe.

        Args:
            df: Input pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            logger.warning("Initializing with empty DataFrame")
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}

    def encode_categorical_label(self, columns: List[str], inplace: bool = False) -> pd.DataFrame:
        """
        Encode categorical columns using label encoding.

        Args:
            columns: List of columns to encode
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with label-encoded columns
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue

            encoder = LabelEncoder()
            df_result[col] = encoder.fit_transform(df_result[col].astype(str))
            self.encoders[f"{col}_label"] = encoder
            logger.info(f"Label encoded column '{col}' with {len(encoder.classes_)} unique values")

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def encode_categorical_onehot(self, columns: List[str], drop_first: bool = False,
                                   prefix: Optional[Dict[str, str]] = None,
                                   inplace: bool = False) -> pd.DataFrame:
        """
        Encode categorical columns using one-hot encoding.

        Args:
            columns: List of columns to encode
            drop_first: Drop first category to avoid multicollinearity
            prefix: Dictionary mapping column names to prefixes for encoded columns
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with one-hot encoded columns
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        if prefix is None:
            prefix = {col: col for col in columns}

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue

            unique_count = df_result[col].nunique()
            if unique_count > 100:
                logger.warning(f"Column '{col}' has {unique_count} unique values. "
                             "Consider using other encoding methods for high cardinality.")

            dummies = pd.get_dummies(
                df_result[col],
                prefix=prefix.get(col, col),
                drop_first=drop_first,
                dtype=int
            )

            df_result = pd.concat([df_result, dummies], axis=1)
            df_result = df_result.drop(columns=[col])
            logger.info(f"One-hot encoded column '{col}' into {len(dummies.columns)} columns")

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def encode_categorical_ordinal(self, column: str, categories: List[Any],
                                    inplace: bool = False) -> pd.DataFrame:
        """
        Encode categorical column with ordinal relationship.

        Args:
            column: Column name to encode
            categories: Ordered list of categories
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with ordinally encoded column
        """
        if not isinstance(categories, list):
            raise TypeError("categories must be a list")
        if len(categories) == 0:
            raise ValueError("categories cannot be empty")

        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df_result if not inplace else self.df

        encoder = OrdinalEncoder(
            categories=[categories],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        df_result[column] = encoder.fit_transform(df_result[[column]])
        self.encoders[f"{column}_ordinal"] = encoder
        logger.info(f"Ordinal encoded column '{column}' with {len(categories)} categories")

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def scale_features(self, columns: List[str], method: str = 'standard',
                       inplace: bool = False) -> pd.DataFrame:
        """
        Scale numeric features using specified method.

        Args:
            columns: List of columns to scale
            method: Scaling method ('standard', 'minmax', or 'robust')
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with scaled features

        Raises:
            ValueError: If invalid method specified
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        scalers_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        if method not in scalers_map:
            raise InvalidMethodError(method, list(scalers_map.keys()))

        df_result = self.df if inplace else self.df.copy()

        scaler = scalers_map[method]

        valid_cols = [col for col in columns if col in df_result.columns]
        if not valid_cols:
            logger.warning("None of the specified columns found in dataframe")
            return df_result if not inplace else self.df

        # Validate columns are numeric
        non_numeric = [col for col in valid_cols
                      if not np.issubdtype(df_result[col].dtype, np.number)]
        if non_numeric:
            logger.warning(f"Non-numeric columns will be skipped: {non_numeric}")
            valid_cols = [col for col in valid_cols if col not in non_numeric]

        if not valid_cols:
            logger.warning("No numeric columns to scale")
            return df_result if not inplace else self.df

        df_result[valid_cols] = scaler.fit_transform(df_result[valid_cols])
        self.scalers[f"{method}_scaler"] = scaler
        logger.info(f"Scaled {len(valid_cols)} columns using {method} method")

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_polynomial_features(self, columns: List[str], degree: int = 2,
                                    interaction_only: bool = False,
                                    inplace: bool = False) -> pd.DataFrame:
        """
        Create polynomial and interaction features.

        Args:
            columns: List of columns to create polynomial features from
            degree: Polynomial degree (2 or 3)
            interaction_only: If True, only create interaction terms (no powers)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with polynomial features

        Raises:
            ValueError: If degree not in (2, 3) or invalid parameters
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")
        if degree not in [2, 3]:
            raise ValueError("degree must be 2 or 3")

        df_result = self.df if inplace else self.df.copy()

        valid_cols = [col for col in columns if col in df_result.columns]
        if not valid_cols:
            logger.warning("None of the specified columns found in dataframe")
            return df_result if not inplace else self.df

        # Validate numeric
        non_numeric = [col for col in valid_cols
                      if not np.issubdtype(df_result[col].dtype, np.number)]
        if non_numeric:
            logger.warning(f"Non-numeric columns will be skipped: {non_numeric}")
            valid_cols = [col for col in valid_cols if col not in non_numeric]

        if not valid_cols:
            return df_result if not inplace else self.df

        features_created = 0

        if degree == 2:
            if not interaction_only:
                for col in valid_cols:
                    df_result[f"{col}_squared"] = df_result[col] ** 2
                    features_created += 1

            for i, col1 in enumerate(valid_cols):
                for col2 in valid_cols[i + 1:]:
                    df_result[f"{col1}_x_{col2}"] = df_result[col1] * df_result[col2]
                    features_created += 1

        elif degree == 3 and not interaction_only:
            for col in valid_cols:
                df_result[f"{col}_squared"] = df_result[col] ** 2
                df_result[f"{col}_cubed"] = df_result[col] ** 3
                features_created += 2

        logger.info(f"Created {features_created} polynomial features")

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_binning(self, column: str, bins: Union[int, List[float]],
                       labels: Optional[List[str]] = None,
                       strategy: str = 'quantile',
                       inplace: bool = False) -> pd.DataFrame:
        """
        Bin continuous features into discrete intervals.

        Args:
            column: Column name to bin
            bins: Number of bins or list of bin edges
            labels: Labels for bins (optional)
            strategy: Binning strategy ('quantile' or 'uniform')
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with binned column

        Raises:
            ValueError: If invalid strategy or parameters
        """
        if strategy not in ['quantile', 'uniform']:
            raise ValueError("strategy must be 'quantile' or 'uniform'")

        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df_result if not inplace else self.df

        if not np.issubdtype(df_result[column].dtype, np.number):
            logger.warning(f"Column '{column}' is not numeric")
            return df_result if not inplace else self.df

        new_col_name = f"{column}_binned"

        try:
            if isinstance(bins, int):
                if strategy == 'quantile':
                    df_result[new_col_name] = pd.qcut(
                        df_result[column],
                        q=bins,
                        labels=labels,
                        duplicates='drop'
                    )
                else:
                    df_result[new_col_name] = pd.cut(
                        df_result[column],
                        bins=bins,
                        labels=labels
                    )
            else:
                df_result[new_col_name] = pd.cut(
                    df_result[column],
                    bins=bins,
                    labels=labels
                )
            logger.info(f"Binned column '{column}' into '{new_col_name}'")
        except Exception as e:
            logger.error(f"Error binning column '{column}': {e}")
            raise

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_log_transform(self, columns: List[str], inplace: bool = False) -> pd.DataFrame:
        """
        Apply log transformation to features.

        Args:
            columns: List of columns to transform
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with log-transformed columns
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue

            if not np.issubdtype(df_result[col].dtype, np.number):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue

            min_val = df_result[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                df_result[f"{col}_log"] = np.log(df_result[col] + offset)
                logger.info(f"Log transformed column '{col}' with offset {offset}")
            else:
                df_result[f"{col}_log"] = np.log(df_result[col])
                logger.info(f"Log transformed column '{col}'")

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_sqrt_transform(self, columns: List[str], inplace: bool = False) -> pd.DataFrame:
        """
        Apply square root transformation to features.

        Args:
            columns: List of columns to transform
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with sqrt-transformed columns
        """
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")

        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                continue

            if not np.issubdtype(df_result[col].dtype, np.number):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue

            min_val = df_result[col].min()
            if min_val < 0:
                logger.warning(f"Column '{col}' contains negative values, skipping")
                continue

            df_result[f"{col}_sqrt"] = np.sqrt(df_result[col])
            logger.info(f"Square root transformed column '{col}'")

        # Fixed: Update self.df when inplace=True
        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_datetime_features(self, column: str, features: Optional[List[str]] = None,
                                  inplace: bool = False) -> pd.DataFrame:
        """
        Extract datetime features from a datetime column.

        Args:
            column: Column name containing datetime data
            features: List of features to extract (default: common features)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with extracted datetime features
        """
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df_result if not inplace else self.df

        if not pd.api.types.is_datetime64_any_dtype(df_result[column]):
            try:
                df_result[column] = pd.to_datetime(df_result[column])
                logger.info(f"Converted column '{column}' to datetime")
            except Exception as e:
                logger.error(f"Could not convert '{column}' to datetime: {e}")
                return df_result if not inplace else self.df

        if features is None:
            features = ['year', 'month', 'day', 'dayofweek', 'hour', 'minute', 'quarter']

        feature_extractors = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day': lambda x: x.dt.day,
            'dayofweek': lambda x: x.dt.dayofweek,
            'hour': lambda x: x.dt.hour,
            'minute': lambda x: x.dt.minute,
            'second': lambda x: x.dt.second,
            'quarter': lambda x: x.dt.quarter,
            'dayofyear': lambda x: x.dt.dayofyear,
            'weekofyear': lambda x: x.dt.isocalendar().week,
            'is_weekend': lambda x: x.dt.dayofweek.isin([5, 6]).astype(int),
            'is_month_start': lambda x: x.dt.is_month_start.astype(int),
            'is_month_end': lambda x: x.dt.is_month_end.astype(int),
        }

        features_created = 0
        for feature in features:
            if feature in feature_extractors:
                try:
                    df_result[f"{column}_{feature}"] = feature_extractors[feature](df_result[column])
                    features_created += 1
                except Exception as e:
                    logger.warning(f"Could not extract feature '{feature}': {e}")

        logger.info(f"Created {features_created} datetime features from column '{column}'")

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_aggregations(self, group_by: Union[str, List[str]],
                            agg_column: str,
                            agg_funcs: List[str] = ['mean', 'sum', 'std', 'min', 'max'],
                            inplace: bool = False) -> pd.DataFrame:
        """
        Create aggregation features based on grouping.

        Args:
            group_by: Column(s) to group by
            agg_column: Column to aggregate
            agg_funcs: List of aggregation functions
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with aggregation features
        """
        df_result = self.df if inplace else self.df.copy()

        if isinstance(group_by, str):
            group_by = [group_by]

        # Validate columns
        for col in group_by + [agg_column]:
            if col not in df_result.columns:
                logger.warning(f"Column '{col}' not found in dataframe")
                return df_result if not inplace else self.df

        if not np.issubdtype(df_result[agg_column].dtype, np.number):
            logger.warning(f"Aggregation column '{agg_column}' is not numeric")
            return df_result if not inplace else self.df

        features_created = 0
        for func in agg_funcs:
            try:
                agg_name = f"{agg_column}_{'_'.join(group_by)}_{func}"
                agg_values = df_result.groupby(group_by)[agg_column].transform(func)
                df_result[agg_name] = agg_values
                features_created += 1
            except Exception as e:
                logger.warning(f"Could not create aggregation with function '{func}': {e}")

        logger.info(f"Created {features_created} aggregation features")

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_ratio_features(self, numerator: str, denominator: str,
                               name: Optional[str] = None,
                               epsilon: float = 1e-8,
                               inplace: bool = False) -> pd.DataFrame:
        """
        Create ratio features from two numeric columns.

        Args:
            numerator: Numerator column name
            denominator: Denominator column name
            name: Name for ratio feature (auto-generated if None)
            epsilon: Small value to add to denominator to avoid division by zero
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with ratio feature
        """
        df_result = self.df if inplace else self.df.copy()

        if numerator not in df_result.columns or denominator not in df_result.columns:
            logger.warning("One or both columns not found in dataframe")
            return df_result if not inplace else self.df

        for col in [numerator, denominator]:
            if not np.issubdtype(df_result[col].dtype, np.number):
                logger.warning(f"Column '{col}' is not numeric")
                return df_result if not inplace else self.df

        if name is None:
            name = f"{numerator}_to_{denominator}_ratio"

        # Fixed: Division by zero protection with configurable epsilon
        df_result[name] = df_result[numerator] / (df_result[denominator] + epsilon)
        logger.info(f"Created ratio feature '{name}'")

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_flag_features(self, column: str, condition: Any,
                             flag_name: Optional[str] = None,
                             inplace: bool = False) -> pd.DataFrame:
        """
        Create binary flag features based on conditions.

        Args:
            column: Column to evaluate condition on
            condition: Condition value or callable
            flag_name: Name for flag column (auto-generated if None)
            inplace: If True, modifies internal dataframe. Default False.

        Returns:
            DataFrame with flag feature
        """
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df_result if not inplace else self.df

        if flag_name is None:
            flag_name = f"{column}_flag"

        try:
            if callable(condition):
                df_result[flag_name] = df_result[column].apply(condition).astype(int)
            else:
                df_result[flag_name] = (df_result[column] == condition).astype(int)
            logger.info(f"Created flag feature '{flag_name}'")
        except Exception as e:
            logger.error(f"Error creating flag feature: {e}")
            raise

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def save_transformers(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted encoders and scalers to file for production use.

        Args:
            filepath: Path to save transformers (should end in .pkl or .joblib)

        Raises:
            ValueError: If no transformers have been fitted
        """
        if not self.encoders and not self.scalers:
            raise TransformerNotFittedError("encoder or scaler")

        transformers = {
            'encoders': self.encoders,
            'scalers': self.scalers
        }

        try:
            joblib.dump(transformers, filepath)
            logger.info(f"Saved {len(self.encoders)} encoders and {len(self.scalers)} scalers to {filepath}")
        except Exception as e:
            logger.error(f"Error saving transformers: {e}")
            raise

    def load_transformers(self, filepath: Union[str, Path]) -> None:
        """
        Load fitted encoders and scalers from file.

        Args:
            filepath: Path to load transformers from

        Raises:
            FileNotFoundError: If filepath doesn't exist
        """
        try:
            transformers = joblib.load(filepath)
            self.encoders = transformers.get('encoders', {})
            self.scalers = transformers.get('scalers', {})
            logger.info(f"Loaded {len(self.encoders)} encoders and {len(self.scalers)} scalers from {filepath}")
        except Exception as e:
            logger.error(f"Error loading transformers: {e}")
            raise

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the current dataframe.

        Returns:
            Copy of internal DataFrame
        """
        return self.df.copy()
