import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, OrdinalEncoder
)
from typing import List, Optional, Dict, Union, Any


class FeatureEngineer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}

    def encode_categorical_label(self, columns: List[str], inplace: bool = True) -> pd.DataFrame:
        """Encode categorical columns using label encoding."""
        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found in dataframe.")
                continue

            encoder = LabelEncoder()
            df_result[col] = encoder.fit_transform(df_result[col].astype(str))
            self.encoders[f"{col}_label"] = encoder

        if not inplace:
            return df_result
        return self.df

    def encode_categorical_onehot(self, columns: List[str], drop_first: bool = False,
                                   prefix: Optional[Dict[str, str]] = None,
                                   inplace: bool = True) -> pd.DataFrame:
        """Encode categorical columns using one-hot encoding."""
        df_result = self.df if inplace else self.df.copy()

        if prefix is None:
            prefix = {col: col for col in columns}

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found in dataframe.")
                continue

            dummies = pd.get_dummies(
                df_result[col],
                prefix=prefix.get(col, col),
                drop_first=drop_first,
                dtype=int
            )

            df_result = pd.concat([df_result, dummies], axis=1)
            df_result = df_result.drop(columns=[col])

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def encode_categorical_ordinal(self, column: str, categories: List[Any],
                                    inplace: bool = True) -> pd.DataFrame:
        """Encode categorical column with ordinal relationship."""
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            print(f"Warning: Column '{column}' not found in dataframe.")
            return df_result

        encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
        df_result[column] = encoder.fit_transform(df_result[[column]])
        self.encoders[f"{column}_ordinal"] = encoder

        if not inplace:
            return df_result
        return self.df

    def scale_features(self, columns: List[str], method: str = 'standard',
                       inplace: bool = True) -> pd.DataFrame:
        """Scale numeric features using specified method."""
        df_result = self.df if inplace else self.df.copy()

        scalers_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        if method not in scalers_map:
            raise ValueError(f"Method must be one of {list(scalers_map.keys())}")

        scaler = scalers_map[method]

        valid_cols = [col for col in columns if col in df_result.columns]
        if not valid_cols:
            print(f"Warning: None of the specified columns found in dataframe.")
            return df_result

        df_result[valid_cols] = scaler.fit_transform(df_result[valid_cols])
        self.scalers[f"{method}_scaler"] = scaler

        if not inplace:
            return df_result
        return self.df

    def create_polynomial_features(self, columns: List[str], degree: int = 2,
                                    interaction_only: bool = False,
                                    inplace: bool = True) -> pd.DataFrame:
        """Create polynomial and interaction features."""
        df_result = self.df if inplace else self.df.copy()

        valid_cols = [col for col in columns if col in df_result.columns]
        if not valid_cols:
            print(f"Warning: None of the specified columns found in dataframe.")
            return df_result

        if degree == 2:
            if not interaction_only:
                for col in valid_cols:
                    df_result[f"{col}_squared"] = df_result[col] ** 2

            for i, col1 in enumerate(valid_cols):
                for col2 in valid_cols[i + 1:]:
                    df_result[f"{col1}_x_{col2}"] = df_result[col1] * df_result[col2]

        elif degree == 3 and not interaction_only:
            for col in valid_cols:
                df_result[f"{col}_squared"] = df_result[col] ** 2
                df_result[f"{col}_cubed"] = df_result[col] ** 3

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_binning(self, column: str, bins: Union[int, List[float]],
                       labels: Optional[List[str]] = None,
                       strategy: str = 'quantile',
                       inplace: bool = True) -> pd.DataFrame:
        """Bin continuous features into discrete intervals."""
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            print(f"Warning: Column '{column}' not found in dataframe.")
            return df_result

        new_col_name = f"{column}_binned"

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

        if not inplace:
            return df_result
        return self.df

    def create_log_transform(self, columns: List[str], inplace: bool = True) -> pd.DataFrame:
        """Apply log transformation to features."""
        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found in dataframe.")
                continue

            min_val = df_result[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                df_result[f"{col}_log"] = np.log(df_result[col] + offset)
            else:
                df_result[f"{col}_log"] = np.log(df_result[col])

        if not inplace:
            return df_result
        return self.df

    def create_sqrt_transform(self, columns: List[str], inplace: bool = True) -> pd.DataFrame:
        """Apply square root transformation to features."""
        df_result = self.df if inplace else self.df.copy()

        for col in columns:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found in dataframe.")
                continue

            min_val = df_result[col].min()
            if min_val < 0:
                print(f"Warning: Column '{col}' contains negative values. Skipping.")
                continue

            df_result[f"{col}_sqrt"] = np.sqrt(df_result[col])

        if not inplace:
            return df_result
        return self.df

    def create_datetime_features(self, column: str, features: Optional[List[str]] = None,
                                  inplace: bool = True) -> pd.DataFrame:
        """Extract datetime features from a datetime column."""
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            print(f"Warning: Column '{column}' not found in dataframe.")
            return df_result

        if not pd.api.types.is_datetime64_any_dtype(df_result[column]):
            df_result[column] = pd.to_datetime(df_result[column])

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

        for feature in features:
            if feature in feature_extractors:
                df_result[f"{column}_{feature}"] = feature_extractors[feature](df_result[column])

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_aggregations(self, group_by: Union[str, List[str]],
                            agg_column: str,
                            agg_funcs: List[str] = ['mean', 'sum', 'std', 'min', 'max'],
                            inplace: bool = True) -> pd.DataFrame:
        """Create aggregation features based on grouping."""
        df_result = self.df if inplace else self.df.copy()

        if isinstance(group_by, str):
            group_by = [group_by]

        for col in group_by + [agg_column]:
            if col not in df_result.columns:
                print(f"Warning: Column '{col}' not found in dataframe.")
                return df_result

        for func in agg_funcs:
            agg_name = f"{agg_column}_{'_'.join(group_by)}_{func}"
            agg_values = df_result.groupby(group_by)[agg_column].transform(func)
            df_result[agg_name] = agg_values

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_ratio_features(self, numerator: str, denominator: str,
                               name: Optional[str] = None,
                               inplace: bool = True) -> pd.DataFrame:
        """Create ratio features from two numeric columns."""
        df_result = self.df if inplace else self.df.copy()

        if numerator not in df_result.columns or denominator not in df_result.columns:
            print(f"Warning: One or both columns not found in dataframe.")
            return df_result

        if name is None:
            name = f"{numerator}_to_{denominator}_ratio"

        df_result[name] = df_result[numerator] / (df_result[denominator] + 1e-8)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def create_flag_features(self, column: str, condition: Any,
                             flag_name: Optional[str] = None,
                             inplace: bool = True) -> pd.DataFrame:
        """Create binary flag features based on conditions."""
        df_result = self.df if inplace else self.df.copy()

        if column not in df_result.columns:
            print(f"Warning: Column '{column}' not found in dataframe.")
            return df_result

        if flag_name is None:
            flag_name = f"{column}_flag"

        if callable(condition):
            df_result[flag_name] = df_result[column].apply(condition).astype(int)
        else:
            df_result[flag_name] = (df_result[column] == condition).astype(int)

        if inplace:
            self.df = df_result
            return self.df
        return df_result

    def get_dataframe(self) -> pd.DataFrame:
        """Return the current dataframe."""
        return self.df.copy()
