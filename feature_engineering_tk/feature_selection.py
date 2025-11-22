import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    mutual_info_regression, chi2, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class FeatureSelector:

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.df = df.copy()
        self.target_column = target_column
        self.selected_features = []
        self.feature_scores = {}

    def select_by_variance(self, threshold: float = 0.0,
                           exclude_columns: Optional[List[str]] = None) -> List[str]:
        """Select features based on variance threshold."""
        if exclude_columns is None:
            exclude_columns = []

        if self.target_column:
            exclude_columns.append(self.target_column)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]

        if not feature_cols:
            logger.warning("No numeric features available for variance-based selection")
            return []

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.df[feature_cols])

        selected = [col for col, selected in zip(feature_cols, selector.get_support()) if selected]
        self.selected_features = selected

        variances = dict(zip(feature_cols, self.df[feature_cols].var()))
        self.feature_scores['variance'] = variances

        return selected

    def select_by_correlation(self, threshold: float = 0.7,
                               method: str = 'pearson',
                               exclude_columns: Optional[List[str]] = None) -> List[str]:
        """Remove highly correlated features, keeping one from each pair."""
        if exclude_columns is None:
            exclude_columns = []

        if self.target_column:
            exclude_columns.append(self.target_column)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]

        if len(feature_cols) < 2:
            logger.warning("Not enough features for correlation-based selection")
            return feature_cols

        corr_matrix = self.df[feature_cols].corr(method=method).abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

        selected = [col for col in feature_cols if col not in to_drop]
        self.selected_features = selected

        return selected

    def select_by_target_correlation(self, k: int = 10,
                                      method: str = 'pearson',
                                      exclude_columns: Optional[List[str]] = None) -> List[str]:
        """Select top k features most correlated with target."""
        if not self.target_column:
            raise ValueError("target_column must be specified for this method")

        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")

        if exclude_columns is None:
            exclude_columns = []

        exclude_columns.append(self.target_column)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]

        if not feature_cols:
            logger.warning("No numeric features available for correlation-based selection")
            return []

        correlations = self.df[feature_cols].corrwith(self.df[self.target_column], method=method).abs()
        correlations = correlations.sort_values(ascending=False)

        top_k = min(k, len(correlations))
        selected = correlations.head(top_k).index.tolist()
        self.selected_features = selected
        self.feature_scores['correlation_with_target'] = correlations.to_dict()

        return selected

    def select_by_statistical_test(self, k: int = 10,
                                    task: str = 'classification',
                                    score_func: Optional[str] = None,
                                    exclude_columns: Optional[List[str]] = None) -> List[str]:
        """Select features using statistical tests."""
        if not self.target_column:
            raise ValueError("target_column must be specified for this method")

        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")

        if exclude_columns is None:
            exclude_columns = []

        exclude_columns.append(self.target_column)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]

        if not feature_cols:
            logger.warning("No numeric features available for statistical test selection")
            return []

        X = self.df[feature_cols]
        y = self.df[self.target_column]

        if score_func is None:
            if task == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
        else:
            score_func_map = {
                'f_classif': f_classif,
                'f_regression': f_regression,
                'mutual_info_classif': mutual_info_classif,
                'mutual_info_regression': mutual_info_regression,
                'chi2': chi2
            }
            score_func = score_func_map.get(score_func, f_classif)

        top_k = min(k, len(feature_cols))
        selector = SelectKBest(score_func=score_func, k=top_k)

        try:
            selector.fit(X, y)
            selected_mask = selector.get_support()
            selected = [col for col, selected in zip(feature_cols, selected_mask) if selected]

            scores = dict(zip(feature_cols, selector.scores_))
            self.feature_scores['statistical_test'] = scores

            self.selected_features = selected
            return selected

        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            return []

    def select_by_importance(self, k: int = 10,
                             task: str = 'classification',
                             n_estimators: int = 100,
                             random_state: Optional[int] = 42,
                             exclude_columns: Optional[List[str]] = None) -> List[str]:
        """Select features using tree-based feature importance."""
        if not self.target_column:
            raise ValueError("target_column must be specified for this method")

        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")

        if exclude_columns is None:
            exclude_columns = []

        exclude_columns.append(self.target_column)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]

        if not feature_cols:
            logger.warning("No numeric features available for importance-based selection")
            return []

        X = self.df[feature_cols]
        y = self.df[self.target_column]

        if task == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

        try:
            model.fit(X, y)
            importances = dict(zip(feature_cols, model.feature_importances_))
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            top_k = min(k, len(sorted_features))
            selected = [feat for feat, _ in sorted_features[:top_k]]

            self.feature_scores['importance'] = importances
            self.selected_features = selected

            return selected

        except Exception as e:
            logger.error(f"Error during importance-based selection: {e}")
            return []

    def select_by_missing_values(self, threshold: float = 0.5,
                                  exclude_columns: Optional[List[str]] = None) -> List[str]:
        """Select features with missing values below threshold."""
        if exclude_columns is None:
            exclude_columns = []

        if self.target_column:
            exclude_columns.append(self.target_column)

        feature_cols = [col for col in self.df.columns if col not in exclude_columns]

        missing_ratios = self.df[feature_cols].isnull().sum() / len(self.df)
        selected = missing_ratios[missing_ratios <= threshold].index.tolist()

        self.selected_features = selected
        self.feature_scores['missing_ratios'] = missing_ratios.to_dict()

        return selected

    def get_feature_importance_df(self, sort: bool = True) -> pd.DataFrame:
        """Get a dataframe of feature scores from the last selection method."""
        if not self.feature_scores:
            logger.warning("No feature scores available. Run a selection method first")
            return pd.DataFrame()

        latest_score_type = list(self.feature_scores.keys())[-1]
        scores = self.feature_scores[latest_score_type]

        df_scores = pd.DataFrame({
            'feature': list(scores.keys()),
            'score': list(scores.values()),
            'score_type': latest_score_type
        })

        if sort:
            df_scores = df_scores.sort_values('score', ascending=False)

        return df_scores.reset_index(drop=True)

    def apply_selection(self, selected_features: Optional[List[str]] = None,
                        keep_target: bool = True) -> pd.DataFrame:
        """Apply feature selection to dataframe."""
        if selected_features is None:
            selected_features = self.selected_features

        if not selected_features:
            logger.warning("No features selected. Returning original dataframe")
            return self.df.copy()

        cols_to_keep = selected_features.copy()

        if keep_target and self.target_column:
            if self.target_column not in cols_to_keep:
                cols_to_keep.append(self.target_column)

        valid_cols = [col for col in cols_to_keep if col in self.df.columns]

        return self.df[valid_cols].copy()

    def get_selected_features(self) -> List[str]:
        """Return the list of selected features."""
        return self.selected_features.copy()


def select_features_auto(df: pd.DataFrame,
                          target_column: str,
                          task: str = 'classification',
                          max_features: int = 20,
                          variance_threshold: float = 0.01,
                          correlation_threshold: float = 0.9) -> pd.DataFrame:
    """Automatically select features using multiple methods."""
    selector = FeatureSelector(df, target_column)

    print("Step 1: Removing low variance features...")
    variance_features = selector.select_by_variance(threshold=variance_threshold)
    print(f"  - {len(variance_features)} features passed variance threshold")

    print("\nStep 2: Removing highly correlated features...")
    df_variance = selector.apply_selection(variance_features, keep_target=True)
    selector_corr = FeatureSelector(df_variance, target_column)
    correlation_features = selector_corr.select_by_correlation(threshold=correlation_threshold)
    print(f"  - {len(correlation_features)} features after correlation filtering")

    print("\nStep 3: Selecting top features by importance...")
    df_corr = selector_corr.apply_selection(correlation_features, keep_target=True)
    selector_final = FeatureSelector(df_corr, target_column)
    k = min(max_features, len(correlation_features))
    final_features = selector_final.select_by_importance(k=k, task=task)
    print(f"  - {len(final_features)} final features selected")

    print("\nFinal selected features:")
    for i, feat in enumerate(final_features, 1):
        print(f"  {i}. {feat}")

    result_df = selector_final.apply_selection(final_features, keep_target=True)
    return result_df
