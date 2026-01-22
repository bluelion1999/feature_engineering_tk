"""
Feature engineering and model suggestion mixins for TargetAnalyzer.

Contains methods for suggesting feature transformations and ML models.
"""

import numpy as np
import logging
from typing import Dict, Any, List

from ..utils import get_feature_columns, get_numeric_columns

# Configure logging
logger = logging.getLogger(__name__)


class SuggestionsMixin:
    """
    Mixin providing feature engineering and model suggestions for TargetAnalyzer.

    Requires: self.df, self.target_column, self.task,
              self.NONLINEAR_IMPROVEMENT_THRESHOLD,
              self.analyze_feature_correlations, self.analyze_feature_target_relationship,
              self.get_task_info, self.get_class_imbalance_info, self.analyze_target_distribution
    """

    def suggest_feature_engineering(self) -> List[Dict[str, Any]]:
        """
        Analyze features and generate intelligent feature engineering suggestions.

        Provides actionable recommendations for:
        - Transformations (log, sqrt, polynomial)
        - Scaling strategies
        - Encoding approaches for categorical features
        - Interaction terms
        - Binning strategies

        Returns:
            List of dicts with 'feature', 'suggestion', 'reason', and 'priority' keys
        """
        suggestions = []
        features = get_feature_columns(self.df, exclude_columns=[self.target_column], numeric_only=False)
        numeric_features = get_numeric_columns(self.df, columns=features)
        categorical_features = [col for col in features if col not in numeric_features]

        # 1. Analyze numeric features for transformations
        for feature in numeric_features:
            col_data = self.df[feature].dropna()
            if len(col_data) < 10:
                continue

            # Check skewness
            skewness = col_data.skew()
            if abs(skewness) > 1.0:
                direction = "right" if skewness > 0 else "left"
                transform = "log or sqrt" if skewness > 0 else "square or exponential"
                suggestions.append({
                    'feature': feature,
                    'suggestion': f'Apply {transform} transformation',
                    'reason': f'Feature is {direction}-skewed (skewness={skewness:.2f})',
                    'priority': 'high' if abs(skewness) > 2 else 'medium'
                })

            # Check for non-linear relationships with target (regression only)
            if self.task == 'regression' and len(col_data) > 20:
                target_clean = self.df[self.target_column].loc[col_data.index]
                linear_corr = np.corrcoef(col_data, target_clean)[0, 1]

                # Skip if correlation is NaN (e.g., constant feature)
                if np.isnan(linear_corr):
                    continue

                linear_corr = abs(linear_corr)

                # Create polynomial features
                col_squared = col_data ** 2
                poly_corr = np.corrcoef(col_squared, target_clean)[0, 1]

                # Skip if polynomial correlation is NaN
                if np.isnan(poly_corr):
                    continue

                poly_corr = abs(poly_corr)

                if poly_corr > linear_corr * self.NONLINEAR_IMPROVEMENT_THRESHOLD:
                    suggestions.append({
                        'feature': feature,
                        'suggestion': 'Create polynomial features (squared, cubed)',
                        'reason': f'Non-linear relationship detected (poly corr: {poly_corr:.3f} vs linear: {linear_corr:.3f})',
                        'priority': 'high'
                    })

            # Check range for scaling recommendation
            min_val, max_val = col_data.min(), col_data.max()
            value_range = max_val - min_val

            if value_range > 100:
                suggestions.append({
                    'feature': feature,
                    'suggestion': 'Apply StandardScaler or MinMaxScaler',
                    'reason': f'Large value range ({min_val:.1f} to {max_val:.1f})',
                    'priority': 'medium'
                })

        # 2. Analyze categorical features
        for feature in categorical_features:
            col_data = self.df[feature].dropna()
            if len(col_data) == 0:
                continue

            cardinality = col_data.nunique()

            if cardinality <= 5:
                suggestions.append({
                    'feature': feature,
                    'suggestion': 'One-hot encode',
                    'reason': f'Low cardinality ({cardinality} unique values)',
                    'priority': 'high'
                })
            elif cardinality <= 15:
                suggestions.append({
                    'feature': feature,
                    'suggestion': 'Target encode or ordinal encode',
                    'reason': f'Medium cardinality ({cardinality} unique values)',
                    'priority': 'medium'
                })
            else:
                suggestions.append({
                    'feature': feature,
                    'suggestion': 'Target encode or group rare categories',
                    'reason': f'High cardinality ({cardinality} unique values)',
                    'priority': 'high'
                })

        # 3. Interaction terms (top correlated features)
        if len(numeric_features) >= 2 and self.task == 'regression':
            corr_df = self.analyze_feature_correlations(method='pearson')
            if not corr_df.empty:
                top_features = corr_df.head(3)['feature'].tolist()
                if len(top_features) >= 2:
                    suggestions.append({
                        'feature': ', '.join(top_features[:2]),
                        'suggestion': 'Create interaction terms (multiplication, ratios)',
                        'reason': f'Top features strongly correlated with target',
                        'priority': 'medium'
                    })

        # 4. Binning for continuous features with weak linear relationships
        if self.task == 'classification' and len(numeric_features) > 0:
            relationships = self.analyze_feature_target_relationship()
            if not relationships.empty:
                weak_linear = relationships[
                    (relationships['test_type'].str.contains('ANOVA', na=False)) &
                    (relationships['pvalue'] > 0.05)
                ]['feature'].tolist()

                for feature in weak_linear[:3]:
                    suggestions.append({
                        'feature': feature,
                        'suggestion': 'Bin into categorical groups',
                        'reason': 'Weak linear relationship with target - binning may capture non-linear patterns',
                        'priority': 'low'
                    })

        # 5. Missing value patterns
        missing_info = self.df[features].isnull().sum()
        features_with_missing = missing_info[missing_info > 0].index.tolist()

        for feature in features_with_missing:
            missing_pct = (missing_info[feature] / len(self.df)) * 100
            if missing_pct > 5:
                suggestions.append({
                    'feature': feature,
                    'suggestion': f'Create missing indicator flag',
                    'reason': f'{missing_pct:.1f}% missing - missingness may be informative',
                    'priority': 'medium' if missing_pct > 20 else 'low'
                })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order[x['priority']])

        logger.info(f"Generated {len(suggestions)} feature engineering suggestions")
        return suggestions

    def recommend_models(self) -> List[Dict[str, Any]]:
        """
        Recommend ML algorithms based on data characteristics and task type.

        Analyzes:
        - Task type (classification/regression)
        - Dataset size and dimensionality
        - Class imbalance (classification)
        - Feature relationships (linear/non-linear)
        - Target distribution (regression)

        Returns:
            List of dicts with 'model', 'reason', 'priority', and 'considerations' keys
        """
        recommendations = []
        n_samples, n_features = len(self.df), len(self.df.columns) - 1
        task_info = self.get_task_info()

        # Dataset size categories
        is_small = n_samples < 1000
        is_large = n_samples > 50000
        is_high_dim = n_features > 50

        if self.task == 'classification':
            imbalance_info = self.get_class_imbalance_info()
            is_imbalanced = not imbalance_info['is_balanced']
            is_binary = task_info.get('class_count', 0) == 2

            # Tree-based models
            if is_imbalanced:
                recommendations.append({
                    'model': 'Random Forest with class_weight="balanced"',
                    'reason': 'Handles class imbalance well, robust to outliers, good baseline',
                    'priority': 'high',
                    'considerations': 'May overfit on small datasets. Tune max_depth and min_samples_split.'
                })

                recommendations.append({
                    'model': 'XGBoost with scale_pos_weight',
                    'reason': f'Excellent for imbalanced data (ratio: {imbalance_info["imbalance_ratio"]:.1f}:1), high performance',
                    'priority': 'high',
                    'considerations': 'Tune learning_rate, max_depth. Can be slower to train.'
                })
            else:
                recommendations.append({
                    'model': 'Random Forest',
                    'reason': 'Balanced classes, robust ensemble method, good feature importance',
                    'priority': 'high',
                    'considerations': 'Fast training, handles non-linear relationships well.'
                })

            # Logistic Regression
            if not is_high_dim or n_samples > n_features * 10:
                penalty = 'l1 or l2' if is_high_dim else 'l2'
                recommendations.append({
                    'model': f'Logistic Regression (penalty={penalty})',
                    'reason': 'Fast, interpretable, good for linear relationships',
                    'priority': 'medium',
                    'considerations': f'Requires feature scaling. Good baseline for {"binary" if is_binary else "multiclass"} classification.'
                })

            # SVM (for small-medium datasets)
            if is_small:
                recommendations.append({
                    'model': 'Support Vector Machine (SVM)',
                    'reason': 'Effective for small datasets with clear margin of separation',
                    'priority': 'medium',
                    'considerations': 'Sensitive to scaling. Try RBF kernel for non-linear boundaries.'
                })

            # Neural Networks (for large datasets)
            if is_large and not is_high_dim:
                recommendations.append({
                    'model': 'Neural Network (MLP)',
                    'reason': f'Large dataset ({n_samples} samples) can leverage deep learning',
                    'priority': 'medium',
                    'considerations': 'Requires careful tuning, feature scaling, and regularization.'
                })

            # Gradient Boosting
            recommendations.append({
                'model': 'LightGBM',
                'reason': 'Fast, memory efficient, handles categorical features natively',
                'priority': 'high' if is_large else 'medium',
                'considerations': 'Excellent speed/performance trade-off. Good default parameters.'
            })

        else:  # Regression
            target_dist = self.analyze_target_distribution()
            has_outliers = target_dist.get('has_outliers', False)

            # Tree-based models
            recommendations.append({
                'model': 'Random Forest Regressor',
                'reason': 'Robust baseline, handles non-linear relationships, feature importance',
                'priority': 'high',
                'considerations': 'Good starting point. Tune n_estimators and max_depth.'
            })

            recommendations.append({
                'model': 'XGBoost Regressor',
                'reason': 'High performance, handles complex patterns',
                'priority': 'high',
                'considerations': 'Often wins competitions. Tune learning_rate, max_depth, subsample.'
            })

            # Linear models
            if has_outliers:
                recommendations.append({
                    'model': 'Huber Regressor',
                    'reason': f'Target has outliers (IQR method), robust to outliers',
                    'priority': 'medium',
                    'considerations': 'More robust than Linear Regression for noisy data.'
                })
            else:
                if is_high_dim:
                    recommendations.append({
                        'model': 'Ridge or Lasso Regression',
                        'reason': f'High dimensional ({n_features} features), regularization prevents overfitting',
                        'priority': 'medium',
                        'considerations': 'Use Ridge for correlated features, Lasso for feature selection.'
                    })
                else:
                    recommendations.append({
                        'model': 'Linear Regression',
                        'reason': 'Simple, interpretable, good baseline',
                        'priority': 'medium',
                        'considerations': 'Fast training. Check residuals for linearity assumption.'
                    })

            # Gradient Boosting
            recommendations.append({
                'model': 'LightGBM Regressor',
                'reason': 'Fast training, excellent performance',
                'priority': 'high' if is_large else 'medium',
                'considerations': 'Great speed/accuracy balance. Handle categorical features automatically.'
            })

            # Neural Networks (for large datasets)
            if is_large:
                recommendations.append({
                    'model': 'Neural Network (MLP Regressor)',
                    'reason': f'Large dataset ({n_samples} samples) suitable for deep learning',
                    'priority': 'low',
                    'considerations': 'Requires scaling, tuning, and more training time.'
                })

        # General recommendations
        if is_small:
            recommendations.append({
                'model': 'Cross-Validation with multiple models',
                'reason': f'Small dataset ({n_samples} samples) - compare multiple approaches',
                'priority': 'high',
                'considerations': 'Use stratified k-fold. Avoid complex models that may overfit.'
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])

        logger.info(f"Generated {len(recommendations)} model recommendations for {self.task}")
        return recommendations
