import numpy as np
import pandas as pd
from typing import Optional, Union, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif, f_regression


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Automated feature selector that combines:
    1. Variance Thresholding (removes zero and near-zero variance features).
    2. Statistical Significance / Correlation Ranking (SelectPercentile with ANOVA/F-statistic).
    Preserves and outputs consistent feature names.
    """
    def __init__(
        self,
        task: str = "classification",
        variance_threshold: float = 0.0,
        percentile: float = 90.0,
        enable_selection: bool = True
    ):
        self.task = task
        self.variance_threshold = variance_threshold
        self.percentile = percentile
        self.enable_selection = enable_selection
        
        self.var_selector_: Optional[VarianceThreshold] = None
        self.stat_selector_: Optional[SelectPercentile] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.input_feature_names_: List[str] = []
        self.selected_feature_names_: List[str] = []
        self._fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        if isinstance(X, pd.DataFrame):
            self.input_feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.input_feature_names_ = [f"f_{i}" for i in range(X_arr.shape[1])]

        n_features = X_arr.shape[1]

        if not self.enable_selection or n_features <= 5 or y is None:
            self.selected_indices_ = np.arange(n_features)
            self.selected_feature_names_ = self.input_feature_names_
            self._fitted = True
            return self

        # Step 1: Variance threshold
        self.var_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        try:
            X_var = self.var_selector_.fit_transform(X_arr)
            var_mask = self.var_selector_.get_support()
        except ValueError:
            # If all features fail variance threshold, keep original
            var_mask = np.ones(n_features, dtype=bool)
            X_var = X_arr

        # Step 2: Statistical significance
        score_func = f_classif if self.task == "classification" else f_regression
        
        # Format y for statistical testing
        y_arr = np.asarray(y)
        if self.task == "classification" and y_arr.dtype.kind in ('U', 'S', 'O'):
            # Convert categorical string labels to integers for f_classif
            from sklearn.preprocessing import LabelEncoder
            y_arr = LabelEncoder().fit_transform(y_arr)

        if X_var.shape[1] > 5 and self.percentile < 100.0:
            self.stat_selector_ = SelectPercentile(score_func=score_func, percentile=self.percentile)
            try:
                self.stat_selector_.fit(X_var, y_arr)
                stat_mask = self.stat_selector_.get_support()
            except Exception:
                stat_mask = np.ones(X_var.shape[1], dtype=bool)
        else:
            stat_mask = np.ones(X_var.shape[1], dtype=bool)

        # Combine masks
        final_mask = np.zeros(n_features, dtype=bool)
        var_indices = np.where(var_mask)[0]
        selected_var_indices = var_indices[stat_mask]
        final_mask[selected_var_indices] = True

        # Safety: Ensure at least 1 feature is selected
        if not np.any(final_mask):
            final_mask[0] = True

        self.selected_indices_ = np.where(final_mask)[0]
        self.selected_feature_names_ = [self.input_feature_names_[i] for i in self.selected_indices_]
        self._fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureSelector must be fitted before calling transform.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        return X_arr[:, self.selected_indices_]

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.selected_feature_names_
