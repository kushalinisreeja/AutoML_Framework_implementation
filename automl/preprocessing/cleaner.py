import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class TaskDetector:
    """
    Automatically infers whether a task is classification or regression,
    detecting binary vs multiclass and computing class balance.
    """
    @staticmethod
    def detect_task(y: Union[pd.Series, np.ndarray, list]) -> Dict[str, Any]:
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y_series = pd.Series(y)
        else:
            y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y

        # Remove nulls for task detection
        y_valid = y_series.dropna()
        n_samples = len(y_valid)
        if n_samples == 0:
            raise ValueError("Target variable y contains only missing values.")

        unique_vals = y_valid.unique()
        n_unique = len(unique_vals)

        dtype = y_valid.dtype
        is_string_or_cat = pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype)
        is_bool = pd.api.types.is_bool_dtype(dtype)

        # Decision heuristics
        if is_bool or is_string_or_cat:
            task = "classification"
        elif n_unique == 2:
            task = "classification"
        elif pd.api.types.is_integer_dtype(dtype) and (n_unique <= 20 and (n_unique / n_samples) < 0.2):
            task = "classification"
        elif pd.api.types.is_float_dtype(dtype):
            # Check if all floats are whole numbers and low cardinality
            all_int_like = (y_valid % 1 == 0).all()
            if all_int_like and n_unique <= 10 and (n_unique / n_samples) < 0.1:
                task = "classification"
            else:
                task = "regression"
        else:
            task = "regression"

        info = {
            "task": task,
            "n_samples": n_samples,
            "n_unique": n_unique,
            "unique_values": unique_vals.tolist() if n_unique <= 50 else unique_vals[:50].tolist(),
        }

        if task == "classification":
            info["classification_type"] = "binary" if n_unique == 2 else "multiclass"
            value_counts = y_valid.value_counts(normalize=True).to_dict()
            info["class_distribution"] = value_counts
            min_class_ratio = min(value_counts.values()) if value_counts else 0
            info["is_imbalanced"] = min_class_ratio < 0.2
        else:
            info["classification_type"] = None
            info["mean"] = float(y_valid.mean())
            info["std"] = float(y_valid.std())
            info["min"] = float(y_valid.min())
            info["max"] = float(y_valid.max())

        return info


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Automated preprocessor for tabular data:
    1. Type inference (numeric, categorical, datetime, drop candidates).
    2. Datetime feature extraction.
    3. Missing value imputation.
    4. Categorical encoding (One-Hot for low-cardinality, Ordinal for high-cardinality).
    5. Feature scaling (Standard, Robust, MinMax).
    """
    def __init__(
        self,
        impute_strategy: str = "auto",
        scale_numeric: bool = True,
        scaling_method: str = "standard",
        encode_categorical: bool = True,
        max_one_hot_cardinality: int = 15,
        extract_datetime: bool = True
    ):
        self.impute_strategy = impute_strategy
        self.scale_numeric = scale_numeric
        self.scaling_method = scaling_method
        self.encode_categorical = encode_categorical
        self.max_one_hot_cardinality = max_one_hot_cardinality
        self.extract_datetime = extract_datetime
        
        self.dropped_cols_: List[str] = []
        self.datetime_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.low_cardinality_cols_: List[str] = []
        self.high_cardinality_cols_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.pipeline_: Optional[ColumnTransformer] = None
        self._fitted = False

    def _infer_column_types(self, df: pd.DataFrame):
        self.dropped_cols_ = []
        self.datetime_cols_ = []
        self.numeric_cols_ = []
        self.low_cardinality_cols_ = []
        self.high_cardinality_cols_ = []

        n_rows = len(df)
        for col in df.columns:
            series = df[col]
            n_unique = series.nunique(dropna=True)

            # Check 1: Drop constant column
            if n_unique <= 1:
                self.dropped_cols_.append(col)
                continue

            # Check 2: Drop pure identifier columns (e.g., 'id', or 100% unique string/int column with ID-like name)
            col_lower = str(col).lower()
            is_id_name = any(re.match(pattern, col_lower) for pattern in [r"^id$", r".*_id$", r"^uuid$", r"^guid$", r"^index$"])
            if is_id_name and (n_unique / n_rows > 0.8):
                self.dropped_cols_.append(col)
                continue

            # Check 3: Datetime detection
            if self.extract_datetime:
                if pd.api.types.is_datetime64_any_dtype(series):
                    self.datetime_cols_.append(col)
                    continue
                elif series.dtype == "object":
                    sample = series.dropna().astype(str).head(50)
                    if len(sample) > 0:
                        try:
                            # Try parsing dates if format matches common patterns
                            parsed = pd.to_datetime(sample, errors='coerce')
                            if parsed.notna().mean() > 0.8:
                                self.datetime_cols_.append(col)
                                continue
                        except Exception:
                            pass

            # Check 4: Numeric vs Categorical
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                self.numeric_cols_.append(col)
            else:
                if n_unique <= self.max_one_hot_cardinality:
                    self.low_cardinality_cols_.append(col)
                else:
                    self.high_cardinality_cols_.append(col)

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.datetime_cols_:
            dt_series = pd.to_datetime(df_out[col], errors='coerce')
            df_out[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
            df_out[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
            df_out[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
            df_out[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
            df_out[f"{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            df_out = df_out.drop(columns=[col])
        return df_out

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        else:
            X = X.copy()

        self._infer_column_types(X)

        # Pre-process datetimes if any
        if self.datetime_cols_:
            X_transformed = self._extract_datetime_features(X)
            # Re-infer new datetime derived features (they are numeric)
            for col in self.datetime_cols_:
                for suffix in ["_year", "_month", "_day", "_dayofweek", "_is_weekend"]:
                    dt_feat = f"{col}{suffix}"
                    if dt_feat in X_transformed.columns and dt_feat not in self.numeric_cols_:
                        self.numeric_cols_.append(dt_feat)
        else:
            X_transformed = X

        # Build ColumnTransformer transformers
        transformers = []

        # 1. Numeric pipeline
        if self.numeric_cols_:
            num_imputer_strategy = "median" if self.impute_strategy in ["auto", "median"] else self.impute_strategy
            num_steps = [("imputer", SimpleImputer(strategy=num_imputer_strategy))]
            if self.scale_numeric:
                if self.scaling_method == "robust":
                    num_steps.append(("scaler", RobustScaler()))
                elif self.scaling_method == "minmax":
                    num_steps.append(("scaler", MinMaxScaler()))
                else:
                    num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(num_steps), self.numeric_cols_))

        # 2. Low-cardinality categorical pipeline (One-Hot)
        if self.low_cardinality_cols_ and self.encode_categorical:
            cat_low_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat_low", cat_low_pipeline, self.low_cardinality_cols_))

        # 3. High-cardinality categorical pipeline (Ordinal)
        if self.high_cardinality_cols_ and self.encode_categorical:
            cat_high_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
            transformers.append(("cat_high", cat_high_pipeline, self.high_cardinality_cols_))

        if not transformers:
            # Fallback if no features identified
            transformers.append(("fallback_num", SimpleImputer(strategy="median"), list(X_transformed.columns)))

        self.pipeline_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )

        self.pipeline_.fit(X_transformed, y)
        self._fitted = True

        # Extract output feature names
        try:
            self.feature_names_out_ = list(self.pipeline_.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = [f"f_{i}" for i in range(self.transform(X).shape[1])]

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        else:
            X = X.copy()

        if self.datetime_cols_:
            X = self._extract_datetime_features(X)

        return self.pipeline_.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_out_
