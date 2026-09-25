import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error,
    confusion_matrix, classification_report
)


class Evaluator:
    """
    Computes comprehensive evaluation metrics for classification and regression tasks.
    """
    @staticmethod
    def get_default_metric(task: str, is_binary: bool = True) -> str:
        if task == "classification":
            return "roc_auc" if is_binary else "f1_weighted"
        elif task == "regression":
            return "r2"
        return "accuracy"

    @staticmethod
    def is_higher_better(metric: str) -> bool:
        lower_is_better = ["rmse", "mse", "mae", "mape", "medae", "log_loss"]
        return metric.lower() not in lower_is_better

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # Calculate ROC AUC and Log Loss if probabilities are provided
        if y_proba is not None:
            try:
                unique_classes = np.unique(y_true)
                if len(unique_classes) == 2:
                    # Binary classification
                    proba_col = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba_col))
                    metrics["log_loss"] = float(log_loss(y_true, y_proba))
                elif len(unique_classes) > 2:
                    # Multiclass
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
                    metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception:
                pass

        return metrics

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        medae = float(median_absolute_error(y_true, y_pred))
        
        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mse": float(mse),
            "mae": mae,
            "medae": medae
        }

        try:
            metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            pass

        return metrics

    @classmethod
    def evaluate(
        cls,
        task: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        if task == "classification":
            return cls.evaluate_classification(y_true, y_pred, y_proba)
        elif task == "regression":
            return cls.evaluate_regression(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task: {task}")


class Leaderboard:
    """
    Maintains, sorts, and displays model benchmark results.
    """
    def __init__(self, primary_metric: str, higher_is_better: bool = True):
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self.entries: List[Dict[str, Any]] = []

    def add_entry(
        self,
        model_name: str,
        mean_score: float,
        std_score: float,
        fit_time_secs: float,
        best_params: Dict[str, Any],
        is_ensemble: bool = False,
        all_metrics: Optional[Dict[str, float]] = None
    ):
        entry = {
            "model": model_name,
            "mean_cv_score": round(mean_score, 4),
            "std_cv_score": round(std_score, 4),
            "fit_time_secs": round(fit_time_secs, 2),
            "is_ensemble": is_ensemble,
            "best_params": best_params,
            "all_metrics": all_metrics or {}
        }
        self.entries.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame()
        df = pd.DataFrame(self.entries)
        # Sort by primary metric
        df = df.sort_values(
            by="mean_cv_score",
            ascending=not self.higher_is_better
        ).reset_index(drop=True)
        df.index = df.index + 1  # 1-indexed ranks
        df.index.name = "rank"
        return df

    def get_best_model_name(self) -> Optional[str]:
        df = self.to_dataframe()
        if df.empty:
            return None
        return df.iloc[0]["model"]

    def __repr__(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "<Leaderboard: Empty>"
        cols_to_show = ["model", "mean_cv_score", "std_cv_score", "fit_time_secs", "is_ensemble"]
        return df[cols_to_show].to_string()
