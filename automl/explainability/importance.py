import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from sklearn.inspection import permutation_importance


class Explainer:
    """
    Computes global feature importance using model-agnostic Permutation Importance
    and intrinsic model importances/coefficients.
    """
    @staticmethod
    def get_feature_importance(
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 5,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Computes permutation importance on validation set.
        """
        n_features = X_val.shape[1]
        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Permutation importance
        try:
            perm_result = permutation_importance(
                model,
                X_val,
                y_val,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1
            )
            mean_importance = perm_result.importances_mean
            std_importance = perm_result.importances_std
        except Exception:
            # Fallback if permutation fails on ensemble
            mean_importance = np.zeros(n_features)
            std_importance = np.zeros(n_features)
            # Try native feature_importances_
            if hasattr(model, "feature_importances_"):
                mean_importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                mean_importance = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_importance,
            "std": std_importance
        })

        df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df
