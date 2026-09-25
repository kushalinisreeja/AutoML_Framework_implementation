"""
AutoML Regression Demo
Demonstrates automated end-to-end model training, hyperparameter optimization,
voting/stacking ensembling, evaluation, and persistence on continuous regression data.
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from automl import AutoMLRegressor, AutoML

def run_demo():
    print("=" * 70)
    print("       AUTOMATED MACHINE LEARNING (AutoML) - REGRESSION DEMO        ")
    print("=" * 70)

    # 1. Load sample dataset (take 1000 samples for swift demo)
    print("\n[Step 1] Loading California Housing Dataset (1,000 sample subset)...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.sample(n=1000, random_state=42)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"Target Summary: Mean={y.mean():.2f}, Std={y.std():.2f}, Min={y.min():.2f}, Max={y.max():.2f}")

    # 2. Instantiate and Fit AutoML Regressor
    print("\n[Step 2] Initializing and Fitting AutoMLRegressor...")
    automl = AutoMLRegressor(
        time_budget_secs=120,
        n_iter_per_model=4,
        n_splits=4,
        ensemble=True,
        ensemble_type="both",
        top_k_models=3,
        random_state=42,
        verbose=1
    )
    automl.fit(X, y)

    # 3. Display Leaderboard
    print("\n[Step 3] Model Leaderboard:")
    print("-" * 70)
    leaderboard_df = automl.leaderboard()
    print(leaderboard_df[["model", "mean_cv_score", "std_cv_score", "fit_time_secs", "is_ensemble"]].to_string())
    print("-" * 70)
    print(f"Winning Champion Model: {automl.best_model_name_}")

    # 4. Final Validation Metrics
    print("\n[Step 4] Holdout Validation Set Metrics:")
    for metric, val in automl.validation_metrics_.items():
        print(f"  - {metric:20s}: {val:.4f}")

    # 5. Feature Importance
    print("\n[Step 5] Top Feature Importances:")
    fi = automl.get_feature_importance(top_n=8)
    print(fi.to_string(index=False))

    # 6. Model Persistence & Inference Test
    save_path = "california_housing_automl.joblib"
    print(f"\n[Step 6] Saving winning pipeline to '{save_path}'...")
    automl.save(save_path)

    print("Reloading model from disk to test inference...")
    loaded_model = AutoML.load(save_path)
    sample_preds = loaded_model.predict(X.head(5))
    print(f"Sample Actual Values (first 5 samples):\n{y.head(5).values}")
    print(f"Sample Predictions (first 5 samples):\n{sample_preds}")
    print("\nRegression Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
