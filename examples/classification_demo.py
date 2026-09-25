"""
AutoML Classification Demo
Demonstrates automated end-to-end model training, hyperparameter optimization,
voting/stacking ensembling, evaluation, explainability, and persistence on tabular data.
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from automl import AutoMLClassifier, AutoML

def run_demo():
    print("=" * 70)
    print("      AUTOMATED MACHINE LEARNING (AutoML) - CLASSIFICATION DEMO      ")
    print("=" * 70)

    # 1. Load sample dataset
    print("\n[Step 1] Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"Target distribution:\n{y.value_counts(normalize=True).to_dict()}")

    # 2. Instantiate and Fit AutoML Classifier
    print("\n[Step 2] Initializing and Fitting AutoMLClassifier...")
    automl = AutoMLClassifier(
        time_budget_secs=120,
        n_iter_per_model=5,
        n_splits=5,
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
    print("\n[Step 5] Top 10 Most Important Features:")
    fi = automl.get_feature_importance(top_n=10)
    print(fi.to_string(index=False))

    # 6. Model Persistence & Inference Test
    save_path = "breast_cancer_automl.joblib"
    print(f"\n[Step 6] Saving winning pipeline to '{save_path}'...")
    automl.save(save_path)

    print("Reloading model from disk to test inference...")
    loaded_model = AutoML.load(save_path)
    sample_preds = loaded_model.predict(X.head(5))
    sample_proba = loaded_model.predict_proba(X.head(5))
    print(f"Sample Predictions (first 5 samples): {sample_preds}")
    print(f"Sample Probabilities (first 5 samples):\n{sample_proba}")
    print("\nClassification Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
