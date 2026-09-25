"""
Command-Line Interface (CLI) for AutoML Framework.
Train, evaluate, and export production models directly from your terminal.
"""

import argparse
import sys
import os
import pandas as pd
from automl import AutoML

def main():
    parser = argparse.ArgumentParser(
        description="AutoML Framework CLI - Automatically train, benchmark, and export machine learning models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, required=True, help="Path to input dataset (CSV format).")
    parser.add_argument("--target", type=str, required=True, help="Name of the target column.")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "classification", "regression"], help="Machine learning task type.")
    parser.add_argument("--metric", type=str, default=None, help="Optimization metric (e.g., accuracy, f1_weighted, roc_auc, r2, rmse).")
    parser.add_argument("--time-budget", type=int, default=180, help="Maximum total training time in seconds.")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--trials", type=int, default=6, help="Hyperparameter search trials per candidate algorithm.")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable voting/stacking ensembling.")
    parser.add_argument("--output", type=str, default="automl_model.joblib", help="Filepath to save the trained pipeline artifact.")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file not found at '{args.data}'.", file=sys.stderr)
        sys.exit(1)

    print(f"[AutoML CLI] Loading dataset from '{args.data}'...")
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.target not in df.columns:
        print(f"Error: Target column '{args.target}' not found in dataset columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"[AutoML CLI] Dataset shape: {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"[AutoML CLI] Target column: '{args.target}'")

    automl = AutoML(
        task=args.task,
        primary_metric=args.metric,
        time_budget_secs=args.time_budget,
        n_splits=args.folds,
        n_iter_per_model=args.trials,
        ensemble=not args.no_ensemble,
        verbose=1
    )

    automl.fit(X, y)

    print("\n" + "=" * 60)
    print("                 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Champion Model: {automl.best_model_name_}")
    print("\nLeaderboard:")
    lb = automl.leaderboard()
    print(lb[["model", "mean_cv_score", "std_cv_score", "fit_time_secs", "is_ensemble"]].to_string())

    print("\nValidation Set Metrics:")
    for k, v in automl.validation_metrics_.items():
        print(f"  {k:20s}: {v:.4f}")

    automl.save(args.output)
    print(f"\nModel pipeline successfully saved to '{args.output}'.")
    print("Done!")


if __name__ == "__main__":
    main()
