from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from scipy.stats import uniform, randint, loguniform

from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor


@dataclass
class ModelCandidate:
    name: str
    estimator_class: Any
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_distributions: Dict[str, Any] = field(default_factory=dict)
    fast_priority: int = 1  # 1 is fastest/evaluated first


class ModelRegistry:
    """
    Curated model zoo and hyperparameter distributions for Classification and Regression.
    """
    @staticmethod
    def get_classification_candidates(random_state: int = 42) -> Dict[str, ModelCandidate]:
        return {
            "LogisticRegression": ModelCandidate(
                name="LogisticRegression",
                estimator_class=LogisticRegression,
                default_params={"max_iter": 1000, "random_state": random_state},
                param_distributions={
                    "C": loguniform(1e-3, 1e2),
                    "penalty": ["l2"],
                    "solver": ["lbfgs"]
                },
                fast_priority=1
            ),
            "GaussianNB": ModelCandidate(
                name="GaussianNB",
                estimator_class=GaussianNB,
                default_params={},
                param_distributions={
                    "var_smoothing": loguniform(1e-11, 1e-7)
                },
                fast_priority=1
            ),
            "KNeighborsClassifier": ModelCandidate(
                name="KNeighborsClassifier",
                estimator_class=KNeighborsClassifier,
                default_params={"n_neighbors": 5},
                param_distributions={
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "metric": ["minkowski", "manhattan"]
                },
                fast_priority=2
            ),
            "HistGradientBoostingClassifier": ModelCandidate(
                name="HistGradientBoostingClassifier",
                estimator_class=HistGradientBoostingClassifier,
                default_params={"random_state": random_state, "early_stopping": True},
                param_distributions={
                    "learning_rate": loguniform(0.01, 0.3),
                    "max_iter": [50, 100, 150, 200],
                    "max_leaf_nodes": [15, 31, 63],
                    "min_samples_leaf": [10, 20, 40],
                    "l2_regularization": [0.0, 1e-3, 1e-1, 1.0]
                },
                fast_priority=2
            ),
            "RandomForestClassifier": ModelCandidate(
                name="RandomForestClassifier",
                estimator_class=RandomForestClassifier,
                default_params={"random_state": random_state, "n_jobs": -1},
                param_distributions={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },
                fast_priority=3
            ),
            "ExtraTreesClassifier": ModelCandidate(
                name="ExtraTreesClassifier",
                estimator_class=ExtraTreesClassifier,
                default_params={"random_state": random_state, "n_jobs": -1},
                param_distributions={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                fast_priority=3
            ),
            "GradientBoostingClassifier": ModelCandidate(
                name="GradientBoostingClassifier",
                estimator_class=GradientBoostingClassifier,
                default_params={"random_state": random_state},
                param_distributions={
                    "n_estimators": [50, 100, 150],
                    "learning_rate": loguniform(0.01, 0.2),
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                },
                fast_priority=4
            ),
            "SVC": ModelCandidate(
                name="SVC",
                estimator_class=SVC,
                default_params={"random_state": random_state, "probability": True},
                param_distributions={
                    "C": loguniform(1e-2, 1e2),
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                },
                fast_priority=4
            ),
            "MLPClassifier": ModelCandidate(
                name="MLPClassifier",
                estimator_class=MLPClassifier,
                default_params={"random_state": random_state, "max_iter": 300, "early_stopping": True},
                param_distributions={
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "alpha": loguniform(1e-5, 1e-1),
                    "learning_rate_init": loguniform(1e-4, 1e-2)
                },
                fast_priority=5
            )
        }

    @staticmethod
    def get_regression_candidates(random_state: int = 42) -> Dict[str, ModelCandidate]:
        return {
            "Ridge": ModelCandidate(
                name="Ridge",
                estimator_class=Ridge,
                default_params={"random_state": random_state},
                param_distributions={
                    "alpha": loguniform(1e-3, 1e3)
                },
                fast_priority=1
            ),
            "Lasso": ModelCandidate(
                name="Lasso",
                estimator_class=Lasso,
                default_params={"random_state": random_state, "max_iter": 2000},
                param_distributions={
                    "alpha": loguniform(1e-4, 1e2)
                },
                fast_priority=1
            ),
            "ElasticNet": ModelCandidate(
                name="ElasticNet",
                estimator_class=ElasticNet,
                default_params={"random_state": random_state, "max_iter": 2000},
                param_distributions={
                    "alpha": loguniform(1e-4, 1e2),
                    "l1_ratio": uniform(0.1, 0.8)
                },
                fast_priority=1
            ),
            "KNeighborsRegressor": ModelCandidate(
                name="KNeighborsRegressor",
                estimator_class=KNeighborsRegressor,
                default_params={"n_neighbors": 5},
                param_distributions={
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "metric": ["minkowski", "manhattan"]
                },
                fast_priority=2
            ),
            "HistGradientBoostingRegressor": ModelCandidate(
                name="HistGradientBoostingRegressor",
                estimator_class=HistGradientBoostingRegressor,
                default_params={"random_state": random_state, "early_stopping": True},
                param_distributions={
                    "learning_rate": loguniform(0.01, 0.3),
                    "max_iter": [50, 100, 150, 200],
                    "max_leaf_nodes": [15, 31, 63],
                    "min_samples_leaf": [10, 20, 40],
                    "l2_regularization": [0.0, 1e-3, 1e-1, 1.0]
                },
                fast_priority=2
            ),
            "RandomForestRegressor": ModelCandidate(
                name="RandomForestRegressor",
                estimator_class=RandomForestRegressor,
                default_params={"random_state": random_state, "n_jobs": -1},
                param_distributions={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 1.0]
                },
                fast_priority=3
            ),
            "ExtraTreesRegressor": ModelCandidate(
                name="ExtraTreesRegressor",
                estimator_class=ExtraTreesRegressor,
                default_params={"random_state": random_state, "n_jobs": -1},
                param_distributions={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                fast_priority=3
            ),
            "GradientBoostingRegressor": ModelCandidate(
                name="GradientBoostingRegressor",
                estimator_class=GradientBoostingRegressor,
                default_params={"random_state": random_state},
                param_distributions={
                    "n_estimators": [50, 100, 150],
                    "learning_rate": loguniform(0.01, 0.2),
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                },
                fast_priority=4
            ),
            "SVR": ModelCandidate(
                name="SVR",
                estimator_class=SVR,
                default_params={},
                param_distributions={
                    "C": loguniform(1e-2, 1e2),
                    "kernel": ["linear", "rbf"],
                    "epsilon": [0.01, 0.1, 0.2]
                },
                fast_priority=4
            ),
            "MLPRegressor": ModelCandidate(
                name="MLPRegressor",
                estimator_class=MLPRegressor,
                default_params={"random_state": random_state, "max_iter": 300, "early_stopping": True},
                param_distributions={
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "alpha": loguniform(1e-5, 1e-1),
                    "learning_rate_init": loguniform(1e-4, 1e-2)
                },
                fast_priority=5
            )
        }

    @classmethod
    def get_candidates(
        cls,
        task: str,
        include_models: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        random_state: int = 42
    ) -> List[ModelCandidate]:
        if task == "classification":
            candidates_dict = cls.get_classification_candidates(random_state)
        elif task == "regression":
            candidates_dict = cls.get_regression_candidates(random_state)
        else:
            raise ValueError(f"Unknown task type: {task}. Must be 'classification' or 'regression'.")

        if include_models:
            candidates_dict = {k: v for k, v in candidates_dict.items() if k in include_models}
        if exclude_models:
            candidates_dict = {k: v for k, v in candidates_dict.items() if k not in exclude_models}

        # Sort by fast_priority so quick models run first
        sorted_candidates = sorted(candidates_dict.values(), key=lambda c: c.fast_priority)
        return sorted_candidates
