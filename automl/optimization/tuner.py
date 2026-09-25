import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone

from automl.models.registry import ModelCandidate


class ModelTuner:
    """
    Tunes hyperparameters for a single ModelCandidate using cross-validated random search.
    Handles timeout budgets and fallback defaults.
    """
    def __init__(
        self,
        candidate: ModelCandidate,
        cv_splitter: Any,
        scoring: str,
        n_iter: int = 10,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.candidate = candidate
        self.cv_splitter = cv_splitter
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        remaining_time_secs: Optional[float] = None
    ) -> Tuple[Any, float, float, Dict[str, Any], float]:
        """
        Executes tuning. Returns:
        (best_estimator, mean_cv_score, std_cv_score, best_params, fit_time_secs)
        """
        start_time = time.perf_counter()

        # If time is nearly exhausted (< 5s), just fit default estimator
        if remaining_time_secs is not None and remaining_time_secs < 5.0:
            estimator = self.candidate.estimator_class(**self.candidate.default_params)
            estimator.fit(X, y)
            fit_time = time.perf_counter() - start_time
            return estimator, 0.0, 0.0, self.candidate.default_params, fit_time

        # Base estimator
        base_estimator = self.candidate.estimator_class(**self.candidate.default_params)

        # If model has no hyperparameter distribution to sample from, do simple CV fit
        if not self.candidate.param_distributions or self.n_iter <= 1:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(base_estimator, X, y, cv=self.cv_splitter, scoring=self.scoring, n_jobs=self.n_jobs)
            base_estimator.fit(X, y)
            fit_time = time.perf_counter() - start_time
            return base_estimator, float(np.mean(scores)), float(np.std(scores)), self.candidate.default_params, fit_time

        # Adjust n_iter if search space is smaller than n_iter
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=self.candidate.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv_splitter,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            refit=True,
            error_score="raise"
        )

        try:
            search.fit(X, y)
            fit_time = time.perf_counter() - start_time
            best_idx = search.best_index_
            mean_score = float(search.cv_results_["mean_test_score"][best_idx])
            std_score = float(search.cv_results_["std_test_score"][best_idx])
            return search.best_estimator_, mean_score, std_score, search.best_params_, fit_time
        except Exception:
            # Fallback to default params fit if random search encounters numerical instability
            fallback = clone(base_estimator)
            fallback.fit(X, y)
            fit_time = time.perf_counter() - start_time
            return fallback, -999.0, 0.0, self.candidate.default_params, fit_time
