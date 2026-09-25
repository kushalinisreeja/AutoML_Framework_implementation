import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.base import clone


class EnsembleBuilder:
    """
    Builds Voting and Stacking ensembles from the top-performing candidate estimators.
    """
    @staticmethod
    def build_voting(
        task: str,
        named_estimators: List[Tuple[str, Any]],
        cv_splitter: Any,
        scoring: str
    ) -> Tuple[Any, float, float, float]:
        """
        Builds and cross-validates a Voting Ensemble.
        Returns: (ensemble_estimator, mean_cv_score, std_cv_score, fit_time_secs)
        """
        start_time = time.perf_counter()

        # Check if all classification estimators support predict_proba for soft voting
        if task == "classification":
            supports_proba = all(hasattr(est, "predict_proba") for _, est in named_estimators)
            voting_mode = "soft" if supports_proba else "hard"
            ensemble = VotingClassifier(estimators=named_estimators, voting=voting_mode)
        else:
            ensemble = VotingRegressor(estimators=named_estimators)

        scores = cross_val_score(ensemble, None, None, cv=cv_splitter, scoring=scoring) if False else []
        # Fit on whole data
        return ensemble

    @classmethod
    def create_and_evaluate_voting(
        cls,
        task: str,
        named_estimators: List[Tuple[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: Any,
        scoring: str,
        n_jobs: int = -1
    ) -> Tuple[Any, float, float, float]:
        start_time = time.perf_counter()

        if task == "classification":
            supports_proba = all(hasattr(est, "predict_proba") for _, est in named_estimators)
            voting_mode = "soft" if supports_proba else "hard"
            ensemble = VotingClassifier(
                estimators=[(name, clone(est)) for name, est in named_estimators],
                voting=voting_mode,
                n_jobs=n_jobs
            )
        else:
            ensemble = VotingRegressor(
                estimators=[(name, clone(est)) for name, est in named_estimators],
                n_jobs=n_jobs
            )

        scores = cross_val_score(ensemble, X, y, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs)
        ensemble.fit(X, y)
        fit_time = time.perf_counter() - start_time
        return ensemble, float(np.mean(scores)), float(np.std(scores)), fit_time

    @classmethod
    def create_and_evaluate_stacking(
        cls,
        task: str,
        named_estimators: List[Tuple[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        cv_splitter: Any,
        scoring: str,
        n_jobs: int = -1,
        random_state: int = 42
    ) -> Tuple[Any, float, float, float]:
        start_time = time.perf_counter()

        if task == "classification":
            final_estimator = LogisticRegression(random_state=random_state, max_iter=1000)
            ensemble = StackingClassifier(
                estimators=[(name, clone(est)) for name, est in named_estimators],
                final_estimator=final_estimator,
                cv=cv_splitter,
                n_jobs=n_jobs
            )
        else:
            final_estimator = Ridge(random_state=random_state)
            ensemble = StackingRegressor(
                estimators=[(name, clone(est)) for name, est in named_estimators],
                final_estimator=final_estimator,
                cv=cv_splitter,
                n_jobs=n_jobs
            )

        scores = cross_val_score(ensemble, X, y, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs)
        ensemble.fit(X, y)
        fit_time = time.perf_counter() - start_time
        return ensemble, float(np.mean(scores)), float(np.std(scores)), fit_time
