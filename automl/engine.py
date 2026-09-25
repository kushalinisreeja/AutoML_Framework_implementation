import time
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from automl.config import AutoMLConfig
from automl.preprocessing.cleaner import TaskDetector, DataPreprocessor
from automl.feature_engineering.engineer import FeatureSelector
from automl.models.registry import ModelRegistry, ModelCandidate
from automl.optimization.tuner import ModelTuner
from automl.ensemble.stacking import EnsembleBuilder
from automl.evaluation.metrics import Evaluator, Leaderboard
from automl.explainability.importance import Explainer


class InferencePipeline(BaseEstimator):
    """
    Production-ready end-to-end inference pipeline bundling
    data preprocessing, feature selection, and the winning model.
    """
    def __init__(self, preprocessor: DataPreprocessor, selector: FeatureSelector, model: Any):
        self.preprocessor = preprocessor
        self.selector = selector
        self.model = model

    def transform_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_prep = self.preprocessor.transform(X)
        X_sel = self.selector.transform(X_prep)
        return X_sel

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_trans = self.transform_features(X)
        return self.model.predict(X_trans)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The winning model does not support predict_proba.")
        X_trans = self.transform_features(X)
        return self.model.predict_proba(X_trans)


class AutoML(BaseEstimator):
    """
    Unified Automated Machine Learning Engine for Tabular Data.
    Automatically handles task detection, preprocessing, feature selection,
    multi-model benchmarking, hyperparameter tuning, ensembling, and evaluation.
    """
    def __init__(
        self,
        task: str = "auto",
        primary_metric: Optional[str] = None,
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        time_budget_secs: Optional[int] = 300,
        n_iter_per_model: int = 8,
        scale_numeric: bool = True,
        scaling_method: str = "standard",
        encode_categorical: bool = True,
        max_one_hot_cardinality: int = 15,
        extract_datetime: bool = True,
        feature_selection: bool = True,
        include_models: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        ensemble: bool = True,
        ensemble_type: str = "both",
        top_k_models: int = 3,
        verbose: int = 1
    ):
        self.task = task
        self.primary_metric = primary_metric
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.time_budget_secs = time_budget_secs
        self.n_iter_per_model = n_iter_per_model
        self.scale_numeric = scale_numeric
        self.scaling_method = scaling_method
        self.encode_categorical = encode_categorical
        self.max_one_hot_cardinality = max_one_hot_cardinality
        self.extract_datetime = extract_datetime
        self.feature_selection = feature_selection
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.ensemble = ensemble
        self.ensemble_type = ensemble_type
        self.top_k_models = top_k_models
        self.verbose = verbose

        # Fitted attributes
        self.task_info_: Dict[str, Any] = {}
        self.preprocessor_: Optional[DataPreprocessor] = None
        self.selector_: Optional[FeatureSelector] = None
        self.leaderboard_: Optional[Leaderboard] = None
        self.best_model_: Any = None
        self.best_model_name_: str = ""
        self.models_fitted_: Dict[str, Any] = {}
        self.pipeline_: Optional[InferencePipeline] = None
        self.validation_metrics_: Dict[str, float] = {}
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.training_time_secs_: float = 0.0

    def _log(self, message: str, level: int = 1):
        if self.verbose >= level:
            print(f"[AutoML] {message}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray, list]):
        start_overall = time.perf_counter()

        # Step 1: Detect or validate task
        if isinstance(y, (list, np.ndarray)):
            y_series = pd.Series(y)
        else:
            y_series = y

        self.task_info_ = TaskDetector.detect_task(y_series)
        detected_task = self.task_info_["task"]
        if self.task == "auto":
            self.task = detected_task
        elif self.task != detected_task:
            self._log(f"Warning: Specified task '{self.task}' differs from detected task '{detected_task}'. Proceeding with specified task.")

        is_binary = self.task_info_.get("classification_type") == "binary"

        # Determine primary optimization metric
        if self.primary_metric is None:
            self.primary_metric = Evaluator.get_default_metric(self.task, is_binary=is_binary)

        higher_is_better = Evaluator.is_higher_better(self.primary_metric)
        self.leaderboard_ = Leaderboard(primary_metric=self.primary_metric, higher_is_better=higher_is_better)

        self._log(f"Initiating AutoML execution on {len(X)} samples with task: {self.task.upper()}")
        self._log(f"Primary optimization metric: {self.primary_metric} (Higher is better: {higher_is_better})")

        # Step 2: Validation split
        stratify = y_series if self.task == "classification" and not self.task_info_.get("is_imbalanced") else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_series,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify
            )
        except Exception:
            # Fallback without stratification if any class has too few instances
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_series,
                test_size=self.test_size,
                random_state=self.random_state
            )

        # Step 3: Data Preprocessing
        self._log("Fitting automated data preprocessing pipeline...")
        self.preprocessor_ = DataPreprocessor(
            scale_numeric=self.scale_numeric,
            scaling_method=self.scaling_method,
            encode_categorical=self.encode_categorical,
            max_one_hot_cardinality=self.max_one_hot_cardinality,
            extract_datetime=self.extract_datetime
        )
        X_train_prep = self.preprocessor_.fit_transform(X_train)
        X_val_prep = self.preprocessor_.transform(X_val)

        prep_feature_names = self.preprocessor_.get_feature_names_out()
        self._log(f"Preprocessed features: {X_train_prep.shape[1]} input dimensions.")

        # Step 4: Feature Engineering & Selection
        self._log("Performing feature selection & ranking...")
        self.selector_ = FeatureSelector(
            task=self.task,
            enable_selection=self.feature_selection
        )
        # Pass feature names as DataFrame columns so selector preserves them
        df_train_prep = pd.DataFrame(X_train_prep, columns=prep_feature_names)
        df_val_prep = pd.DataFrame(X_val_prep, columns=prep_feature_names)

        X_train_sel = self.selector_.fit_transform(df_train_prep, y_train)
        X_val_sel = self.selector_.transform(df_val_prep)

        final_feature_names = self.selector_.get_feature_names_out()
        self._log(f"Selected {len(final_feature_names)} features for model training.")

        # Step 5: Cross-Validation Splitter
        if self.task == "classification":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scoring_map = {
                "accuracy": "accuracy",
                "f1": "f1" if is_binary else "f1_weighted",
                "f1_weighted": "f1_weighted",
                "f1_macro": "f1_macro",
                "roc_auc": "roc_auc" if is_binary else "roc_auc_ovr_weighted",
                "precision": "precision_weighted",
                "recall": "recall_weighted"
            }
            scoring_key = scoring_map.get(self.primary_metric, "accuracy")
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scoring_map = {
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "mse": "neg_mean_squared_error",
                "mae": "neg_mean_absolute_error"
            }
            scoring_key = scoring_map.get(self.primary_metric, "r2")

        # Step 6: Candidate Models Benchmarking & Tuning
        candidates = ModelRegistry.get_candidates(
            task=self.task,
            include_models=self.include_models,
            exclude_models=self.exclude_models,
            random_state=self.random_state
        )

        self._log(f"Benchmarking {len(candidates)} candidate algorithms...")
        time_budget = self.time_budget_secs
        
        y_train_arr = np.asarray(y_train)

        for candidate in candidates:
            elapsed = time.perf_counter() - start_overall
            remaining = (time_budget - elapsed) if time_budget else None

            if remaining is not None and remaining <= 0:
                self._log("Time budget exhausted. Concluding individual candidate search.")
                break

            self._log(f"Training & tuning candidate: {candidate.name}...")
            tuner = ModelTuner(
                candidate=candidate,
                cv_splitter=cv,
                scoring=scoring_key,
                n_iter=self.n_iter_per_model,
                random_state=self.random_state
            )

            try:
                fitted_est, mean_score, std_score, best_params, fit_time = tuner.tune(
                    X_train_sel, y_train_arr, remaining_time_secs=remaining
                )
                
                # Invert negative regression scores for display if needed
                display_score = mean_score
                if scoring_key.startswith("neg_"):
                    display_score = abs(mean_score)

                self.models_fitted_[candidate.name] = fitted_est
                self.leaderboard_.add_entry(
                    model_name=candidate.name,
                    mean_score=display_score,
                    std_score=std_score,
                    fit_time_secs=fit_time,
                    best_params=best_params,
                    is_ensemble=False
                )
                self._log(f"  -> {candidate.name}: Score = {display_score:.4f} (±{std_score:.4f}) in {fit_time:.2f}s")
            except Exception as e:
                self._log(f"  -> Error tuning {candidate.name}: {str(e)}", level=2)

        # Step 7: Ensembling
        if self.ensemble and len(self.models_fitted_) >= 2:
            self._log("Generating ensemble models from top candidates...")
            lb_df = self.leaderboard_.to_dataframe()
            # Select top K distinct candidate models (excluding any previous ensembles)
            top_candidate_names = lb_df[~lb_df["is_ensemble"]]["model"].head(self.top_k_models).tolist()
            top_estimators = [(name, self.models_fitted_[name]) for name in top_candidate_names]

            if len(top_estimators) >= 2:
                # 1. Voting Ensemble
                if self.ensemble_type in ["voting", "both"]:
                    try:
                        self._log(f"Constructing Voting Ensemble from: {top_candidate_names}...")
                        v_model, v_score, v_std, v_time = EnsembleBuilder.create_and_evaluate_voting(
                            task=self.task,
                            named_estimators=top_estimators,
                            X=X_train_sel,
                            y=y_train_arr,
                            cv_splitter=cv,
                            scoring=scoring_key
                        )
                        v_disp_score = abs(v_score) if scoring_key.startswith("neg_") else v_score
                        self.models_fitted_["VotingEnsemble"] = v_model
                        self.leaderboard_.add_entry(
                            model_name="VotingEnsemble",
                            mean_score=v_disp_score,
                            std_score=v_std,
                            fit_time_secs=v_time,
                            best_params={"base_models": top_candidate_names},
                            is_ensemble=True
                        )
                        self._log(f"  -> VotingEnsemble: Score = {v_disp_score:.4f} (±{v_std:.4f})")
                    except Exception as e:
                        self._log(f"Voting ensemble failed: {str(e)}", level=2)

                # 2. Stacking Ensemble
                if self.ensemble_type in ["stacking", "both"]:
                    try:
                        self._log(f"Constructing Stacking Ensemble from: {top_candidate_names}...")
                        s_model, s_score, s_std, s_time = EnsembleBuilder.create_and_evaluate_stacking(
                            task=self.task,
                            named_estimators=top_estimators,
                            X=X_train_sel,
                            y=y_train_arr,
                            cv_splitter=cv,
                            scoring=scoring_key,
                            random_state=self.random_state
                        )
                        s_disp_score = abs(s_score) if scoring_key.startswith("neg_") else s_score
                        self.models_fitted_["StackingEnsemble"] = s_model
                        self.leaderboard_.add_entry(
                            model_name="StackingEnsemble",
                            mean_score=s_disp_score,
                            std_score=s_std,
                            fit_time_secs=s_time,
                            best_params={"base_models": top_candidate_names},
                            is_ensemble=True
                        )
                        self._log(f"  -> StackingEnsemble: Score = {s_disp_score:.4f} (±{s_std:.4f})")
                    except Exception as e:
                        self._log(f"Stacking ensemble failed: {str(e)}", level=2)

        # Step 8: Select Champion Model & Build Inference Pipeline
        self.best_model_name_ = self.leaderboard_.get_best_model_name()
        if not self.best_model_name_ or self.best_model_name_ not in self.models_fitted_:
            raise RuntimeError("AutoML failed to train any candidate models successfully.")

        self.best_model_ = self.models_fitted_[self.best_model_name_]
        self._log(f"Champion Model selected: {self.best_model_name_}")

        # Assemble unified inference pipeline
        self.pipeline_ = InferencePipeline(
            preprocessor=self.preprocessor_,
            selector=self.selector_,
            model=self.best_model_
        )

        # Step 9: Final Validation Evaluation on Holdout Set
        y_val_arr = np.asarray(y_val)
        y_val_pred = self.best_model_.predict(X_val_sel)
        y_val_proba = None
        if self.task == "classification" and hasattr(self.best_model_, "predict_proba"):
            try:
                y_val_proba = self.best_model_.predict_proba(X_val_sel)
            except Exception:
                y_val_proba = None

        self.validation_metrics_ = Evaluator.evaluate(
            task=self.task,
            y_true=y_val_arr,
            y_pred=y_val_pred,
            y_proba=y_val_proba
        )

        # Step 10: Compute Feature Explainability
        self._log("Computing model explainability & feature importances...")
        self.feature_importance_ = Explainer.get_feature_importance(
            model=self.best_model_,
            X_val=X_val_sel,
            y_val=y_val_arr,
            feature_names=final_feature_names,
            random_state=self.random_state
        )

        self.training_time_secs_ = round(time.perf_counter() - start_overall, 2)
        self._log(f"AutoML training successfully completed in {self.training_time_secs_} seconds.")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self.pipeline_.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("AutoML instance is not fitted yet. Call fit() first.")
        return self.pipeline_.predict_proba(X)

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray, list]) -> Dict[str, float]:
        y_pred = self.predict(X)
        y_proba = None
        if self.task == "classification" and hasattr(self.pipeline_.model, "predict_proba"):
            try:
                y_proba = self.predict_proba(X)
            except Exception:
                pass
        return Evaluator.evaluate(self.task, np.asarray(y), y_pred, y_proba)

    def leaderboard(self) -> pd.DataFrame:
        if self.leaderboard_ is None:
            return pd.DataFrame()
        return self.leaderboard_.to_dataframe()

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        if self.feature_importance_ is None:
            return pd.DataFrame()
        if top_n is not None:
            return self.feature_importance_.head(top_n)
        return self.feature_importance_

    def save(self, filepath: str):
        """
        Saves the fitted AutoML model pipeline to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        self._log(f"AutoML model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AutoML":
        """
        Loads a saved AutoML instance from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        loaded = joblib.load(filepath)
        if not isinstance(loaded, AutoML):
            raise TypeError(f"Loaded object is not an AutoML instance: {type(loaded)}")
        return loaded


class AutoMLClassifier(AutoML, ClassifierMixin):
    """Convenience class configured explicitly for classification tasks."""
    def __init__(self, **kwargs):
        super().__init__(task="classification", **kwargs)


class AutoMLRegressor(AutoML, RegressorMixin):
    """Convenience class configured explicitly for regression tasks."""
    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)
