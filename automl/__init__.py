"""
AutoML Framework Implementation
An end-to-end automated machine learning framework in Python for tabular data.
"""

from automl.engine import AutoML, AutoMLClassifier, AutoMLRegressor, InferencePipeline
from automl.config import AutoMLConfig
from automl.preprocessing.cleaner import TaskDetector, DataPreprocessor
from automl.feature_engineering.engineer import FeatureSelector
from automl.models.registry import ModelRegistry
from automl.evaluation.metrics import Evaluator, Leaderboard
from automl.explainability.importance import Explainer

__version__ = "1.0.0"
__all__ = [
    "AutoML",
    "AutoMLClassifier",
    "AutoMLRegressor",
    "InferencePipeline",
    "AutoMLConfig",
    "TaskDetector",
    "DataPreprocessor",
    "FeatureSelector",
    "ModelRegistry",
    "Evaluator",
    "Leaderboard",
    "Explainer"
]
