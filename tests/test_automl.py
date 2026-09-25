import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes

from automl import AutoML, AutoMLClassifier, AutoMLRegressor, TaskDetector, DataPreprocessor


class TestAutoMLFramework(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_task_detector_classification(self):
        y_binary = pd.Series([0, 1, 0, 1, 1, 0])
        info = TaskDetector.detect_task(y_binary)
        self.assertEqual(info["task"], "classification")
        self.assertEqual(info["classification_type"], "binary")

        y_str = pd.Series(["cat", "dog", "bird", "dog"])
        info_str = TaskDetector.detect_task(y_str)
        self.assertEqual(info_str["task"], "classification")

    def test_task_detector_regression(self):
        y_cont = pd.Series(np.linspace(10.5, 99.8, 100))
        info = TaskDetector.detect_task(y_cont)
        self.assertEqual(info["task"], "regression")

    def test_preprocessor(self):
        df = pd.DataFrame({
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "cat1": ["a", "b", "a", np.nan, "b"],
            "date1": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
            "constant_col": [1, 1, 1, 1, 1],
            "id": [101, 102, 103, 104, 105]
        })
        prep = DataPreprocessor()
        X_trans = prep.fit_transform(df)
        self.assertGreater(X_trans.shape[1], 0)
        self.assertIn("constant_col", prep.dropped_cols_)
        self.assertIn("id", prep.dropped_cols_)

    def test_classification_end_to_end(self):
        # Quick run on Iris dataset
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target

        automl = AutoMLClassifier(
            include_models=["LogisticRegression", "GaussianNB"],
            n_iter_per_model=2,
            n_splits=3,
            ensemble=True,
            ensemble_type="voting",
            verbose=0
        )
        automl.fit(X, y)

        # Assertions
        self.assertIsNotNone(automl.best_model_)
        preds = automl.predict(X)
        self.assertEqual(len(preds), len(X))

        proba = automl.predict_proba(X)
        self.assertEqual(proba.shape[0], len(X))
        self.assertEqual(proba.shape[1], 3)

        lb = automl.leaderboard()
        self.assertFalse(lb.empty)
        self.assertIn("VotingEnsemble", lb["model"].values)

        fi = automl.get_feature_importance()
        self.assertFalse(fi.empty)

        # Test model persistence & reloading
        save_path = os.path.join(self.temp_dir, "test_model.joblib")
        automl.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded = AutoML.load(save_path)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(preds, loaded_preds)

    def test_regression_end_to_end(self):
        # Quick run on Diabetes dataset
        diabetes = load_diabetes(as_frame=True)
        X = diabetes.data.iloc[:80]
        y = diabetes.target.iloc[:80]

        automl = AutoMLRegressor(
            include_models=["Ridge", "KNeighborsRegressor"],
            n_iter_per_model=2,
            n_splits=3,
            ensemble=True,
            ensemble_type="voting",
            verbose=0
        )
        automl.fit(X, y)

        self.assertIsNotNone(automl.best_model_)
        preds = automl.predict(X)
        self.assertEqual(len(preds), len(X))

        scores = automl.score(X, y)
        self.assertIn("r2", scores)
        self.assertIn("rmse", scores)


if __name__ == "__main__":
    unittest.main()
