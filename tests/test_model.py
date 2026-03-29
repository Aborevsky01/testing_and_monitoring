import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_service.model import Model, ModelInferenceError, ModelLoadError


class FakeEstimator:
    pass


class FakePipeline:
    def __init__(
        self,
        features,
        probabilities=None,
        should_fail=False,
    ) -> None:
        self.feature_names_in_ = np.array(features, dtype=object)
        self._probabilities = probabilities or [[0.2, 0.8]]
        self._should_fail = should_fail
        self._estimator = FakeEstimator()

    def __getitem__(self, index: int):
        if index != -1:
            raise IndexError(index)
        return self._estimator

    def predict_proba(self, dataframe: pd.DataFrame):
        if self._should_fail:
            raise RuntimeError('boom')
        return np.array(self._probabilities)


class ModelTests(unittest.TestCase):
    def test_set_loads_model_and_exposes_metadata(self) -> None:
        container = Model()
        pipeline = FakePipeline(features=['race', 'sex'])

        with patch('ml_service.model.load_model', return_value=pipeline):
            container.set('run-123')

        state = container.get()
        self.assertIs(state.model, pipeline)
        self.assertEqual(state.run_id, 'run-123')
        self.assertEqual(state.features, ('race', 'sex'))
        self.assertEqual(state.model_type, 'FakeEstimator')
        self.assertIsNone(state.error)

    def test_failed_update_does_not_override_previous_model(self) -> None:
        container = Model()

        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race', 'sex']),
        ):
            container.set('stable-run')

        with patch(
            'ml_service.model.load_model',
            side_effect=RuntimeError('not found'),
        ):
            with self.assertRaises(ModelLoadError):
                container.set('broken-run')

        state = container.get()
        self.assertEqual(state.run_id, 'stable-run')
        self.assertEqual(state.features, ('race', 'sex'))
        self.assertIsNotNone(state.model)

    def test_rejects_blank_run_id(self) -> None:
        container = Model()

        with self.assertRaisesRegex(ModelLoadError, 'must not be empty'):
            container.set('   ')

    def test_predict_returns_class_and_probability(self) -> None:
        container = Model()
        dataframe = pd.DataFrame([['White', 'Male']], columns=['race', 'sex'])

        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race', 'sex'], probabilities=[[0.1, 0.9]]),
        ):
            container.set('run-123')

        prediction, probability = container.predict(dataframe)

        self.assertEqual(prediction, 1)
        self.assertAlmostEqual(probability, 0.9)

    def test_predict_raises_on_bad_probability_shape(self) -> None:
        container = Model()
        dataframe = pd.DataFrame([['White']], columns=['race'])

        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race'], probabilities=[0.9]),
        ):
            container.set('run-123')

        with self.assertRaises(ModelInferenceError):
            container.predict(dataframe)

    def test_predict_raises_when_model_predict_proba_fails(self) -> None:
        container = Model()
        dataframe = pd.DataFrame([['White']], columns=['race'])

        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race'], should_fail=True),
        ):
            container.set('run-123')

        with self.assertRaisesRegex(ModelInferenceError, 'Model inference failed'):
            container.predict(dataframe)

    def test_set_rejects_model_without_feature_names(self) -> None:
        container = Model()

        class NoFeatureNamesModel:
            def predict_proba(self, dataframe):
                return np.array([[0.4, 0.6]])

        with patch(
            'ml_service.model.load_model',
            return_value=NoFeatureNamesModel(),
        ):
            with self.assertRaisesRegex(
                ModelLoadError,
                'feature_names_in_',
            ):
                container.set('run-123')

    def test_set_rejects_model_without_predict_proba(self) -> None:
        container = Model()

        class NoPredictProbaModel:
            feature_names_in_ = np.array(['race'], dtype=object)

        with patch(
            'ml_service.model.load_model',
            return_value=NoPredictProbaModel(),
        ):
            with self.assertRaisesRegex(
                ModelLoadError,
                'predict_proba',
            ):
                container.set('run-123')
