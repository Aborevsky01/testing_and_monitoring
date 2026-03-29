import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from ml_service.app import MODEL, create_app
from ml_service.model import ModelLoadError
from ml_service.schemas import PredictRequest, UpdateModelRequest


class FakeEstimator:
    pass


class FakePipeline:
    def __init__(self, features, probabilities=None) -> None:
        self.feature_names_in_ = np.array(features, dtype=object)
        self._probabilities = probabilities or [[0.2, 0.8]]
        self._estimator = FakeEstimator()

    def __getitem__(self, index: int):
        if index != -1:
            raise IndexError(index)
        return self._estimator

    def predict_proba(self, dataframe):
        return np.array(self._probabilities)


class AppHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        MODEL.set_unavailable('Model is not loaded yet')
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self.client.close()
        MODEL.set_unavailable('Model is not loaded yet')

    def test_health_returns_degraded_when_model_is_unavailable(self) -> None:
        response = self.client.get('/health')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'degraded')

    def test_health_returns_model_metadata_after_successful_load(self) -> None:
        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race', 'sex']),
        ):
            MODEL.set('run-123')

        response = self.client.get('/health')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            'status': 'ok',
            'run_id': 'run-123',
            'model_type': 'FakeEstimator',
            'features': ['race', 'sex'],
            'error': None,
        })

    def test_predict_returns_503_without_loaded_model(self) -> None:
        response = self.client.post('/predict', json={'race': 'White'})

        self.assertEqual(response.status_code, 503)
        self.assertIn('Model is not loaded', response.json()['detail'])

    def test_predict_returns_422_when_required_feature_is_missing(self) -> None:
        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race', 'sex']),
        ):
            MODEL.set('run-123')

        response = self.client.post('/predict', json={'race': 'White'})

        self.assertEqual(response.status_code, 422)
        self.assertIn('sex', response.json()['detail'])

    def test_predict_rejects_invalid_request_payload(self) -> None:
        response = self.client.post(
            '/predict',
            json={'age': -1, 'unexpected': 'field'},
        )

        self.assertEqual(response.status_code, 422)

    def test_update_model_returns_400_for_invalid_run_id(self) -> None:
        with patch(
            'ml_service.app.MODEL.set',
            side_effect=ModelLoadError('broken'),
        ):
            response = self.client.post('/updateModel', json={'run_id': 'missing'})

        self.assertEqual(response.status_code, 400)
        self.assertIn('broken', response.json()['detail'])

    def test_service_smoke_predicts_successfully(self) -> None:
        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race', 'sex'], probabilities=[[0.35, 0.65]]),
        ):
            MODEL.set('run-123')

        response = self.client.post(
            '/predict',
            json={'race': 'White', 'sex': 'Male'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['prediction'], 1)
        self.assertAlmostEqual(response.json()['probability'], 0.65)

    def test_metrics_endpoint_exposes_custom_metrics_after_requests(self) -> None:
        with patch(
            'ml_service.model.load_model',
            return_value=FakePipeline(features=['race', 'sex'], probabilities=[[0.35, 0.65]]),
        ):
            MODEL.set('run-123')

        self.client.get('/health')
        self.client.post('/predict', json={'race': 'White', 'sex': 'Male'})
        response = self.client.get('/metrics')

        self.assertEqual(response.status_code, 200)
        self.assertIn('bor_ml_service_requests_total', response.text)
        self.assertIn('bor_ml_service_request_duration_seconds_bucket', response.text)
        self.assertIn('bor_ml_service_preprocess_duration_seconds_bucket', response.text)
        self.assertIn('bor_ml_service_inference_duration_seconds_bucket', response.text)
        self.assertIn('bor_ml_service_model_probability_bucket', response.text)
        self.assertIn('bor_ml_service_current_model_info', response.text)


class LifespanTests(unittest.TestCase):
    def tearDown(self) -> None:
        MODEL.set_unavailable('Model is not loaded yet')

    def test_app_starts_in_degraded_mode_when_default_model_load_fails(self) -> None:
        with patch('ml_service.app.configure_mlflow'), patch(
            'ml_service.app.config.default_run_id',
            return_value='broken-run',
        ), patch(
            'ml_service.app.MODEL.set',
            side_effect=ModelLoadError('startup failure'),
        ):
            with TestClient(create_app()) as client:
                response = client.get('/health')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'degraded')
        self.assertIn('startup failure', response.json()['error'])


class SchemaValidationTests(unittest.TestCase):
    def test_predict_request_strips_whitespace_and_accepts_aliases(self) -> None:
        request = PredictRequest(
            race=' White ',
            sex=' Male ',
            **{'native.country': ' United-States '},
        )

        self.assertEqual(request.race, 'White')
        self.assertEqual(request.sex, 'Male')
        self.assertEqual(request.native_country, 'United-States')

    def test_update_model_request_strips_run_id(self) -> None:
        request = UpdateModelRequest(run_id='  run-123  ')

        self.assertEqual(request.run_id, 'run-123')
