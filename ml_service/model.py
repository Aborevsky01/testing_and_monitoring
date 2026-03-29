import threading
from typing import NamedTuple

import numpy as np
from sklearn.pipeline import Pipeline

from ml_service.mlflow_utils import load_model


class ModelData(NamedTuple):
    model: Pipeline | None
    run_id: str | None
    features: tuple[str, ...]
    model_type: str | None
    error: str | None


class ModelError(RuntimeError):
    """
    Base exception for model lifecycle and inference errors.
    """


class ModelLoadError(ModelError):
    """
    Raised when a model cannot be loaded or validated.
    """


class ModelInferenceError(ModelError):
    """
    Raised when model inference returns an invalid result.
    """


class Model:
    """
    Thread-safe container for the currently active model.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData(
            model=None,
            run_id=None,
            features=tuple(),
            model_type=None,
            error='Model is not loaded yet',
        )

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set_unavailable(self, error: str) -> None:
        with self.lock:
            self.data = ModelData(
                model=None,
                run_id=None,
                features=tuple(),
                model_type=None,
                error=error,
            )

    def set(self, run_id: str) -> None:
        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ModelLoadError('run_id must not be empty')

        try:
            model = load_model(run_id=normalized_run_id)
        except Exception as exc:
            raise ModelLoadError(
                f'Failed to load model for run_id={normalized_run_id}'
            ) from exc

        feature_names = getattr(model, 'feature_names_in_', None)
        if feature_names is None:
            raise ModelLoadError('Loaded model does not expose feature_names_in_')

        features = tuple(str(name) for name in feature_names)
        if not features:
            raise ModelLoadError('Loaded model contains an empty feature list')

        if not hasattr(model, 'predict_proba'):
            raise ModelLoadError('Loaded model does not support predict_proba')

        model_type = (
            type(model[-1]).__name__
            if hasattr(model, '__getitem__')
            else type(model).__name__
        )

        with self.lock:
            self.data = ModelData(
                model=model,
                run_id=normalized_run_id,
                features=features,
                model_type=model_type,
                error=None,
            )

    def predict(self, dataframe) -> tuple[int, float]:
        state = self.get()
        model = state.model
        if model is None:
            raise ModelInferenceError(state.error or 'Model is not loaded yet')

        try:
            probabilities = model.predict_proba(dataframe)
        except Exception as exc:
            raise ModelInferenceError('Model inference failed') from exc

        probabilities = np.asarray(probabilities)
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != 1
            or probabilities.shape[1] < 2
        ):
            raise ModelInferenceError('Model returned probabilities with unexpected shape')

        probability = float(probabilities[0][1])
        if not np.isfinite(probability):
            raise ModelInferenceError('Model returned a non-finite probability')

        prediction = int(probability >= 0.5)
        return prediction, probability

    @property
    def features(self) -> list[str]:
        with self.lock:
            return list(self.data.features)
