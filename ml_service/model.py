import threading
from typing import NamedTuple

from sklearn.pipeline import Pipeline

from ml_service.mlflow_utils import load_model


class ModelData(NamedTuple):
    model: Pipeline | None
    run_id: str | None


class Model:
    """
    Thread-safe container for the currently active model.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData(model=None, run_id=None)

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set(self, run_id: str) -> None:
        model = load_model(run_id=run_id)
        with self.lock:
            self.data = ModelData(model=model, run_id=run_id)

    @property
    def features(self) -> list[str]:
        return self.data.model.feature_names_in_
