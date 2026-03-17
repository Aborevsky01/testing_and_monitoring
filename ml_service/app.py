from typing import Any
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


MODEL = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    configure_mlflow()
    run_id = config.default_run_id()
    MODEL.set(run_id=run_id)
    yield
    # add any teardown logic here if needed


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model = MODEL.get().model
        if model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        df = to_dataframe(request, needed_columns=MODEL.features)

        probability = model.predict_proba(df)[0][1]
        prediction = int(probability >= 0.5)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        MODEL.set(run_id=run_id)
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
