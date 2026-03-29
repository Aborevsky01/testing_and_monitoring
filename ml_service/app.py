from typing import Any
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException
from fastapi import Request

from ml_service import config
from ml_service.features import (
    FeatureSchemaError,
    FeatureValidationError,
    to_dataframe,
)
from ml_service.metrics import (
    IN_PROGRESS_REQUESTS,
    RequestTimer,
    metrics_response,
    observe_features,
    observe_inference,
    observe_prediction,
    observe_preprocess,
    observe_request,
    record_model_update,
    update_model_metrics,
)
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model, ModelInferenceError, ModelLoadError
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
    try:
        configure_mlflow()
        run_id = config.default_run_id()
        MODEL.set(run_id=run_id)
        state = MODEL.get()
        update_model_metrics(
            run_id=state.run_id,
            model_type=state.model_type,
            features=state.features,
            available=True,
            touched_at=time.time(),
        )
    except Exception as exc:
        MODEL.set_unavailable(f'Startup model load failed: {exc}')
        update_model_metrics(
            run_id=None,
            model_type=None,
            features=(),
            available=False,
        )
    yield
    # add any teardown logic here if needed


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def metrics_middleware(request: Request, call_next):
        path = request.url.path
        if path == '/metrics':
            return await call_next(request)

        timer = RequestTimer()
        IN_PROGRESS_REQUESTS.inc()
        try:
            response = await call_next(request)
        except Exception:
            observe_request(
                method=request.method,
                path=path,
                status_code=500,
                duration_seconds=timer.elapsed(),
            )
            raise
        finally:
            IN_PROGRESS_REQUESTS.dec()

        observe_request(
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_seconds=timer.elapsed(),
        )
        return response

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        return {
            'status': 'ok' if model_state.model is not None else 'degraded',
            'run_id': model_state.run_id,
            'model_type': model_state.model_type,
            'features': list(model_state.features),
            'error': model_state.error,
        }

    @app.get('/metrics')
    def metrics():
        return metrics_response()

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        state = MODEL.get()
        if state.model is None:
            raise HTTPException(
                status_code=503,
                detail=state.error or 'Model is not loaded yet',
            )

        preprocess_timer = RequestTimer()
        try:
            df = to_dataframe(request, needed_columns=state.features)
            observe_preprocess(preprocess_timer.elapsed(), 'success')
            observe_features(df.iloc[0].to_dict())

            inference_timer = RequestTimer()
            prediction, probability = MODEL.predict(df)
            observe_inference(
                inference_timer.elapsed(),
                'success',
                state.model_type,
            )
            observe_prediction(prediction, probability)
        except FeatureSchemaError as exc:
            observe_preprocess(preprocess_timer.elapsed(), 'schema_error')
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FeatureValidationError as exc:
            observe_preprocess(preprocess_timer.elapsed(), 'validation_error')
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelInferenceError as exc:
            observe_inference(
                inference_timer.elapsed(),
                'error',
                state.model_type,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        try:
            MODEL.set(run_id=run_id)
            state = MODEL.get()
            update_model_metrics(
                run_id=state.run_id,
                model_type=state.model_type,
                features=state.features,
                available=True,
                touched_at=time.time(),
            )
            record_model_update('success')
        except ModelLoadError as exc:
            record_model_update('failure')
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
