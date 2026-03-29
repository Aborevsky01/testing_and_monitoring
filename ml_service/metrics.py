from __future__ import annotations

import time
from collections.abc import Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from starlette.responses import Response


REQUEST_DURATION_SECONDS = Histogram(
    'bor_ml_service_request_duration_seconds',
    'End-to-end HTTP request duration.',
    labelnames=('method', 'path'),
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)
REQUESTS_TOTAL = Counter(
    'bor_ml_service_requests_total',
    'Number of handled HTTP requests.',
    labelnames=('method', 'path', 'status_code'),
)
IN_PROGRESS_REQUESTS = Gauge(
    'bor_ml_service_in_progress_requests',
    'Number of HTTP requests currently being processed.',
)

PREPROCESS_DURATION_SECONDS = Histogram(
    'bor_ml_service_preprocess_duration_seconds',
    'Time spent converting input payload into a model-ready DataFrame.',
    labelnames=('outcome',),
    buckets=(
        0.0001,
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
    ),
)
INFERENCE_DURATION_SECONDS = Histogram(
    'bor_ml_service_inference_duration_seconds',
    'Time spent running model.predict_proba.',
    labelnames=('outcome', 'model_type'),
    buckets=(
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
    ),
)

FEATURE_NUMERIC_VALUE = Histogram(
    'bor_ml_service_feature_numeric_value',
    'Distribution of numeric feature values sent to the service.',
    labelnames=('feature',),
    buckets=(
        0.0,
        1.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        500.0,
        1_000.0,
        5_000.0,
        10_000.0,
        50_000.0,
        100_000.0,
    ),
)
FEATURE_CATEGORICAL_TOTAL = Counter(
    'bor_ml_service_feature_categorical_total',
    'Number of times a categorical feature value was observed.',
    labelnames=('feature', 'value'),
)

MODEL_PROBABILITY = Histogram(
    'bor_ml_service_model_probability',
    'Distribution of model output probabilities for the positive class.',
    buckets=tuple(value / 20 for value in range(21)),
)
MODEL_PREDICTIONS_TOTAL = Counter(
    'bor_ml_service_model_predictions_total',
    'Number of emitted model predictions.',
    labelnames=('prediction',),
)

MODEL_UPDATES_TOTAL = Counter(
    'bor_ml_service_model_updates_total',
    'Number of model update attempts.',
    labelnames=('outcome',),
)
CURRENT_MODEL_INFO = Info(
    'bor_ml_service_current_model',
    'Metadata of the model currently served by the application.',
)
CURRENT_MODEL_AVAILABLE = Gauge(
    'bor_ml_service_current_model_available',
    'Whether a model is currently available for inference.',
)
CURRENT_MODEL_FEATURE_COUNT = Gauge(
    'bor_ml_service_current_model_feature_count',
    'Number of features required by the current model.',
)
CURRENT_MODEL_LAST_UPDATE_TIME = Gauge(
    'bor_ml_service_current_model_last_update_unixtime',
    'Unix timestamp of the last successful model load.',
)


NUMERIC_FEATURES = {
    'age',
    'fnlwgt',
    'education.num',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
}


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    REQUESTS_TOTAL.labels(
        method=method,
        path=path,
        status_code=str(status_code),
    ).inc()
    REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration_seconds)


def observe_preprocess(duration_seconds: float, outcome: str) -> None:
    PREPROCESS_DURATION_SECONDS.labels(outcome=outcome).observe(duration_seconds)


def observe_inference(duration_seconds: float, outcome: str, model_type: str | None) -> None:
    INFERENCE_DURATION_SECONDS.labels(
        outcome=outcome,
        model_type=model_type or 'unknown',
    ).observe(duration_seconds)


def observe_features(row: dict[str, object]) -> None:
    for feature, value in row.items():
        if feature in NUMERIC_FEATURES:
            FEATURE_NUMERIC_VALUE.labels(feature=feature).observe(float(value))
            continue
        FEATURE_CATEGORICAL_TOTAL.labels(feature=feature, value=str(value)).inc()


def observe_prediction(prediction: int, probability: float) -> None:
    MODEL_PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()
    MODEL_PROBABILITY.observe(probability)


def update_model_metrics(
    *,
    run_id: str | None,
    model_type: str | None,
    features: Iterable[str],
    available: bool,
    touched_at: float | None = None,
) -> None:
    feature_list = list(features)
    CURRENT_MODEL_AVAILABLE.set(1 if available else 0)
    CURRENT_MODEL_FEATURE_COUNT.set(len(feature_list))
    CURRENT_MODEL_INFO.info({
        'run_id': run_id or 'unavailable',
        'model_type': model_type or 'unavailable',
        'features': ','.join(feature_list),
    })
    if touched_at is not None:
        CURRENT_MODEL_LAST_UPDATE_TIME.set(touched_at)


def record_model_update(outcome: str) -> None:
    MODEL_UPDATES_TOTAL.labels(outcome=outcome).inc()


class RequestTimer:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at
