import os

MODEL_ARTIFACT_PATH = 'model'


def tracking_uri() -> str:
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        raise RuntimeError('Please set MLFLOW_TRACKING_URI')
    return tracking_uri


def default_run_id() -> str:
    """
    Returns model URI for startup.
    """

    default_run_id = os.getenv('DEFAULT_RUN_ID')
    if not default_run_id:
        raise RuntimeError('Set DEFAULT_RUN_ID to load model on startup')
    return default_run_id
