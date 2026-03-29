import os

MODEL_ARTIFACT_PATH = 'model'
DEFAULT_EVIDENTLY_URL = 'http://158.160.2.37:8000/'
DEFAULT_EVIDENTLY_PROJECT_ID = '019d061f-cc08-7b5e-b932-d792a1f258e2'


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


def evidently_url() -> str:
    return os.getenv('EVIDENTLY_URL', DEFAULT_EVIDENTLY_URL)


def evidently_project_id() -> str:
    return os.getenv('EVIDENTLY_PROJECT_ID', DEFAULT_EVIDENTLY_PROJECT_ID)


def evidently_report_interval_seconds() -> int:
    return int(os.getenv('EVIDENTLY_REPORT_INTERVAL_SECONDS', '60'))


def evidently_reference_size() -> int:
    return int(os.getenv('EVIDENTLY_REFERENCE_SIZE', '50'))


def evidently_current_size() -> int:
    return int(os.getenv('EVIDENTLY_CURRENT_SIZE', '50'))
