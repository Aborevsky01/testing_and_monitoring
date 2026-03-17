import mlflow

from ml_service import config


def configure_mlflow() -> None:
    uri = config.tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)


def get_model_uri(run_id: str) -> str:
    return f'runs:/{run_id}/model'


def load_model(model_uri: str = None, run_id: str = None) -> mlflow.pyfunc.PyFuncModel:
    """
    Downloads artifacts locally (if needed) and loads model as an MLflow PyFunc model.
    """
    if not model_uri:
        model_uri = get_model_uri(run_id)
    return mlflow.sklearn.load_model(model_uri)

