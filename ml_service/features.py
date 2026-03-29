import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]


class FeatureValidationError(ValueError):
    """
    Raised when request data cannot be converted into a valid model input row.
    """


class FeatureSchemaError(RuntimeError):
    """
    Raised when the loaded model expects a feature schema unsupported by the service.
    """


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    columns = [
        column for column in needed_columns if column in FEATURE_COLUMNS
    ] if needed_columns is not None else FEATURE_COLUMNS

    if needed_columns is not None:
        unexpected_columns = sorted(set(needed_columns) - set(FEATURE_COLUMNS))
        if unexpected_columns:
            raise FeatureSchemaError(
                'Model expects unsupported features: '
                f'{", ".join(unexpected_columns)}'
            )

    missing_columns = [
        column
        for column in columns
        if getattr(req, column.replace('.', '_')) is None
    ]
    if missing_columns:
        raise FeatureValidationError(
            'Missing required features for current model: '
            f'{", ".join(missing_columns)}'
        )

    row = [getattr(req, column.replace('.', '_')) for column in columns]
    return pd.DataFrame([row], columns=columns)
