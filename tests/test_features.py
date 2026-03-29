import unittest

from ml_service.features import (
    FeatureSchemaError,
    FeatureValidationError,
    to_dataframe,
)
from ml_service.schemas import PredictRequest


class ToDataFrameTests(unittest.TestCase):
    def test_keeps_only_required_columns_in_model_order(self) -> None:
        request = PredictRequest(
            age=39,
            workclass='Private',
            education='Bachelors',
            occupation='Tech-support',
            race='White',
            sex='Male',
            **{
                'capital.gain': 2174,
                'native.country': 'United-States',
            },
        )

        dataframe = to_dataframe(
            request,
            needed_columns=['race', 'sex', 'capital.gain'],
        )

        self.assertEqual(list(dataframe.columns), ['race', 'sex', 'capital.gain'])
        self.assertEqual(dataframe.iloc[0].to_dict(), {
            'race': 'White',
            'sex': 'Male',
            'capital.gain': 2174,
        })

    def test_raises_when_required_feature_is_missing(self) -> None:
        request = PredictRequest(
            race='White',
            sex='Male',
        )

        with self.assertRaisesRegex(
            FeatureValidationError,
            'hours.per.week',
        ):
            to_dataframe(
                request,
                needed_columns=['race', 'hours.per.week'],
            )

    def test_raises_when_model_requests_unsupported_feature(self) -> None:
        request = PredictRequest(race='White')

        with self.assertRaisesRegex(
            FeatureSchemaError,
            'unsupported_feature',
        ):
            to_dataframe(request, needed_columns=['race', 'unsupported_feature'])
