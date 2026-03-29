from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    age: int | None = Field(default=None, ge=0, description='Возраст человека')
    workclass: str | None = Field(default=None, min_length=1, description='Тип занятости')
    fnlwgt: int | None = Field(
        default=None,
        ge=0,
        description='Вес наблюдения в данных переписи',
    )
    education: str | None = Field(default=None, min_length=1, description='Образование')
    education_num: int | None = Field(
        default=None,
        alias='education.num',
        ge=0,
        description='Уровень образования в виде числа',
    )
    marital_status: str | None = Field(
        default=None,
        alias='marital.status',
        min_length=1,
        description='Семейное положение',
    )
    occupation: str | None = Field(
        default=None,
        min_length=1,
        description='Профессия / род деятельности',
    )
    relationship: str | None = Field(
        default=None,
        min_length=1,
        description='Роль человека в семье',
    )
    race: str | None = Field(default=None, min_length=1, description='Расовая группа')
    sex: str | None = Field(
        default=None,
        min_length=1,
        description='Пол человека (Male / Female)',
    )
    capital_gain: int | None = Field(
        default=None,
        alias='capital.gain',
        ge=0,
        description='Доход от капитала (прибыль от продажи активов)',
    )
    capital_loss: int | None = Field(
        default=None,
        alias='capital.loss',
        ge=0,
        description='Убытки от капитала',
    )
    hours_per_week: int | None = Field(
        default=None,
        alias='hours.per.week',
        ge=0,
        description='Количество рабочих часов в неделю',
    )
    native_country: str | None = Field(
        default=None,
        alias='native.country',
        min_length=1,
        description='Страна происхождения',
    )


class PredictResponse(BaseModel):
    prediction: int = Field(description='Предсказанный класс')
    probability: float = Field(description='Вероятность предсказанного класса')


class UpdateModelRequest(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        str_strip_whitespace=True,
    )

    run_id: str = Field(min_length=1, description='MLflow run_id')


class UpdateModelResponse(BaseModel):
    run_id: str = Field(description='MLflow run_id')
