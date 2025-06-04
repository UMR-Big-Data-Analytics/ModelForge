from modelforge.datasets.anomaly_models.anomaly_model_entity import AnomalyModelEntity
from modelforge.datasets.immoscout_models.house_price.immoscout_house_price_model_entity import (
    ImmoscoutHousePriceModelEntity,
)
from modelforge.datasets.weather_models.weather_model_entity import (
    WeatherModelProbEntity,
)
from modelforge.model_clustering.entity.model_data_set_factory import (
    LocalDiskModelDataSetFactory,
)


def anomaly_dataset(
    prediction_mode: str,
    processed_filepath="data/processed/anomaly_models",
):
    if prediction_mode not in ["score", "anomaly"]:
        raise ValueError("prediction_model must be either 'score' or 'anomaly'")
    return LocalDiskModelDataSetFactory.create(
        AnomalyModelEntity, f"{processed_filepath}/{prediction_mode}/data"
    )


def immoscout_house_price_dataset(
    processed_filepath="data/processed/immoscout24_models/house_price/data",
):
    return LocalDiskModelDataSetFactory.create(
        ImmoscoutHousePriceModelEntity, processed_filepath
    )


def weather_dataset_probabilistic_regression(
    processed_filepath="data/processed/weather_models/prob/data",
):
    return LocalDiskModelDataSetFactory.create(
        WeatherModelProbEntity, processed_filepath
    )
