from enum import Enum, unique


@unique
class ModelGroup(Enum):

    ENERGY_EFFICIENCY = "energy_efficiency"


@unique
class ModelName(Enum):

    TRAIN_PREDICT_CONSUMPTION_FOR_HEATING = "train_predict_consumption_for_heating"

    TRAIN_PREDICT_CONSUMPTION_FOR_DHW = "train_predict_consumption_for_dhw"


def get_name(name: str) -> ModelName:
    return getattr(ModelName, name)