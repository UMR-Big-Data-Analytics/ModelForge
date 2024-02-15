from typing import Union, Optional

import joblib
import pandas as pd
from distributed import Client
import logging

from src.remote_service_algorithms.models.heating_consumption import generate_feature_dataset, train_model
from src.shared.logger import logger_factory
from src.shared.persistence.model_data_factory import ModelDataFactory


class ModelTrainer:
    def __init__(self,  model_data_factory: ModelDataFactory):
        self.logger = logger_factory(__name__)
        self.model_data_factory = model_data_factory

    def process(self, file: str, device: str) -> Union[pd.DataFrame, None]:
        frame = self.model_data_factory.get_raw_data(file)
        dataset = generate_feature_dataset(frame)
        if dataset is None:
            return None
        self.model_data_factory.save_data(dataset, device)
        return dataset

    def train(self, dataset: Union[pd.DataFrame, None], device: str) -> Optional[str]:
        if dataset is None:
            return None
        try:
            model_dict = train_model(dataset, device)
            self.model_data_factory.save_model_dict(model_dict, device)
            return device
        except ValueError as e:
            self.logger.error(f'Error training model for {device}', e)
            return None
