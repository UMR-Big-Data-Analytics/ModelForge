import pandas as pd
from joblib import load, dump
from functools import cache, lru_cache

from src.shared.persistence.data_path_factory import DataPathFactory


class ModelDataFactory:
    def __init__(self, data_path_factory: DataPathFactory):
        self.data_path_factory = data_path_factory

    @lru_cache(maxsize=100)
    def get_model_dict(self, device_uuid: str) -> dict:
        path = self.data_path_factory.get_model_dict_path(device_uuid)
        return load(path)

    def get_model(self, device_uuid: str):
        model_dict = self.get_model_dict(device_uuid)
        return model_dict['model']

    @lru_cache(maxsize=100)
    def get_raw_data(self, device_uuid: str) -> pd.DataFrame:
        path = self.data_path_factory.get_raw_data_path(device_uuid)
        return pd.read_parquet(path)

    @lru_cache(maxsize=100)
    def get_data(self, device_uuid: str) -> pd.DataFrame:
        path = self.data_path_factory.get_processed_data_path(device_uuid)
        return pd.read_parquet(path)

    @lru_cache(maxsize=100)
    def get_test_data(self, device_uuid: str) -> pd.DataFrame:
        model_dict = self.get_model_dict(device_uuid)
        test_index = model_dict['model_params']['test_index']
        data = self.get_data(device_uuid)
        return data.loc[data.index.isin(test_index)]

    @lru_cache(maxsize=100)
    def get_train_data(self, device_uuid: str) -> pd.DataFrame:
        model_dict = self.get_model_dict(device_uuid)
        train_index = model_dict['model_params']['train_index']
        data = self.get_data(device_uuid)
        return data.loc[data.index.isin(train_index)]

    def feature_list(self, device_uuid: str) -> list[str]:
        model_dict = self.get_model_dict(device_uuid)
        return model_dict['model_params']['features']

    def save_data(self, dataset: pd.DataFrame, device: str):
        path = self.data_path_factory.get_processed_data_path(device)
        dataset.to_parquet(path)

    def save_model_dict(self, model: dict, device: str):
        path = self.data_path_factory.get_model_dict_path(device)
        dump(model, path)
