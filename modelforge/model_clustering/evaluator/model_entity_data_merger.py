import math
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

import pandas as pd

from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.shared.logger import logger_factory

F = TypeVar("F")
T = TypeVar("T")


class ModelEntityDataMerger(ABC, Generic[F, T]):
    @abstractmethod
    def merge(self, model_entities: List[ModelEntity]) -> (F, T, F, T):
        """
        Merges the point from the provided datasets into a single point structure.

        Parameters:
        datasets (List[ModelEntity]): The datasets to be merged.

        Returns:
        tuple: The merged point in the form of a tuple (train_x, train_y, test_x, test_y).
               "F" is the feature point (e.g., pd.DataFrame) and "T" is the target point (e.g., pd.Series).
        """
        raise NotImplementedError("Method not implemented")


class PandasModelEntityDataMerger(ModelEntityDataMerger[pd.DataFrame, pd.Series]):

    # size_threshold 2**25 = 32MB
    def __init__(self, approximate=False, size_threshold=2**25):
        self.approximate = approximate
        self.size_threshold = size_threshold
        self.logger = logger_factory(__name__)

    def merge(
        self, model_entities: List[ModelEntity]
    ) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """
        Merges the point from the provided datasets into a DataFrame and Series.

        Parameters:
        datasets (List[ModelEntity]): The datasets to be merged.

        Returns:
        tuple: The merged point in the form of a tuple (train_x, train_y, test_x, test_y).
        """
        train_xs = []
        train_ys = []
        test_xs = []
        test_ys = []
        approximation_count = 0
        for entity in model_entities:
            train_x, train_y, test_x, test_y, counter = self.approximate_if_necessary(
                entity
            )
            approximation_count += counter
            train_xs.append(train_x)
            train_ys.append(train_y)
            test_xs.append(test_x)
            test_ys.append(test_y)

        if self.approximate:
            self.logger.info(
                f"Approximated {approximation_count}/{len(model_entities)} entities"
            )
        train_x = pd.concat(train_xs)
        train_y = pd.concat(train_ys)
        test_x = pd.concat(test_xs)
        test_y = pd.concat(test_ys)
        return train_x, train_y, test_x, test_y

    def approximate_if_necessary(self, model_entity: ModelEntity):
        if not self.approximate:
            return (
                model_entity.train_x,
                model_entity.train_y,
                model_entity.test_x,
                model_entity.test_y,
                0,
            )

        size_per_row = (
            model_entity.train_x.memory_usage().sum() / model_entity.train_x.shape[0]
        )

        total_size = size_per_row * (
            model_entity.train_x.shape[0] + model_entity.test_x.shape[0]
        )

        if total_size < self.size_threshold:
            return (
                model_entity.train_x,
                model_entity.train_y,
                model_entity.test_x,
                model_entity.test_y,
                0,
            )

        train_test_ration = (
            float(model_entity.train_x.shape[0]) / model_entity.test_x.shape[0]
        )

        max_rows = math.floor(self.size_threshold / size_per_row)

        n_test_rows = math.floor(max_rows / (train_test_ration + 1))
        n_training_rows = math.floor(n_test_rows * train_test_ration)

        train_x = model_entity.train_x.sample(n_training_rows)
        train_y = model_entity.train_y[train_x.index]
        test_x = model_entity.test_x.sample(n_test_rows)
        test_y = model_entity.test_y[test_x.index]
        return train_x, train_y, test_x, test_y, 1


class ListModelEntityDataMerger(
    ModelEntityDataMerger[List[pd.DataFrame], List[pd.Series]]
):
    def merge(
        self, model_entities: List[ModelEntity]
    ) -> (List[pd.DataFrame], List[pd.Series], List[pd.DataFrame], List[pd.Series]):
        """
        Merges the point from the provided datasets into a list of DataFrames and Series.

        Parameters:
        datasets (List[ModelEntity]): The datasets to be merged.

        Returns:
        tuple: The merged point in the form of a tuple (train_x, train_y, test_x, test_y).
        """
        train_xs = []
        train_ys = []
        test_xs = []
        test_ys = []
        for entity in model_entities:
            train_xs.append(entity.train_x)
            train_ys.append(entity.train_y)
            test_xs.append(entity.test_x)
            test_ys.append(entity.test_y)

        return train_xs, train_ys, test_xs, test_ys
