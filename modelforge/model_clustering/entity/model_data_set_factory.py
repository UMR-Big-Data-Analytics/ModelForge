import os
from typing import Type

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity


class LocalDiskModelDataSetFactory:
    @staticmethod
    def create(cls: Type[ModelEntity], path: str) -> ModelDataSet:
        dataset = ModelDataSet()
        ids = os.listdir(path)

        for entity_id in ids:
            model_entity = cls.load(f"{path}/{entity_id}")
            dataset.add_model_entity(model_entity)

        return dataset
