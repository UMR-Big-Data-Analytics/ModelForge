import pandas as pd

from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.sampler.set.criterion_selector.criterion_selector import (
    CriterionSelector,
)


class TargetSelector(CriterionSelector):
    def get_criterion(self, model_entity: ModelEntity) -> pd.Series:
        return model_entity.train_y

    def __str__(self):
        return "TargetSelector"
