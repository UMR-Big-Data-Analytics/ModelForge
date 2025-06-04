from typing import List

import pandas as pd

from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.sampler.set.criterion_selector.criterion_selector import (
    CriterionSelector,
)


class FeatureSelector(CriterionSelector):
    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns

    def get_criterion(self, model_entity: ModelEntity) -> pd.Series:
        return model_entity.train_x[self.columns]

    def __str__(self):
        return f"FeatureSelector_{'_'.join(self.columns)}"
