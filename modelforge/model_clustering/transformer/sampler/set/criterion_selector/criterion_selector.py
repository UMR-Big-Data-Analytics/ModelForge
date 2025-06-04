from abc import abstractmethod

import numpy as np
import pandas as pd

from modelforge.model_clustering.entity.model_entity import ModelEntity


class CriterionSelector:
    @abstractmethod
    def get_criterion(
        self, model_entity: ModelEntity
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        raise NotImplementedError("CriterionSelector should implement get_criterion()")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("CriterionSelector should implement __str__()")
