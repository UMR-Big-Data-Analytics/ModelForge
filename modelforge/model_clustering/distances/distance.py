from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class Distance(BaseEstimator, TransformerMixin, ABC):
    @abstractmethod
    def transform(self, model_dataset: ModelDataSet) -> np.ndarray:
        """
        Calculate the distance matrix for the given set dataset.
        """
        raise NotImplementedError("The transform method must be implemented")
