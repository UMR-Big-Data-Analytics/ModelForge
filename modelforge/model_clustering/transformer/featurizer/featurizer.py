import time
from abc import ABCMeta, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class Featurizer(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Mixin class for featuring the set. Used for potential subclassing.
    """

    def __init__(self):
        super().__init__()
        self.featurizer_runtime = None

    def fit(self, _x, _y=None):
        return self

    def transform(self, model_dataset: ModelDataSet) -> pd.DataFrame:
        start_time = time.time()
        result = self.featurize(model_dataset)
        end_time = time.time()
        self.featurizer_runtime = end_time - start_time
        return result

    @abstractmethod
    def featurize(self, model_dataset: ModelDataSet) -> pd.DataFrame:
        raise NotImplementedError("Featurizer should implement transform()")

    def get_params(self, deep=True):
        return {"featurizer_runtime": self.featurizer_runtime}
