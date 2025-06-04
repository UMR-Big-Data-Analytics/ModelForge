import abc
from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class ConsolidationStrategy(BaseEstimator, TransformerMixin, abc.ABC):
    @abstractmethod
    def transform(self, cluster: ModelDataSet) -> Pipeline:
        """
        Consolidate the models in the cluster into a single set.
        """
        raise NotImplementedError(
            "Method transform not implemented in class ConsolidationStrategy"
        )

    def fit(self, X, y=None):
        """
        Fit the consolidation strategy to the point.
        """

        return self
