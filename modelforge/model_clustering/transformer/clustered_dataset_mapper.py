import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from modelforge.model_clustering.entity.clustered_model_dataset import (
    ClusteredModelDataSet,
)
from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class ClusteredDataSetMapper(BaseEstimator, TransformerMixin):
    """
    Mixin class for mapping
    """

    def __init__(self, model_dataset: ModelDataSet):
        self.model_dataset = model_dataset

    def fit(self, _cluster_dataframe: pd.DataFrame, _y=None):
        return self

    def transform(self, cluster_dataframe: pd.DataFrame):
        return ClusteredModelDataSet.from_dataframe(
            cluster_dataframe, self.model_dataset
        )
