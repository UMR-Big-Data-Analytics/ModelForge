import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClusterLabeler(TransformerMixin, BaseEstimator):
    def __init__(self, ids: pd.Series):
        self.ids = ids

    def fit(self, _x, _y=None):
        return self

    def transform(self, label: pd.Series):
        return pd.DataFrame(index=self.ids, data=label, columns=["cluster"])
