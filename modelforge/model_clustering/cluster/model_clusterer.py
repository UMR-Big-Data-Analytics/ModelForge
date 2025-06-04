import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.cluster.cluster_labeler import ClusterLabeler
from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class ModelClusterer(TransformerMixin, BaseEstimator):
    """
    A transformer that takes a dataframe and returns a dataframe with the cluster labels

    Parameters
    ----------
    cluster_mixin : ClusterAlgorithm
        A clustering algorithm that implements the fit and predict methods
    ids : pd.Series
        The ids of the models
    """

    def __init__(self, cluster_mixin: ClusterAlgorithm, ids: pd.Series):
        self.cluster_mixin = cluster_mixin
        self.ids = ids
        self.set_params(**{"cluster_mixin": cluster_mixin, "ids": ids})

    def fit(self, x: pd.DataFrame, _y=None, **kwargs):
        """
        Fit the cluster algorithm to the point
        """
        self.cluster_mixin.fit(x, **kwargs)
        return self

    def transform(self, x: pd.DataFrame | ModelDataSet):
        """
        Transform the point into cluster label dataframe containing the set id as index and the cluster label in the 'cluster' column
        """
        check_is_fitted(self.cluster_mixin)
        # Because the sklearn.Pipeline allows a estimator only in the last step of the pipeline, we need to fit the cluster algorithm in a transformer such that other transformers can be applied after the clustering
        # Check if cluster_mixin has a predict method
        if hasattr(self.cluster_mixin, "predict"):
            labels = self.cluster_mixin.predict(x)
        else:
            labels = self.cluster_mixin.labels_
        cluster_labeler = ClusterLabeler(self.ids)
        df = cluster_labeler.transform(labels)
        if isinstance(x, ModelDataSet):
            return df
        if isinstance(x, pd.DataFrame):
            x = x.values
        embedding = pd.DataFrame(
            x, index=self.ids, columns=[f"embedding_{i}" for i in range(x.shape[1])]
        )
        return pd.concat([df, embedding], axis=1)

    def fit_transform(self, x: pd.DataFrame, _y=None, **kwargs):
        """
        Fit the cluster algorithm to the point and transform the point into cluster label dataframe containing the
        set id as index and the cluster label in the 'cluster' column
        """
        # Call internal fit_predict method of the cluster algorithm
        self.fit(x, **kwargs)
        return self.transform(x)
