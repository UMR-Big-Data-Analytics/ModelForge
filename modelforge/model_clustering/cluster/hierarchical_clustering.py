import numpy as np
from joblib import Memory
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator

from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.distances.distance import Distance
from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class HierarchicalClustering(ClusterAlgorithm, BaseEstimator):
    def __init__(
        self,
        n_clusters: int = 3,
        distance_measure: Distance = None,
        distance_matrix: np.ndarray = None,
        linkage_method: str = "complete",
        memory: Memory = None,
    ):
        self.n_clusters = n_clusters
        self.linkage_matrix = None
        self.distance_measure = distance_measure
        self.distance_matrix = distance_matrix
        self.linkage_method = linkage_method
        if memory is None:
            memory = Memory(location="./.cache/")
        self.memory = memory
        self.isFitted_ = False

    def fit(self, model_dataset: ModelDataSet, _y=None):
        """
        Fit the hierarchical clustering algorithm to the set dataset
        """
        if self.distance_measure is None and self.distance_matrix is None:
            raise ValueError(
                "Distance measure or distance matrix must be set before fitting"
            )

        if self.distance_matrix is None:
            self.distance_matrix = self.compute_distance_matrix(model_dataset)

        self.linkage_matrix = linkage(self.distance_matrix, method=self.linkage_method)

        self.isFitted_ = True
        return self

    def compute_distance_matrix(self, model_dataset: ModelDataSet) -> np.ndarray:
        dist = self.distance_measure.transform(model_dataset)
        if np.isnan(dist).any():
            # Print where the NaN values are n
            raise ValueError("Distance matrix contains NaN values")
        if dist.shape[0] == dist.shape[1]:
            # Convert to condensed form
            dist = squareform(dist)
        if np.any(dist < 0):
            raise ValueError("Distance matrix contains negative values")
        return dist

    def predict(self, model_dataset: ModelDataSet):
        """
        Predict the clusters for the set dataset
        """

        if self.linkage_matrix is None:
            raise ValueError("Model must be fit before predicting")

        return fcluster(self.linkage_matrix, t=self.n_clusters, criterion="maxclust")

    def fit_predict(self, model_dataset: ModelDataSet, _y=None):
        """
        Fit the hierarchical clustering algorithm to the set dataset and predict the clusters
        """
        self.fit(model_dataset)
        return self.predict(model_dataset)
