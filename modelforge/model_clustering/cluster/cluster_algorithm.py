from typing import Protocol


class ClusterAlgorithm(Protocol):
    def fit(self, x, y=None):
        raise NotImplementedError("The fit method must be implemented")

    def fit_predict(self, x, y=None):
        raise NotImplementedError("The fit_predict method must be implemented")

    def predict(self, x):
        raise NotImplementedError("The predict method must be implemented")
