from typing import List

import numpy as np

from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class ParamsFactory:
    @staticmethod
    def cluster_numbers(
        dataset: ModelDataSet,
        factors: List[float] | np.ndarray = np.linspace(0.05, 1, 9),
    ) -> dict:
        if isinstance(factors, np.ndarray) and factors.ndim != 1:
            raise ValueError("Factors must be a 1D array.")
        return {
            "modelclusterer__cluster_mixin__n_clusters": [
                max(int(factor * dataset.size), 1) for factor in factors
            ]
        }
