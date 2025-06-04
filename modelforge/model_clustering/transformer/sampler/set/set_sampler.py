from abc import abstractmethod
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity


class SetSampler(BaseEstimator, TransformerMixin):
    """
    Mixin class for sampling models/sets of training point. Used for potential subclassing.
    """

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def fit(self, _x, _y=None):
        return self

    def transform(self, x: ModelDataSet) -> List[ModelEntity]:
        samples = self.get_sample_candidates(x)
        if len(samples) != self.num_samples:
            raise ValueError(
                f"Sample size is not equal to the specified number of samples: {len(samples)}"
            )
        return samples

    def get_sample_size(self) -> int:
        return self.num_samples

    @abstractmethod
    def get_sample_candidates(self, x: ModelDataSet) -> List[ModelEntity]:
        raise NotImplementedError("Sampler should implement get_sample_candidates()")
