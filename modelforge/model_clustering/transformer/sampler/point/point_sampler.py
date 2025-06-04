from abc import ABCMeta, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PointSampler(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Mixin class for sampling training point. Used for potential subclassing.
    """

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def fit(self, _x, _y=None):
        return self

    def fit_transform(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return self.transform(x, *args, **kwargs)

    def transform(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        sample = self.get_sample_candidates(x, *args, **kwargs)
        if len(sample) != self.num_samples:
            raise ValueError(
                f"Sample size is not equal to the specified number of samples: {len(sample)}. Expected: {self.num_samples}. Sampler: {self.__class__.__name__}"
            )
        return sample

    def get_sample_size(self) -> int:
        return self.num_samples

    @abstractmethod
    def get_sample_candidates(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Sampler should implement get_sample_candidates()")
