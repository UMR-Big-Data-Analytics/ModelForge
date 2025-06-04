import numpy as np
import pandas as pd

from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


class LinearPointSampler(PointSampler):
    """
    Sampler that selects samples linearly based on a specified column.
    """

    def __init__(self, num_samples: int, sort_column: str = None):
        super().__init__(num_samples)
        self.sort_column = sort_column

    def get_sample_candidates(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        x = x.sort_values(by=self.sort_column)
        indices = np.linspace(0, len(x) - 1, self.num_samples, dtype=int)
        return x.iloc[indices]
