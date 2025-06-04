import pandas as pd

from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


class RandomPointSampler(PointSampler):
    def __init__(self, num_samples: int, random_state: int = 42):
        super().__init__(num_samples)
        self.random_state = random_state

    def get_sample_candidates(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return x.sample(self.num_samples, random_state=self.random_state)
