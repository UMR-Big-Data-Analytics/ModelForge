import pandas as pd

from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


class PercentilePointSampler(PointSampler):
    def __init__(
        self,
        column: str,
        percentile_lower: float,
        percentile_upper: float,
        base_sampler: PointSampler,
    ):
        """
        Sampler that selects samples based on the percentile range.

        Parameters
        ----------
        column : str
            Column to select samples from.
        percentile_lower : float
            Lower percentile boundary.
        percentile_upper : float
            Upper percentile boundary.
        base_sampler : PointSampler
            Base sampler to use for sampling.
        """
        super().__init__(num_samples=base_sampler.num_samples)
        self.column = column
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.base_sampler = base_sampler

    def get_sample_candidates(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Get sample candidates based on the percentile range.

        Parameters
        ----------
        x : pd.DataFrame
            Data to sample from.
        args : list
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            Sampled point.
        """
        if self.column not in x.columns:
            raise ValueError(f"Column {self.column} not found in point.")
        x = x.sort_values(by=self.column)
        lower_bound = x[self.column].quantile(self.percentile_lower)
        upper_bound = x[self.column].quantile(self.percentile_upper)
        x = x[(x[self.column] >= lower_bound) & (x[self.column] <= upper_bound)]
        return self.base_sampler.get_sample_candidates(x, *args, **kwargs)
