import pandas as pd

from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


class MinMaxPointSampler(PointSampler):
    def __init__(
        self,
        num_samples: int,
        column: str,
        objective: str = "max",
    ):
        """
        Sampler that selects samples based on a statistical measure.

        Parameters
        ----------
        num_samples : int
            Number of samples to select.
        column : str
            Column to select samples from.
        objective : str
            Objective to optimize.
            Either 'min' or 'max' for the num_samples with minimal or maximal value.
        """
        super().__init__(num_samples)
        self.column = column
        if objective not in ["min", "max"]:
            raise ValueError(f"Objective {objective} not in ['min', 'max']")
        self.objective = objective

    def get_sample_candidates(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Get sample candidates based on the statistical measure.
        Parameters
        ----------
        x : pd.DataFrame
            Data to sample from.
        args : list
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.
        """
        x = x.sort_values(by=self.column)
        if self.objective == "max":
            return x.nlargest(self.num_samples, self.column)
        else:
            return x.nsmallest(self.num_samples, self.column)
