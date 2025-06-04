from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass
class ModelClusteringScoringCalculator:
    scoring: pd.DataFrame

    def model_scores(self) -> (float, float):
        """
        Calculate the weighted mean and variance of the baseline set loss
        """
        # Calculate the weighted mean of x and y
        x_mean, _ = weighted_mean_and_var(
            self.scoring["model_loss"].values, self.scoring["cluster_size"].values
        )
        # Because the length of the test set is not the same for each device, we need to calculate the variance
        # by flatten model_loss column to one large array
        x_var = np.var(np.concatenate(self.scoring["model_losses"].to_list()))
        return x_mean, x_var

    def cluster_scores(self) -> (float, float):
        """
        Calculate the weighted mean and variance of the cluster set loss
        """
        return weighted_mean_and_var(
            self.scoring["cluster_loss"].values, self.scoring["cluster_size"].values
        )


@njit
def weighted_mean_and_var(values: np.ndarray, weights: np.ndarray) -> (float, float):
    """
    Return the weighted average and standard deviation.
    Taken from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    Parameters
    ----------
    values : np.ndarray
        NumPy ndarrays with the same shape.
    weights : np.ndarray
        NumPy ndarrays with the same shape.

    Returns
    -------
    average : float
        The weighted average
    variance : float
        The weighted variance
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, variance
