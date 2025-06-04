from typing import Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm

EPSILON = 1e-10


def diff_loss(
    a: Union[np.ndarray, pd.Series], b: Union[np.ndarray, pd.Series]
) -> Union[np.ndarray, pd.Series]:
    return a - b


def crps_unpacking(
    y: Union[np.ndarray, pd.Series],
    pred: (
        Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]] | np.ndarray
    ),
):
    if isinstance(pred, tuple):
        return crps(pred[0], pred[1], y)
    if isinstance(pred, pd.DataFrame):
        return crps(pred.iloc[:, 0], pred.iloc[:, 1], y)
    if isinstance(pred, np.ndarray):
        if pred.ndim == 2:
            return crps(pred[:, 0], pred[:, 1], y)
        elif pred.ndim == 1:
            return crps(pred[0], pred[1], y)
    raise ValueError("Invalid type for pred")


def crps(
    mu: Union[np.ndarray, pd.Series],
    sigma: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
) -> float:
    """
    This function calculates the Continuous Ranked Probability Score (CRPS) between the predicted mean and standard
    deviation and the true value. The CRPS is calculated as the integral of the squared difference between the
    cumulative distribution function of the predicted normal distribution and the cumulative distribution function of
    the true value. The CRPS is calculated as the mean of the element-wise CRPS.

    Parameters
    ----------
    mu (np.ndarray): The predicted mean.
    sigma (np.ndarray): The predicted standard deviation.
    y (np.ndarray): The true value.

    Returns
    -------
    float: The calculated CRPS.

    """
    if isinstance(mu, pd.Series):
        mu = mu.to_numpy()
    if isinstance(sigma, pd.Series):
        sigma = sigma.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    sigma = np.abs(sigma)
    loc = (y - mu) / sigma
    crpses = sigma * (
        loc * (2 * norm.cdf(loc) - 1) + 2 * norm.pdf(loc) - 1.0 / np.sqrt(np.pi)
    )
    return np.mean(crpses)


def symmetric_mean_absolute_percentage_error_200(
    a: Union[np.ndarray, pd.Series], b: Union[np.ndarray, pd.Series]
) -> float:
    a, b = as_numpy(a, b)
    return symmetric_mean_absolute_percentage_error_200_func(a, b)


def symmetric_mean_absolute_percentage_error_100(
    a: Union[np.ndarray, pd.Series], b: Union[np.ndarray, pd.Series]
) -> float:
    a, b = as_numpy(a, b)
    return symmetric_mean_absolute_percentage_error_100_func(a, b)


def as_numpy(a, b):
    if isinstance(a, pd.Series):
        a = a.to_numpy()
    if isinstance(b, pd.Series):
        b = b.to_numpy()
    return a, b


@njit
def symmetric_mean_absolute_percentage_error_200_func(
    a: np.ndarray, f: np.ndarray
) -> float:
    """
    This function calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two numpy arrays. The SMAPE
    is calculated as the mean of the element-wise 2 * absolute difference between the arrays, divided by the sum of
    their absolute values plus a small constant (EPSILON) to avoid division by zero. This version of SMAPE scales the
    error by 200%.

    SMAPE_{200} = \frac{1}{n} \sum_{i=1}^{n} 2 \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon} \times 100

    Parameters:
    a (np.ndarray): The first numpy array.
    f (np.ndarray): The second numpy array.

    Returns:
    float: The calculated SMAPE.
    """

    return np.mean(2.0 * np.abs(a - f) / ((np.abs(a) + np.abs(f)) + EPSILON))


@njit
def symmetric_mean_absolute_percentage_error_100_func(
    a: np.ndarray, f: np.ndarray
) -> float:
    """
    This function calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two numpy arrays. The SMAPE
    is calculated as the mean of the element-wise absolute difference between the arrays, divided by the sum of their
    absolute values plus a small constant (EPSILON) to avoid division by zero. This version of SMAPE scales the error
    by 100%.

    SMAPE_{100} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon} \times 100

    Parameters:
    a (np.ndarray): The first numpy array.
    f (np.ndarray): The second numpy array.

    Returns:
    float: The calculated SMAPE.
    """

    return np.mean(np.abs(a - f) / ((np.abs(a) + np.abs(f)) + EPSILON))


def symmetric_mean_absolute_percentage_error(
    a: Union[np.ndarray, pd.Series], b: Union[np.ndarray, pd.Series]
) -> float:
    """
    This function calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two numpy arrays using the
    smape_200 function. See also https://medium.com/@davide.sarra/how-to-interpret-smape-just-like-mape-bf799ba03bdc

    Parameters:
    a (np.ndarray): The first numpy array.
    b (np.ndarray): The second numpy array.

    Returns:
    float: The calculated SMAPE.
    """
    return symmetric_mean_absolute_percentage_error_200(a, b)
