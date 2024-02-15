import numpy as np

EPSILON = 1e-10


def symmetric_mean_absolute_percentage_error_200(a: np.ndarray, f: np.ndarray) -> float:
    """
    This function calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two numpy arrays.
    The SMAPE is calculated as the mean of the element-wise 2 * absolute difference between the arrays, divided by the sum of their absolute values plus a small constant (EPSILON) to avoid division by zero.
    This version of SMAPE scales the error by 200%.

    SMAPE_{200} = \frac{1}{n} \sum_{i=1}^{n} 2 \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon} \times 100

    Parameters:
    a (np.ndarray): The first numpy array.
    f (np.ndarray): The second numpy array.

    Returns:
    float: The calculated SMAPE.
    """

    return np.mean(2.0 * np.abs(a - f) / ((np.abs(a) + np.abs(f)) + EPSILON))


def symmetric_mean_absolute_percentage_error_100(a: np.ndarray, f: np.ndarray) -> float:
    """
    This function calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two numpy arrays.
    The SMAPE is calculated as the mean of the element-wise absolute difference between the arrays, divided by the sum of their absolute values plus a small constant (EPSILON) to avoid division by zero.
    This version of SMAPE scales the error by 100%.

    SMAPE_{100} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon} \times 100

    Parameters:
    a (np.ndarray): The first numpy array.
    f (np.ndarray): The second numpy array.

    Returns:
    float: The calculated SMAPE.
    """

    return np.mean(np.abs(a - f) / ((np.abs(a) + np.abs(f)) + EPSILON))


def symmetric_mean_absolute_percentage_error(a: np.ndarray, f: np.ndarray) -> float:
    """
    This function calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two numpy arrays using the smape_200 function. See also https://medium.com/@davide.sarra/how-to-interpret-smape-just-like-mape-bf799ba03bdc

    Parameters:
    a (np.ndarray): The first numpy array.
    f (np.ndarray): The second numpy array.

    Returns:
    float: The calculated SMAPE.
    """
    return symmetric_mean_absolute_percentage_error_200(a, f)
