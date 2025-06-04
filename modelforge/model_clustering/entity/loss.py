from typing import Callable

import numpy as np
import pandas as pd

LossFunction = Callable[[pd.Series | np.ndarray, pd.Series | np.ndarray], float]


def compute_loss_ignore_nan(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series | pd.DataFrame | tuple,
    loss_function: Callable,
) -> float:
    # Handle special cases for weather dataset and reduce to mean value
    if isinstance(y_pred, pd.DataFrame):
        return loss_function(y_true, y_pred)
    if isinstance(y_pred, tuple):
        return loss_function(y_true, y_pred)
    assert len(y_pred) == len(
        y_true
    ), "y_pred and y_true must have the same length. We got {} and {}.".format(
        len(y_pred), len(y_true)
    )
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    # Find potential nan values is y_pred or y_true and ignore these values as they are not comparable in a loss
    # function.
    # This can happen, for example, in time series point where the first value is nan.
    nan_indices = np.isnan(y_pred) | np.isnan(y_true)
    y_pred = y_pred[~nan_indices]
    y_true = y_true[~nan_indices]
    return loss_function(y_true, y_pred)
