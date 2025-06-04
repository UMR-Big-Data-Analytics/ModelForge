import matplotlib
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt

from modelforge.shared.logger import logger_factory
from modelforge.visualization.plotting_functions import scatter_performance, set_size

# plt.style.use('tex.mplstyle')

blueish = [
    "#23396e",
    "#023e8a",
    "#297acc",
    "#0096c7",
    "#00b4d8",
    "#48cae4",
    "#90e0ef",
    "#ade8f4",
    "#caf0f8",
]
colorfull = [
    "#d64265",
    "#f9844a",
    "#f9c74f",
    "#90be6d",
    "#43aa8b",
    "#4d908e",
    "#297acc",
    "#23396e",
]

logger = logger_factory(__name__)


def cmap_exists(name: str) -> bool:
    try:
        cm.get_cmap(name)
        return True
    except ValueError:
        pass
    return False


def register_cmaps():
    if not cmap_exists("blueish"):
        cmap_blueish = matplotlib.colors.LinearSegmentedColormap.from_list(
            "blueish", blueish
        )
        matplotlib.colormaps.register(cmap_blueish)

    if not cmap_exists("colorfull"):
        cmap_colorfull = matplotlib.colors.LinearSegmentedColormap.from_list(
            "colorfull", colorfull
        )
        matplotlib.colormaps.register(cmap_colorfull)


def plot_score(
    score: pd.DataFrame, title: str = None, cmap="blueish", norm: str = "log", **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    plot_range = (
        (15, 15) if kwargs.get("plot_range") is None else kwargs.pop("plot_range")
    )

    scoring = score["scoring"]

    scoring.loc[scoring["model_loss"] > plot_range[0], "model_loss"] = plot_range[0]
    scoring.loc[scoring["cluster_loss"] > plot_range[1], "cluster_loss"] = plot_range[1]
    borders = []
    kwargs["plot_range"] = plot_range
    if kwargs.get("xlabel") is None:
        kwargs["xlabel"] = f"$\mu_b$"
    if kwargs.get("ylabel") is None:
        kwargs["ylabel"] = f"$\mu_c$"

    x = "model_loss"
    y = "cluster_loss"

    x_mean, x_var, y_mean, y_var = (
        score["model_loss"],
        score["model_var"],
        score["cluster_loss"],
        score["cluster_var"],
    )

    if kwargs.get("figsize") is None:
        kwargs["figsize"] = set_size("thesis", ratio=1)

    fig, ax = scatter_performance(
        scoring,
        x=x,
        y=y,
        x_mean=x_mean,
        x_var=x_var,
        y_mean=y_mean,
        y_var=y_var,
        size="cluster_size",
        norm=norm,
        cmap=cmap,
        borders=borders,
        marker="x",
        **kwargs,
    )
    if title is not None:
        ax.set_title(title)

    return fig, ax


def remove_nan(array: np.ndarray) -> (np.ndarray, int):
    non_nan_values = array[~np.isnan(array)]
    if len(non_nan_values) != len(array):
        logger.warning(f"Removed {len(array) - len(non_nan_values)} NaN values")
    return non_nan_values, len(non_nan_values)


def calculate_aggregated_scores(scores, x="model_loss", y="cluster_loss"):
    # Calculate the weighted mean of x and y
    x_mean, _ = weighted_mean_and_var(scores[x], scores["cluster_size"])
    # Flatten model_loss column to one large array
    x_var = np.var(np.concatenate(scores[f"{x}es"].to_list()))
    y_mean, y_var = weighted_mean_and_var(scores[y], scores["cluster_size"])
    return x_mean, x_var, y_mean, y_var


def weighted_mean_and_var(values, weights):
    """
    Return the weighted average and standard deviation.
    Taken from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, variance
