import logging
import math
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
cmap_blueish = matplotlib.colors.LinearSegmentedColormap.from_list("blueish", blueish)
cmap_colorfull = matplotlib.colors.LinearSegmentedColormap.from_list(
    "colorfull", colorfull
)
matplotlib.colormaps.register(cmap_blueish)
matplotlib.colormaps.register(cmap_colorfull)


# Helper functions for plotting the performance of the clustering
def calculate_text_rotation(xyfrom, xyto):
    dx = xyto[0] - xyfrom[0]
    dy = xyto[1] - xyfrom[1]
    rotn = np.degrees(np.arctan2(dy, dx))  # not the transformed p2 and p1
    return rotn


def calculate_text_position(
    xyfrom, xyto, text_offset=0.1, direction: int = 1, figsize=(6, 6)
):
    midx = (xyfrom[0] + xyto[0]) / 2
    midy = (xyfrom[1] + xyto[1]) / 2

    x_ratio = xyto[0] / figsize[0]
    y_ratio = xyto[1] / figsize[1]

    return (
        midx + direction * x_ratio * text_offset,
        midy + -direction * y_ratio * text_offset,
    )


def calculate_xyto(xyfrom, slope, plot_range):
    x = xyfrom[0]
    y = xyfrom[1]
    if x > 0:
        y_correction = -x * slope
        x_correction = y * slope
    else:
        y_correction = y * slope
        x_correction = -y * slope
    if slope == 0:
        y = plot_range[1]
    elif slope == np.inf:
        x = plot_range[0]
    else:
        x = plot_range[0] + x_correction
        y = slope * x + y_correction
    return x, y


def calculate_text_decoration(
    x_from, y_to, slope, x_range, y_range, direction, figsize
):
    xyto = calculate_xyto((x_from, y_to), slope, (x_range, y_range))
    xytext = calculate_text_position(
        (x_from, y_to), xyto, direction=direction, figsize=figsize
    )
    text_rotation = calculate_text_rotation((x_from, y_to), xyto)
    return text_rotation, xytext


def latex_exp(value: float, precision: int = 2) -> str:
    if value == 0:
        return "0"

    if value < 100:
        return f"{value:.{precision}f}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    if exponent == 0:
        return f"{mantissa:.{precision}f}"
    return f"{mantissa:.{precision}f}e{exponent}"


def scatter_performance(
    df,
    x,
    y,
    x_mean,
    x_var,
    y_mean,
    y_var,
    xlabel=None,
    ylabel=None,
    size=None,
    color=None,
    limit_border=3,
    borders=None,
    figsize=(6, 6),
    plot_range=None,
    cmap=None,
    norm=None,
    marker=None,
    markers=None,
    label=None,
    legend=False,
    colorbar=False,
    fig=None,
    ax=None,
):
    if borders is None:
        borders = [5, 8, 11]
    df = df.copy()
    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y

    # If a range was defined, we norm all points which are outside this range
    if plot_range is not None:
        y_range = plot_range[1]
        x_range = plot_range[0]
        df[x] = df[x].apply(lambda val: val if val <= x_range else x_range)
        df[y] = df[y].apply(lambda val: val if val <= y_range else y_range)
    else:
        y_range = df[y].max()
        x_range = df[x].max()

    # Create the scatter plot
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Add mean and std of x in a note on the right bottom
    ax.annotate(
        f"$\mu_B = {latex_exp(x_mean)}$\n$\sigma^2_B = {latex_exp(x_var)}$",  # noqa: W605
        xy=(0.7, 0.05),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    # Add mean and std of y in a note on the left top
    ax.annotate(
        f"$\mu_C = {latex_exp(y_mean)}$\n$\sigma^2_C = {latex_exp(y_var)}$",  # noqa: W605
        xy=(0.7, 0.25),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    if markers is not None:
        for m, d in df.groupby(markers):
            legend_label = d.iloc[0][label] if label is not None else None
            sc = ax.scatter(
                d[x],
                d[y],
                marker=m,
                c=d[color] if color is not None else "black",
                s=d[size] if size is not None else None,
                cmap=cmap if cmap is not None else None,
                norm=norm if color is not None else None,
                label=legend_label,
            )

    if marker is not None:
        sc = ax.scatter(
            df[x],
            df[y],
            marker=marker,
            c=df[color] if color is not None else "black",
            s=df[size] if size is not None else None,
            cmap=cmap if cmap is not None else None,
            norm=norm if norm is not None else None,
            label=label if label is not None else None,
        )

    # Check if a colorbar should be added
    if color is not None and colorbar:
        axins = inset_axes(
            ax, width="5%", height="100%", loc="center right", borderpad=-5
        )
        fig.colorbar(sc, cax=axins, orientation="vertical")

    if legend:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Draw diagonal lines
    ax.axline((0, 0), slope=1, color="dimgrey", linestyle="dashed")
    for border_step in borders:
        ax.axline((0, border_step), slope=1, color="lightgrey", linestyle="dotted")
        ax.axline((border_step, 0), slope=1, color="lightgrey", linestyle="dotted")
        # Draw helper text
        text_rotation, xytext = calculate_text_decoration(
            0, border_step, 1, x_range, y_range, 1, figsize
        )
        ax.text(
            xytext[0],
            xytext[1],
            f"+{border_step}",
            ha="center",
            va="top",
            fontsize=10,
            rotation=text_rotation,
            rotation_mode="anchor",
            transform_rotates_text=True,
        )

        text_rotation, xytext = calculate_text_decoration(
            border_step, 0, 1, x_range, y_range, -1, figsize
        )
        ax.text(
            xytext[0],
            xytext[1],
            f"-{border_step}",
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=text_rotation,
            rotation_mode="anchor",
            transform_rotates_text=True,
        )

    # Align axis
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    if plot_range is not None:
        ax.set_ylim(top=plot_range[1])
        ax.set_xlim(right=plot_range[0])

    return fig, ax


# Helper for high quality plots
def set_size(width, fraction=1, subplots=(1, 1), ratio=None, extra_spacing=(0, 0)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    ratio: float, optional
            Ratio of the figure. If not given golden ration is assumed
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 370
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    if ratio is None:
        ratio = golden_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt + extra_spacing[1] * subplots[1]
    # Figure height in inches
    fig_height_in = (
        fig_width_in * ratio * (subplots[0] / subplots[1])
        + extra_spacing[0] * subplots[0]
    )

    return fig_width_in, fig_height_in


def plot_training_data_summaries(
    df: pd.DataFrame,
    base_plot_dir: str,
    file_postfix: str,
    ignored_columns=None,
    file_type: str = "png",
    with_title: bool = False,
):
    if ignored_columns is None:
        ignored_columns = []
    logging.info("Plotting training point summaries")
    columns = [col for col in df.columns if col not in ignored_columns]
    df = df[columns]

    if len(df) > 10000:
        df = df.sample(10000)

    g = sns.PairGrid(df)
    if with_title:
        g.fig.suptitle(f"Training point summaries (n={len(df)}), (b={file_postfix})")
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True)
    g.savefig(
        f"{base_plot_dir}/summary_{file_postfix}.{file_type}",
        bbox_inches="tight",
        dpi=300,
    )


def plot_training_data_distributions(
    df: pd.DataFrame,
    base_plot_dir: str,
    file_postfix: str,
    ignored_columns=None,
    file_type: str = "png",
    figsize=set_size("thesis"),
    with_title: bool = False,
):
    if ignored_columns is None:
        ignored_columns = []
    logging.info("Plotting training point distributions")
    columns = [col for col in df.columns if col not in ignored_columns]
    df = df[columns]

    if len(df) > 10000:
        df = df.sample(10000)

    for col in columns:
        fig, ax = plt.subplots(figsize=figsize)
        plot_hist(ax, col, df, file_postfix, with_title)
        plt.savefig(
            f"{base_plot_dir}/distributions_{col}_{file_postfix}.{file_type}",
            bbox_inches="tight",
        )
        plt.close()


def plot_hist(ax, col, df, file_postfix, with_title):
    sns.histplot(df[col], ax=ax, stat="density", bins=52)
    if with_title:
        plt.title(f"Distribution of {col} (n={len(df[col])}), (b={file_postfix})")


def plot_dist_subplots(
    cols: int,
    df: pd.DataFrame,
    columns: List[str],
    file_postfix: str,
    base_plot_dir: str,
    file_type,
    with_title: bool = False,
):
    copy_columns = columns.copy()
    if "week" in copy_columns:
        copy_columns.remove("week")
    copy_columns.append("date")
    df = df[copy_columns]
    rows = math.ceil(len(columns) / cols)
    figsize = set_size("thesis", subplots=(rows, cols), extra_spacing=(1, 1.5))
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=figsize)
    assert len(axes.flatten()) >= len(columns)
    ax = axes.flatten()
    for i, col in enumerate(columns):
        if col == "week":
            hist_week(
                df=df,
                figsize=None,
                file_postfix=file_postfix,
                with_title=with_title,
                ax=ax[i],
            )
        else:
            plot_hist(ax[i], col, df, file_postfix, with_title)

    for i in range(len(columns), len(ax)):
        ax[i].axis("off")

    fig.align_ylabels(axes[:, :])
    fig.align_xlabels(axes[:, :])
    plt.savefig(
        f"{base_plot_dir}/distributions_subplots_{file_postfix}.{file_type}",
        bbox_inches="tight",
    )
    plt.close()


def plot_training_data_correlations(
    df: pd.DataFrame,
    base_plot_dir: str,
    file_postfix: str,
    ignored_columns=None,
    file_type: str = "png",
    figsize=set_size("thesis"),
    with_title: bool = False,
):
    if ignored_columns is None:
        ignored_columns = []
    logging.info("Plotting training point correlations")
    columns = [col for col in df.columns if col not in ignored_columns]
    df = df[columns]

    if len(df) > 10000:
        df = df.sample(10000)

    # Calculate the correlation matrix
    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    # Use the heatmap function from seaborn to plot the correlation matrix
    ax = sns.heatmap(
        corr,
        ax=ax,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True,
    )
    ax.figure.tight_layout()

    if with_title:
        ax.figure.suptitle(
            f"Training point correlations (n={len(df)}), (b={file_postfix})"
        )

    # Show the plot
    plt.savefig(
        f"{base_plot_dir}/correlations_{file_postfix}.{file_type}",
    )
    ax.figure.clf()
    plt.close()


def plot_training_data_hist(
    df: pd.DataFrame,
    base_plot_dir: str,
    file_postfix: str,
    ignored_columns=None,
    file_type: str = "png",
    figsize=set_size("thesis"),
    with_title: bool = False,
):
    if ignored_columns is None:
        ignored_columns = []
    # Make sure that date column is not removed
    ignored_columns.remove("date")
    logging.info("Plotting training point histograms")
    columns = [col for col in df.columns if col not in ignored_columns]
    df = df[columns]

    if len(df) > 10000:
        df = df.sample(10000)

    hist_week(df, figsize, file_postfix, with_title)
    plt.savefig(
        f"{base_plot_dir}/hist_week_{file_postfix}.{file_type}",
    )
    plt.close()

    hist_month(df, figsize, file_postfix, with_title)
    plt.savefig(
        f"{base_plot_dir}/hist_month_{file_postfix}.{file_type}",
    )
    plt.close()


def hist_week(df, figsize, file_postfix, with_title, ax=None):
    df["week"] = df["date"].apply(lambda x: int(x.strftime("%U")))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df["week"], ax=ax, bins=52, stat="density")
    plt.xlabel("Week number")
    if with_title:
        plt.title(f"Training point histogram (n={len(df)}), (b={file_postfix})")


def hist_month(df, figsize, file_postfix, with_title, ax=None):
    df["month"] = df["date"].apply(lambda x: int(x.strftime("%m")))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df["month"], ax=ax, bins=12, stat="density")
    plt.xlabel("Month number")
    if with_title:
        plt.title(f"Training point histogram (n={len(df)}), (b={file_postfix})")
