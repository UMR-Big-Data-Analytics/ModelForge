from typing import List

from distributed import Client
from matplotlib import pyplot as plt
from pygments.lexers import go
from sklearn.pipeline import Pipeline

from modelforge.model_clustering.model_selection.base_pipeline_search import (
    BasePipelineSearch,
)


class ClusterAmountPipelineSearch(BasePipelineSearch):
    def __init__(
        self,
        pipelines: dict[str, Pipeline],
        client: Client,
        params,
    ):
        super().__init__(
            pipelines,
            client,
            params,
        )

    def plot_line(
        self,
        x_marker: List[float] = None,
        x_label: str = "Number of clusters",
        with_legend=True,
        backend="matplotlib",
    ):
        if backend == "plotly":
            return self.plot_line_plotly(x_marker, x_label, with_legend)

        if backend == "matplotlib":
            return self.plot_line_matplotlib(x_marker, x_label, with_legend)

        raise ValueError(f"Backend {backend} not supported")

    def plot_line_matplotlib(
        self,
        x_marker: List[float] = None,
        x_label: str = "Number of clusters",
        with_legend=True,
    ):
        fig, ax = plt.subplots()
        if x_marker is None:
            x_marker = self.params["modelclusterer__cluster_mixin__n_clusters"]
        for i, (name, params) in enumerate(self.results_.items()):
            scores = []
            for param in params:
                score = param["score"]
                x_mean, x_var, y_mean, y_var = (
                    score["model_loss"],
                    score["model_var"],
                    score["cluster_loss"],
                    score["cluster_var"],
                )
                scores.append(y_mean)
            ax.plot(
                x_marker,
                scores,
                label=name,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("$\\mu_C$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if with_legend:
            ax.legend()
        return fig

    def plot_line_plotly(
        self,
        x_marker: List[float] = None,
        x_label: str = "Number of clusters",
        with_legend=True,
    ):
        fig = go.Figure()

        if x_marker is None:
            x_marker = self.params["modelclusterer__cluster_mixin__n_clusters"]

        for name, params in self.results_.items():
            scores = []
            for param in params:
                score = param["score"]
                x_mean, x_var, y_mean, y_var = (
                    score["model_loss"],
                    score["model_var"],
                    score["cluster_loss"],
                    score["cluster_var"],
                )
                scores.append(y_mean)
            fig.add_trace(go.Scatter(x=x_marker, y=scores, mode="lines", name=name))

        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="$\\mu_C$",
            yaxis_tickformat=".2e",
            showlegend=with_legend,
        )

        return fig
