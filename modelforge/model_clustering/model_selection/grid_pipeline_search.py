from typing import List

from distributed import Client
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.model_selection.base_pipeline_search import (
    BasePipelineSearch,
)
from modelforge.visualization.clustering_performance_plotter import (
    plot_score,
)


class GridPipelineSearch(BasePipelineSearch):
    def __init__(
        self,
        pipelines: dict[str, Pipeline],
        client: Client,
        factors: List[float] = None,
    ):
        super().__init__(pipelines, client, {})
        self.factors = factors if factors is not None else [0.01, 0.05, 0.1]

    def fit(self, dataset: ModelDataSet):
        self.params = (
            {
                "modelclusterer__cluster_mixin__n_clusters": (
                    [max(1, int(factor * dataset.size)) for factor in self.factors]
                )
            },
        )
        super().fit(dataset)

    def plot_grid(self, **plot_args):
        check_is_fitted(self, "results_")
        n_rows = len(self.pipelines)
        n_cols = len(self.factors)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        for i, (name, params) in enumerate(self.results_.items()):
            for j, param in enumerate(params):
                plot_score(param["score"], ax=axes[i, j], cmap=None, **plot_args)
                axes[i, j].set_title(
                    f"{name} - k={param['params']['modelclusterer__cluster_mixin__n_clusters']}/{self.factors[j] * 100}%"
                )
