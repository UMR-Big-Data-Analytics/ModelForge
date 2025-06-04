import json
import os
import shutil
from dataclasses import dataclass
from typing import Callable, List

import pandas as pd
from distributed import Client
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from modelforge.model_clustering.entity.clustered_model_dataset import (
    ClusteredModelDataSet,
)
from modelforge.shared.logger import logger_factory


@dataclass
class ModelConsolidationScore:
    cluster_loss: float
    model_loss: float
    cluster_var: float
    model_var: float
    num_clusters: int
    scoring: pd.DataFrame
    clustering: ClusteredModelDataSet

    @staticmethod
    def from_dict(d: dict):
        return ModelConsolidationScore(
            cluster_loss=d["cluster_loss"],
            model_loss=d["model_loss"],
            cluster_var=d["cluster_var"],
            model_var=d["model_var"],
            num_clusters=d["num_clusters"],
            scoring=d["scoring"],
            clustering=d["clustering"],
        )


@dataclass
class ModelConsolidationResult:
    params: dict
    score: ModelConsolidationScore

    def sanitize_params(self):
        def sanitize(d):
            if isinstance(d, dict):
                return {
                    "featurizer_runtime" if "featurizer_runtime" in k else k: sanitize(
                        v
                    )
                    for k, v in d.items()
                    if isinstance(v, (int, float, str, bool, type(None), dict, list))
                }
            elif isinstance(d, list):
                return [
                    sanitize(v)
                    for v in d
                    if isinstance(v, (int, float, str, bool, type(None), dict, list))
                ]
            else:
                return d

        return sanitize(self.params)

    def to_disk(self, path: str):
        num_clusters = self.score.num_clusters
        if os.path.exists(f"{path}/{num_clusters}"):
            shutil.rmtree(f"{path}/{num_clusters}")
        os.makedirs(f"{path}/{num_clusters}")
        with open(f"{path}/{num_clusters}/params.json", "w") as f:
            json.dump(self.sanitize_params(), f)
        scores = {
            "cluster_loss": self.score.cluster_loss,
            "model_loss": self.score.model_loss,
            "cluster_var": self.score.cluster_var,
            "model_var": self.score.model_var,
            "num_clusters": self.score.num_clusters,
        }
        with open(f"{path}/{num_clusters}/scores.json", "w") as f:
            json.dump(scores, f)

        self.score.scoring.to_csv(f"{path}/{num_clusters}/scoring.csv")
        clustering, embedding = self.score.clustering.to_dataframe()
        clustering.to_csv(f"{path}/{num_clusters}/clustering.csv")
        if embedding is not None:
            embedding.to_csv(f"{path}/{num_clusters}/embedding.csv")


@dataclass
class ModelConsolidationResultSet:
    results: List[ModelConsolidationResult]

    def append(self, result: ModelConsolidationResult):
        self.results.append(result)

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return self.results[item]

    def __repr__(self):
        return f"ModelConsolidationResultSet({self.results})"

    def to_disk(self, path: str):
        for result in self.results:
            result.to_disk(f"{path}")


class GridSearch(MetaEstimatorMixin):
    def __init__(
        self,
        pipeline: Pipeline,
        param_grid: dict | list | tuple | ParameterGrid,
        client: Client,
        scoring_aggregation: Callable = None,
    ):
        self.pipeline = pipeline
        self.param_grid = ParameterGrid(param_grid)
        self.client = client
        self.scoring_aggregation = scoring_aggregation
        self.candidates_ = ModelConsolidationResultSet([])
        self.logger = logger_factory(__name__)

    def fit(self, x, y=None):
        self.candidates_ = ModelConsolidationResultSet([])
        total_params = len(self.param_grid)
        for i, params in enumerate(self.param_grid):
            self.logger.info(f"GridSearch: Start {i}/{total_params}")
            self.pipeline.set_params(**params)
            self.pipeline.fit(x, y)
            score: dict = self.pipeline.score(x)
            clustering_params = score["clustering_pipeline_params"]
            params["clustering_pipeline_params"] = clustering_params
            consolidation_score = ModelConsolidationScore.from_dict(score)
            self.candidates_.append(
                ModelConsolidationResult(params, consolidation_score)
            )

        return self

    def transform(self, _x) -> ModelConsolidationResultSet:
        return self.params()

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def params(self) -> ModelConsolidationResultSet:
        if len(self.candidates_) == 0:
            raise ValueError(
                "The fit method must be called before the transform method"
            )
        return self.candidates_
