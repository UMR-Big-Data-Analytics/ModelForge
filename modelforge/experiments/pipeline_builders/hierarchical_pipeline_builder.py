import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from modelforge.experiments.pipeline_builders.pipeline_builder import PipelineBuilder
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.hierarchical_clustering import (
    HierarchicalClustering,
)
from modelforge.model_clustering.distances.distance import Distance


class HierarchicalPrecomputedDistPipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        factory: PipelineFactory,
        distance_matrix: np.ndarray,
        linkage: str = "ward",
        preprocessors: list[BaseEstimator] = None,
        eval_loss=None,
    ):
        self.factory = factory
        self.distance_matrix = distance_matrix
        self.linkage = linkage
        self.eval_loss = eval_loss
        self.preprocessors = preprocessors or []

    def build_pipeline(self) -> Pipeline:
        return self.factory.get_pipeline(
            HierarchicalClustering(
                linkage_method=self.linkage, distance_matrix=self.distance_matrix
            ),
            self.preprocessors,
            self.eval_loss,
        )


class HierarchicalDistPipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        factory: PipelineFactory,
        distance_measure: Distance,
        linkage: str = "complete",
        preprocessors: list[BaseEstimator] = None,
        eval_loss=None,
    ):
        self.factory = factory
        self.distance_measure = distance_measure
        self.linkage = linkage
        self.eval_loss = eval_loss
        self.preprocessors = preprocessors or []

    def build_pipeline(self) -> Pipeline:
        return self.factory.get_pipeline(
            HierarchicalClustering(
                linkage_method=self.linkage, distance_measure=self.distance_measure
            ),
            self.preprocessors,
            self.eval_loss,
        )
