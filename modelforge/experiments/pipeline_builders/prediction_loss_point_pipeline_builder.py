from typing import List

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from modelforge.experiments.pipeline_builders.pipeline_builder import PipelineBuilder
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.transformer.featurizer.prediction_loss_point_featurizer import (
    PredictionLossPointFeaturizer,
)
from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


class PredictionLossPointPipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        factory: PipelineFactory,
        cluster_mixin: ClusterAlgorithm,
        sampler: PointSampler,
        loss: LossFunction,
        eval_loss: LossFunction = None,
        preprocessors: List[BaseEstimator] = None,
        additional_processors: List[BaseEstimator] = None,
    ):
        self.factory = factory
        self.cluster_mixin = cluster_mixin
        self.sampler = sampler
        self.loss = loss
        self.eval_loss = eval_loss
        self.preprocessors = preprocessors or []
        self.additional_processors = additional_processors or []

    def build_pipeline(self, skip_cache=False) -> Pipeline:
        return self.factory.get_pipeline(
            self.cluster_mixin,
            [
                PredictionLossPointFeaturizer(
                    client=self.factory.client,
                    sampler=self.sampler,
                    preprocessors=self.preprocessors,
                    loss_function=self.loss,
                    skip_cache=skip_cache,
                )
            ]
            + self.additional_processors,
            self.eval_loss,
        )
