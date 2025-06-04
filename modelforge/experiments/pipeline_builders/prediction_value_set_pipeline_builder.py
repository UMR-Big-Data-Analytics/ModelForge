from typing import List

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from modelforge.experiments.pipeline_builders.pipeline_builder import PipelineBuilder
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.transformer.featurizer.prediction_value_set_featurizer import (
    PredictionValueSetFeaturizer,
)
from modelforge.model_clustering.transformer.sampler.set.set_sampler import (
    SetSampler,
)


class PredictionValueSetPipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        factory: PipelineFactory,
        cluster_mixin: ClusterAlgorithm,
        sampler: SetSampler,
        additional_processors: List[BaseEstimator] = None,
        eval_loss: LossFunction = None,
    ):
        self.factory = factory
        self.cluster_mixin = cluster_mixin
        self.sampler = sampler
        self.additional_processors = additional_processors or []
        self.eval_loss = eval_loss

    def build_pipeline(self, skip_cache=False) -> Pipeline:
        return self.factory.get_pipeline(
            self.cluster_mixin,
            [
                PredictionValueSetFeaturizer(
                    client=self.factory.client,
                    model_sampler=self.sampler,
                    skip_cache=skip_cache,
                )
            ]
            + self.additional_processors,
            self.eval_loss,
        )
