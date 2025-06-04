from typing import List

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from modelforge.experiments.pipeline_builders.pipeline_builder import PipelineBuilder
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.transformer.featurizer.prediction_loss_set_featurizer import (
    PredictionLossSetFeaturizer,
)
from modelforge.model_clustering.transformer.sampler.set.set_sampler import (
    SetSampler,
)


class PredictionLossSetPipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        factory: PipelineFactory,
        cluster_mixin: ClusterAlgorithm,
        model_sampler: SetSampler = None,
        additional_processors: List[BaseEstimator] = None,
        loss_function: LossFunction = None,
        eval_loss: LossFunction = None,
    ):
        self.factory = factory
        self.cluster_mixin = cluster_mixin
        self.model_sampler = model_sampler
        self.additional_processors = additional_processors or []
        self.loss_function = loss_function
        self.eval_loss = eval_loss

    def build_pipeline(self, skip_cache=False, use_train=True) -> Pipeline:
        return self.factory.get_pipeline(
            self.cluster_mixin,
            [
                PredictionLossSetFeaturizer(
                    loss_function=self.loss_function,
                    client=self.factory.client,
                    model_sampler=self.model_sampler,
                    skip_cache=skip_cache,
                    use_train=use_train,
                )
            ]
            + self.additional_processors,
            self.eval_loss,
        )
