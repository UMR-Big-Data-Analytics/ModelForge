from distributed import Client
from sklearn.pipeline import Pipeline

from modelforge.experiments.pipeline_builders.pipeline_builder import PipelineBuilder
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.transformer.featurizer.prediction_featurizer import (
    PredictionValuePointFeaturizer,
)
from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


class PredictionValuePointPipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        factory: PipelineFactory,
        cluster_mixin: ClusterAlgorithm,
        data_sampler: PointSampler,
        client: Client,
        preprocessors=None,
        postprocessors=None,
        eval_loss: LossFunction = None,
    ):
        self.cluster_mixin = cluster_mixin
        self.data_sampler = data_sampler
        self.client = client
        self.factory = factory
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.eval_loss = eval_loss

    def build_pipeline(self, skip_cache=False) -> Pipeline:
        return self.factory.get_pipeline(
            self.cluster_mixin,
            self.preprocessors
            + [
                PredictionValuePointFeaturizer(
                    self.data_sampler, client=self.client, skip_cache=skip_cache
                )
            ]
            + self.postprocessors,
            self.eval_loss,
        )
