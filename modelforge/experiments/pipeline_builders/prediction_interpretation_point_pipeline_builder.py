from typing import List

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from modelforge.experiments.pipeline_builders.pipeline_builder import PipelineBuilder
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.transformer.featurizer.prediction_interpretation_point_featurizer import (
    PredictionInterpretationPointFeaturizer,
    ShapAggregationStrategy,
)
from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


class PredictionInterpretationPointPipelineBuilder(PipelineBuilder):
    def __init__(
        self,
        factory: PipelineFactory,
        cluster_mixin: ClusterAlgorithm,
        sampler: PointSampler,
        preprocessors: List[BaseEstimator] = None,
        additional_processors: List[BaseEstimator] = None,
        eval_loss: LossFunction = None,
        shap_aggregation_strategy: ShapAggregationStrategy = ShapAggregationStrategy.NONE,
        shap_aggregation_axis: int = 0,
    ):
        self.factory = factory
        self.cluster_mixin = cluster_mixin
        self.sampler = sampler
        self.preprocessors = preprocessors or []
        self.additional_processors = additional_processors or []
        self.eval_loss = eval_loss
        self.shap_aggregation_strategy = shap_aggregation_strategy
        self.shap_aggregation_axis = shap_aggregation_axis

    def build_pipeline(self, skip_cache=False) -> Pipeline:
        return self.factory.get_pipeline(
            self.cluster_mixin,
            [
                PredictionInterpretationPointFeaturizer(
                    client=self.factory.client,
                    sampler=self.sampler,
                    preprocessors=self.preprocessors,
                    shap_aggregation_strategy=self.shap_aggregation_strategy,
                    shap_aggregation_axis=self.shap_aggregation_axis,
                    skip_cache=skip_cache,
                )
            ]
            + self.additional_processors,
            self.eval_loss,
        )
