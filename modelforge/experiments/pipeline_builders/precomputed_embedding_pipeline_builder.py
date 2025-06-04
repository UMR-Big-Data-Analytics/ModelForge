from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin

from modelforge.experiments.pipeline_builders.pipeline_builder import PipelineBuilder
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm


class PrecomputedEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, path: str):
        self.path = path

    def fit(self, _x, _y=None):
        return self

    def transform(self, _x):
        return load(self.path)


class PrecomputedEmbeddingPipelineBuilder(PipelineBuilder):
    def __init__(
        self, path: str, factory: PipelineFactory, cluster_mixin: ClusterAlgorithm
    ):
        self.path = path
        self.factory = factory
        self.cluster_mixin = cluster_mixin

    def build_pipeline(self):
        return self.factory.get_pipeline(
            self.cluster_mixin, [PrecomputedEmbeddingTransformer(self.path)]
        )
