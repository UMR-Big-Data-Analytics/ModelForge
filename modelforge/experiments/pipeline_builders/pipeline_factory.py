from typing import List

import pandas as pd
from distributed import Client
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline

from modelforge.model_clustering.cluster.cluster_algorithm import ClusterAlgorithm
from modelforge.model_clustering.cluster.model_clusterer import ModelClusterer
from modelforge.model_clustering.consolidation.consolidation_strategy import (
    ConsolidationStrategy,
)
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.evaluator.model_clustering_evaluator import (
    ModelClusteringEvaluator,
)


class PipelineFactory:
    def __init__(
        self,
        data: ModelDataSet,
        client: Client,
        consolidation_strategy: ConsolidationStrategy,
    ):
        self.data = data
        self.client = client
        self.consolidation_strategy = consolidation_strategy

    def get_pipeline(
        self,
        cluster_mixin: ClusterAlgorithm,
        preprocessors: List[BaseEstimator],
        eval_loss: LossFunction = None,
    ) -> Pipeline:
        ids = pd.Series(self.data.model_entity_ids())
        evaluator = ModelClusteringEvaluator(
            model_dataset=self.data,
            client=self.client,
            consolidation_strategy=self.consolidation_strategy,
            loss=eval_loss,
        )

        pipeline = make_pipeline(
            *preprocessors,
            ModelClusterer(cluster_mixin=cluster_mixin, ids=ids),
            evaluator
        )
        evaluator.set_clustering_pipeline_ref(pipeline)
        return pipeline
