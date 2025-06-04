from copy import deepcopy

from sklearn.pipeline import Pipeline

from modelforge.model_clustering.consolidation.consolidation_strategy import (
    ConsolidationStrategy,
)
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.evaluator.model_entity_data_merger import (
    ModelEntityDataMerger,
    PandasModelEntityDataMerger,
)


class RetrainingConsolidationStrategy(ConsolidationStrategy):
    def __init__(
        self, pipeline: Pipeline, entity_data_merger: ModelEntityDataMerger = None
    ):
        self.pipeline = pipeline
        if entity_data_merger is None:
            self.entity_data_merger = PandasModelEntityDataMerger()
        else:
            self.entity_data_merger = entity_data_merger

    def transform(self, cluster: ModelDataSet) -> Pipeline:
        assert len(cluster.model_entity_ids()) > 0, "Cluster is empty"

        train_x, train_y, test_x, test_y = self.entity_data_merger.merge(
            cluster.model_entities()
        )

        pipeline_clone = deepcopy(self.pipeline)

        # We also provide the test set to the pipeline, so that it can optionally be used for early stopping/
        # set validation or something similar
        pipeline_clone.fit(train_x, train_y, x_test=test_x, y_test=test_y)
        return pipeline_clone
