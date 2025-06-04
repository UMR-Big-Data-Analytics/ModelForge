import pandas as pd
from sklearn.pipeline import Pipeline

from modelforge.model_clustering.consolidation.consolidation_strategy import (
    ConsolidationStrategy,
)
from modelforge.model_clustering.entity.model_dataset import ModelDataSet


class RepresentativeConsolidationStrategy(ConsolidationStrategy):
    def transform(self, cluster: ModelDataSet, y=None) -> Pipeline:
        measurements = []
        assert len(cluster.model_entities()) > 0
        for e1 in cluster.model_entities():
            for e2 in cluster.model_entities():
                score = e1.model_loss(e2.train_x, e2.train_y)
                measurements.append((e1.id, score))
        df = pd.DataFrame(measurements, columns=["id", "score"])
        df = df.groupby("id").mean()
        if cluster.model_entities()[0].loss_minimize:
            df = df.sort_values("score", ascending=True)
        else:
            df = df.sort_values("score", ascending=False)

        df = df.reset_index()

        entity_id = df.iloc[0]["id"]
        return cluster.model_entity_by_id(entity_id).pipeline
