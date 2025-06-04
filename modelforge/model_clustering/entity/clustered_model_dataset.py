from dataclasses import dataclass

import pandas as pd

from .model_dataset import ModelDataSet
from .model_entity import ModelEntity


@dataclass
class ClusteredModelDataSet:
    _clustered_model_entities: dict[int, ModelDataSet]
    _id_to_cluster: dict[int | str, int]
    _embedding: pd.DataFrame

    def __init__(self, clustered_model_entities: dict[int, ModelDataSet] = None):
        if clustered_model_entities is None:
            clustered_model_entities = {}
        self._id_to_cluster = {}
        for cluster_id, cluster in clustered_model_entities.items():
            for entity in cluster.model_entities():
                self._id_to_cluster[entity.id] = cluster_id
        self._clustered_model_entities = clustered_model_entities

    def add_cluster(self, cluster_id: int, model_data_set: ModelDataSet):
        if cluster_id in self._clustered_model_entities:
            raise ValueError(f"Cluster {cluster_id} already exists")
        self._clustered_model_entities[cluster_id] = model_data_set
        for entity in model_data_set.model_entities():
            self._id_to_cluster[entity.id] = cluster_id

    def get_cluster(self, cluster_id: int):
        return self._clustered_model_entities[cluster_id]

    def get_clusters(self):
        return self._clustered_model_entities.keys()

    def add_to_cluster(self, cluster_id: int, model_entity: ModelEntity):
        cluster = self.get_cluster(cluster_id)
        cluster.add_model_entity(model_entity)
        self._id_to_cluster[model_entity.id] = cluster_id

    def get_num_clusters(self):
        return len(self._clustered_model_entities)

    def get_clustered_model_entities(self):
        return self._clustered_model_entities.values()

    def predict(self, model_id: int | str):
        cluster_id = self._id_to_cluster.get(model_id, None)
        if cluster_id is None:
            raise ValueError(f"Model {model_id} not found")
        return cluster_id

    def get_embedding(self, with_cluster_labels: bool = False):
        if with_cluster_labels:
            return self._embedding
        return self._embedding.drop(columns=["cluster"])

    def to_dataframe(self) -> (pd.DataFrame, pd.DataFrame):
        data = []
        for cluster_id, model_data_set in self._clustered_model_entities.items():
            for entity in model_data_set.model_entities():
                data.append({"entity_id": entity.id, "cluster": cluster_id})
        df = pd.DataFrame(data)
        return df, self._embedding

    @staticmethod
    def from_dataframe(df: pd.DataFrame, source_dataset: ModelDataSet):
        required_columns = ["cluster"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Dataframe must contain columns {required_columns}")

        cluster_model_dataset = ClusteredModelDataSet()

        cluster_model_dataset._embedding = df

        for cluster_id, cluster in df.groupby("cluster"):
            cluster_dataset = ModelDataSet.from_iterable(
                [
                    source_dataset.model_entity_by_id(entity_id)
                    for entity_id in cluster.index
                ]
            )
            # Hashing an int is the same as the int itself, but this way we allow for more flexibility
            cluster_model_dataset.add_cluster(cluster_id.__hash__(), cluster_dataset)

        return cluster_model_dataset
