import zlib
from abc import abstractmethod
from typing import Callable

from distributed import Client, LocalCluster
from joblib import Memory
from scipy.spatial.distance import squareform

from modelforge.model_clustering.distances.distance import Distance
from modelforge.model_clustering.distances.pdist import pdist_obj
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity


class MemorizedDistance(Distance):
    def __init__(
        self,
        distance_measure: Callable[[ModelEntity, ModelEntity], float],
        memory: Memory = None,
        client: Client = None,
        skip_cache: bool = False,
    ):
        if memory is None:
            memory = Memory(location="./.cache/")
        if client is None:
            client = LocalCluster(n_workers=1, threads_per_worker=1).get_client()
        self.client = client
        self.memory = memory
        self.distance_measure = distance_measure
        self.skip_cache = skip_cache

    @abstractmethod
    def distance_function_str(self) -> str:
        raise NotImplementedError()

    def transform(self, model_dataset: ModelDataSet):
        if self.skip_cache:
            return _compute_distance_matrix(
                model_dataset, self.distance_measure, self.client, 0, 0
            )
        return self.memory.cache(
            _compute_distance_matrix,
            ignore=["model_dataset", "distance_measure", "client"],
        )(
            model_dataset,
            self.distance_measure,
            self.client,
            model_dataset.__hash__(),
            zlib.adler32(self.distance_function_str().encode()),
        )


def _compute_distance_matrix(
    model_dataset: ModelDataSet,
    distance_measure: Callable[[ModelEntity, ModelEntity], float],
    client: Client,
    _dataset_hash: int,
    _distance_measure_hash: int,
):
    models = [model_entity for model_entity in model_dataset.model_entities()]
    vector_distance_matrix = pdist_obj(models, distance_measure, client)
    return squareform(vector_distance_matrix)
