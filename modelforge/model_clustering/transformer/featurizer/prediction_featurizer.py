from typing import List

import pandas as pd
from distributed import Client
from joblib import Memory
from pandas import DataFrame
from sklearn.base import TransformerMixin

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.transformer.featurizer.base_sample_featurizer import (
    BaseSampleFeaturizer,
    prepare_sample_prediction,
)
from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)
from modelforge.shared.logger import logger_factory


class PredictionValuePointFeaturizer(BaseSampleFeaturizer):
    """
    This transformer takes a set dataset and preprocesses the point with a list of preprocessors.
    It then samples the point and predicts the samples with all models in the dataset.
    """

    def __init__(
        self,
        sampler: PointSampler,
        preprocessors: List[TransformerMixin] = None,
        memory: Memory = None,
        client: Client = None,
        skip_cache: bool = False,
    ):
        super().__init__(sampler, preprocessors, memory, client)
        self.client = client
        self.logger = logger_factory(__name__)
        self.skip_cache = skip_cache

    def featurize(self, model_dataset: ModelDataSet) -> pd.DataFrame:
        """
        Preprocess the point and predict the samples with all models in the dataset.

        @parameter model_dataset: The set dataset to featurize
        @return: A dataframe with each row being the embedding for a single device
        """
        sample, sample_x, sample_y, model_dataset_hash, sample_hash = self.get_samples(
            model_dataset
        )
        if self.skip_cache:
            return generate_embedding(
                model_dataset,
                sample_x,
                self.client,
                model_dataset_hash,
                sample_hash,
            )

        embedding = self.memory.cache(
            generate_embedding,
            ignore=["client", "model_dataset", "sample_x"],
        )(
            model_dataset,
            sample_x,
            self.client,
            model_dataset_hash,
            sample_hash,
        )

        return embedding

    def __repr__(self, n_char_max=700):
        repr = ""
        for preprocessor in self.preprocessors:
            repr += preprocessor.__repr__()
        return f"PredictionFeaturizer(sampler={self.sampler.__repr__(n_char_max)}, preprocessors={repr})"


def generate_embedding(
    model_dataset: ModelDataSet,
    sample_x: pd.DataFrame,
    client: Client,
    _model_dataset_hash: int,
    _sample_hash: int,
) -> DataFrame:
    # Iterate over all models and predict the sample with the client
    results = []
    i = 1
    ids = model_dataset.model_entity_ids()
    for model_id in ids:
        result = client.submit(
            prepare_sample_prediction,
            model_dataset.model_entity_by_id(model_id),
            sample_x,
        )
        results.append(result)
        i += 1

    gathered_results = client.gather(results)
    # Create a dataframe with each row being the embedding for a single device
    embedding_df = pd.DataFrame(
        gathered_results,
        index=ids,
        columns=[f"prediction_{i}" for i in range(len(sample_x))],
    )
    return embedding_df
