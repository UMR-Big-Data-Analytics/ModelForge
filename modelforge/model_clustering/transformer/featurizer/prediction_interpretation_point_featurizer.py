from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import shap
from distributed import Client
from joblib import Memory
from sklearn.base import TransformerMixin

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.featurizer.base_sample_featurizer import (
    BaseSampleFeaturizer,
)
from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)
from modelforge.model_clustering.transformer.sampler.point.random_point_sampler import (
    RandomPointSampler,
)


class ShapAggregationStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    NONE = "none"


class PredictionInterpretationPointFeaturizer(BaseSampleFeaturizer):
    def __init__(
        self,
        sampler: PointSampler,
        shap_aggregation_strategy: ShapAggregationStrategy,
        shap_aggregation_axis: int = 0,
        preprocessors: List[TransformerMixin] = None,
        memory: Memory = None,
        client: Client = None,
        skip_cache: bool = False,
    ):
        super().__init__(sampler, preprocessors, memory, client)
        self.skip_cache = skip_cache
        self.shap_aggregation_strategy = shap_aggregation_strategy
        self.shap_aggregation_axis = shap_aggregation_axis

    def featurize(self, model_dataset: ModelDataSet) -> pd.DataFrame:
        sample, sample_x, sample_y, model_dataset_hash, sample_hash = self.get_samples(
            model_dataset
        )
        model_dataset_hash, train, train_x = self.preprocess_dataset(model_dataset)

        background_mask = RandomPointSampler(200).fit_transform(train_x)
        background_mask = self.convert_bool_features_to_int(background_mask)
        masker = shap.maskers.Independent(background_mask)

        sample_x = self.convert_bool_features_to_int(sample_x)

        if self.skip_cache:
            return generate_embedding(
                model_dataset,
                sample_x,
                self.shap_aggregation_strategy,
                self.shap_aggregation_axis,
                masker,
                self.client,
                model_dataset_hash,
                sample_hash,
            )

        embedding = self.memory.cache(
            generate_embedding, ignore=["client", "sample_x", "model_dataset"]
        )(
            model_dataset,
            sample_x,
            self.shap_aggregation_strategy,
            self.shap_aggregation_axis,
            masker,
            self.client,
            model_dataset_hash,
            sample_hash,
        )
        return embedding

    @staticmethod
    def convert_bool_features_to_int(df: pd.DataFrame):
        return df.map(lambda x: int(x) if isinstance(x, bool) else x)


class CallableModel:
    def __init__(self, model_entity: ModelEntity):
        self.model_entity = model_entity

    def __call__(self, data: pd.DataFrame):
        values = self.model_entity.predict(data)
        if isinstance(values, pd.DataFrame) and values.shape[1] > 1:
            return values.values[:, 0]
        return values


def get_shap_values(
    model_entity: ModelEntity,
    sample_x: pd.DataFrame,
    shap_aggregation_strategy: ShapAggregationStrategy,
    shap_aggregation_axis: int,
    masker: shap.maskers.Masker,
) -> np.ndarray:
    explainer = shap.Explainer(CallableModel(model_entity), masker)
    shap_values = explainer(sample_x).values
    if shap_aggregation_strategy == ShapAggregationStrategy.MEAN:
        return np.mean(shap_values, axis=shap_aggregation_axis)
    elif shap_aggregation_strategy == ShapAggregationStrategy.MEDIAN:
        return np.median(shap_values, axis=shap_aggregation_axis)
    elif shap_aggregation_strategy == ShapAggregationStrategy.MAX:
        return np.max(shap_values, axis=shap_aggregation_axis)
    elif shap_aggregation_strategy == ShapAggregationStrategy.MIN:
        return np.min(shap_values, axis=shap_aggregation_axis)
    elif shap_aggregation_strategy == ShapAggregationStrategy.NONE:
        return shap_values.reshape(-1)
    else:
        raise ValueError("Unknown shap aggregation strategy")


def generate_embedding(
    model_dataset: ModelDataSet,
    sample_x: pd.DataFrame,
    shap_aggregation_strategy: ShapAggregationStrategy,
    shap_aggregation_axis: int,
    masker: shap.maskers.Masker,
    client: Client,
    _model_dataset_hash: int,
    _sample_hash: int,
) -> pd.DataFrame:
    # Iterate over all models and predict the sample with the client
    results = []
    i = 1
    ids = model_dataset.model_entity_ids()
    for model_id in ids:
        result = client.submit(
            get_shap_values,
            model_dataset.model_entity_by_id(model_id),
            sample_x,
            shap_aggregation_strategy,
            shap_aggregation_axis,
            masker,
        )
        results.append(result)
        i += 1

    gathered_results = client.gather(results)
    assert len(gathered_results) == len(
        ids
    ), "Number of results should match number of models"
    # Create a dataframe with each row being the embedding for a single device
    embedding_df = pd.DataFrame(
        gathered_results,
        index=ids,
        columns=[f"shap_value_{i}" for i in range(len(gathered_results[0]))],
    )
    return embedding_df
