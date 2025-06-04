import zlib
from typing import List

import numpy as np
import pandas as pd
import shap
from distributed import Client
from joblib import Memory
from sklearn.utils.validation import check_is_fitted

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.featurizer.featurizer import Featurizer
from modelforge.model_clustering.transformer.featurizer.prediction_interpretation_point_featurizer import (
    CallableModel,
    ShapAggregationStrategy,
)
from modelforge.model_clustering.transformer.sampler.point.random_point_sampler import (
    RandomPointSampler,
)
from modelforge.model_clustering.transformer.sampler.set.set_sampler import (
    SetSampler,
)
from modelforge.shared.logger import logger_factory


def get_shap_values_single(
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


def get_shap_values(
    model_entity: ModelEntity,
    samples_x: List[pd.DataFrame],
    shap_aggregation_strategy: ShapAggregationStrategy,
    shap_aggregation_axis: int,
    masker: shap.maskers.Masker,
) -> np.ndarray:
    values = []
    for x in samples_x:
        val = np.mean(
            get_shap_values_single(
                model_entity,
                x,
                shap_aggregation_strategy,
                shap_aggregation_axis,
                masker,
            )
        )
        values.append(val)
    return np.array(values)


def generate_embedding(
    model_dataset: ModelDataSet,
    samples_x: List[pd.DataFrame],
    shap_aggregation_strategy: ShapAggregationStrategy,
    shap_aggregation_axis: int,
    masker: shap.maskers.Masker,
    client: Client,
    _model_dataset_hash: int,
    _column_ids_hash: int,
) -> pd.DataFrame:
    # Iterate over all models and predict the sample with the client
    results = []
    i = 1
    ids = model_dataset.model_entity_ids()
    for model_id in ids:
        result = client.submit(
            get_shap_values,
            model_dataset.model_entity_by_id(model_id),
            samples_x,
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


class PredictionInterpretationSetFeaturizer(Featurizer):
    def __init__(
        self,
        model_sampler: SetSampler = None,
        memory: Memory = None,
        client: Client = None,
        skip_cache: bool = False,
    ):
        super().__init__()
        self.model_sampler = model_sampler
        self.memory = memory
        self.client = client
        self.skip_cache = skip_cache
        self.logger = logger_factory(__name__)
        if memory is None:
            self.memory = Memory(location="./.cache/", verbose=0)

    def fit(self, model_dataset: ModelDataSet, _y=None):
        self.row_ids_ = model_dataset.model_entity_ids()
        if self.model_sampler is not None:
            self.column_ids_ = sorted(
                [
                    model_entity.id
                    for model_entity in self.model_sampler.get_sample_candidates(
                        model_dataset
                    )
                ]
            )
        else:
            self.column_ids_ = model_dataset.model_entity_ids()

        return self

    def featurize(self, model_dataset: ModelDataSet) -> pd.DataFrame:
        """
        Transform the set dataset into a feature matrix

        @param model_dataset: The set dataset to be featurized
        @return: A feature matrix
        """
        check_is_fitted(self)
        sample_x = pd.concat(
            [x.train_x for x in model_dataset.model_entities()]
        ).sample(200_000, random_state=42)
        background_mask = RandomPointSampler(200).fit_transform(sample_x)
        background_mask = self.convert_bool_features_to_int(background_mask)
        masker = shap.maskers.Independent(background_mask)

        sampled_model_train_x = []
        for entity_id in self.column_ids_:
            entity = model_dataset.model_entity_by_id(entity_id)
            # Computing for large datasets can be slow, so we sample 100 rows
            if len(entity.train_x) > 100:
                train_x = entity.train_x.sample(100)
            else:
                train_x = entity.train_x
            train_x_entity = self.convert_bool_features_to_int(train_x)
            sampled_model_train_x.append(train_x_entity)

        # Joblib memory cache does not support hashing of functions or non numpy object, so we need to pass the hash
        # of the function and dataset as arguments, which are not used in the function. The original arguments of the
        # loss function and set dataset are ignored by joblib.memory.
        # Moreover, memory only supports pure functions and not methods, so we delegate the method to a function.
        model_dataset_hash = model_dataset.__hash__()
        row_ids = zlib.adler32(str(self.row_ids_).encode())
        column_ids = zlib.adler32(str(self.column_ids_).encode())
        column_ids_hash = zlib.adler32(str(column_ids).encode())

        if self.skip_cache:
            return generate_embedding(
                model_dataset,
                sampled_model_train_x,
                ShapAggregationStrategy.MEAN,
                1,
                masker,
                self.client,
                model_dataset_hash,
                column_ids_hash,
            )

        embedding = self.memory.cache(
            generate_embedding,
            ignore=[
                "client",
                "samples_x",
                "model_dataset",
                "masker",
                "shap_aggregation_strategy",
                "shap_aggregation_axis",
            ],
        )(
            model_dataset,
            sampled_model_train_x,
            ShapAggregationStrategy.MEAN,
            1,
            masker,
            self.client,
            model_dataset_hash,
            column_ids_hash,
        )
        return embedding

    def __repr__(self, n_char_max=700):
        return f"ShapSetFeaturizer(model_sampler={self.model_sampler.__str__()})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def convert_bool_features_to_int(df: pd.DataFrame):
        return df.map(lambda x: int(x) if isinstance(x, bool) else x)
