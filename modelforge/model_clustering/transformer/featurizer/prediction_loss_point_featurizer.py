from typing import List

import numpy as np
import pandas as pd
from distributed import Client
from joblib import Memory
from pandas import DataFrame
from sklearn.base import TransformerMixin

from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.featurizer.base_sample_featurizer import (
    BaseSampleFeaturizer,
)
from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)
from modelforge.shared.logger import logger_factory


class PredictionLossPointFeaturizer(BaseSampleFeaturizer):
    """
    This transformer takes a set dataset and preprocesses the point with a list of preprocessors.
    It then samples the point and predicts the samples with all models in the dataset.
    """

    def __init__(
        self,
        sampler: PointSampler,
        loss_function: LossFunction,
        preprocessors: List[TransformerMixin] = None,
        memory: Memory = None,
        client: Client = None,
        skip_cache: bool = True,
    ):
        super().__init__(sampler, preprocessors, memory, client, skip_cache)
        self.loss_function = loss_function
        self.client = client
        self.logger = logger_factory(__name__)

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
                sample_y,
                self.loss_function,
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
            sample_y,
            self.loss_function,
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


def apply_loss(
    loss_function: LossFunction,
    sample_y: pd.Series | np.ndarray,
    result: pd.Series | np.ndarray,
) -> pd.Series:
    if isinstance(sample_y, pd.Series):
        sample_y = sample_y.values
    if isinstance(result, pd.Series):
        result = result.values
    assert len(sample_y) == len(
        result
    ), "The length of the sample and the result should be the same"
    losses = np.zeros(len(sample_y))
    for i in range(len(sample_y)):
        losses[i] = loss_function(np.array([sample_y[i]]), np.array([result[i]]))
    return pd.Series(losses)


def prepare_sample_prediction_loss(
    model_entity: ModelEntity, sample_x: pd.DataFrame
) -> np.ndarray:
    pred = model_entity.predict(sample_x)
    if isinstance(pred, pd.Series):
        pred = pred.dropna().values
    if isinstance(pred, pd.DataFrame):
        pred = pred.dropna().values
    elif isinstance(pred, np.ndarray):
        # Remove nan values
        pred = pred[~np.isnan(pred)]
    # Assert that no nans are contained
    assert not np.isnan(pred).any(), "Prediction contains nan values"
    assert len(pred) == len(
        sample_x
    ), f"Prediction has length {len(pred)} while sample has length {len(sample_x)}"
    return pred


def generate_embedding(
    model_dataset: ModelDataSet,
    sample_x: pd.DataFrame,
    sample_y: pd.Series,
    loss_function: LossFunction,
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
            prepare_sample_prediction_loss,
            model_dataset.model_entity_by_id(model_id),
            sample_x,
        )
        result = client.submit(apply_loss, loss_function, sample_y, result)
        # Somehow, pandas cannot join the list of series correctly, resulting in nans. So we convert to numpy arrays
        result = client.submit(lambda x: x.values, result)
        results.append(result)
        i += 1

    gathered_results = client.gather(results)
    # Create a dataframe with each row being the embedding for a single device
    embedding_df = pd.DataFrame(
        gathered_results,
        index=ids,
        columns=[f"prediction_loss_{i}" for i in range(len(sample_x))],
    )

    assert (
        embedding_df.isna().sum().sum() == 0
    ), "There should be no nans in the embedding"
    return embedding_df
