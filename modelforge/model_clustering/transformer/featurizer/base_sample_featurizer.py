import abc
import zlib
from typing import List

import numpy as np
import pandas as pd
from distributed import Client, LocalCluster
from joblib import Memory
from pandas.core.util.hashing import hash_pandas_object
from sklearn.base import TransformerMixin

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.featurizer.featurizer import Featurizer
from modelforge.model_clustering.transformer.sampler.point.point_sampler import (
    PointSampler,
)


def preprocess(
    model_dataset: ModelDataSet,
    preprocessors: List[TransformerMixin],
    _model_dataset_hash: int,
    _transformer_hash: int,
):
    # Get the training point of all models in one dataframe
    train_x = []
    train_y = []
    for model_entity in model_dataset.model_entities():
        train_x.append(model_entity.train_x)
        train_y.append(model_entity.train_y)
    train_x = pd.concat(train_x, ignore_index=True)
    train_y = pd.concat(train_y, ignore_index=True)
    # Stich together the training point
    train = pd.concat([train_x, train_y], axis=1)

    for preprocessor in preprocessors:
        train = preprocessor.fit_transform(train)
    return train, train_x, train_y


def prepare_sample_prediction(
    model_entity: ModelEntity, sample_x: pd.DataFrame
) -> np.ndarray:
    pred = model_entity.predict(sample_x)
    if isinstance(pred, pd.Series):
        pred = pred.dropna().values
    if isinstance(pred, pd.DataFrame):
        pred = pred.dropna().values
        if pred.shape[1] > 1:
            pred = pred[:, 0]
    elif isinstance(pred, np.ndarray):
        # Remove nan values
        pred = pred[~np.isnan(pred)]
    # Assert that no nans are contained
    assert not np.isnan(pred).any(), "Prediction contains nan values"
    assert len(pred) == len(
        sample_x
    ), f"Prediction has length {len(pred)} while sample has length {len(sample_x)}"
    return pred


class BaseSampleFeaturizer(Featurizer, abc.ABC):
    def __init__(
        self,
        sampler: PointSampler,
        preprocessors: List[TransformerMixin] = None,
        memory: Memory = None,
        client: Client = None,
        skip_cache=True,
    ):
        super().__init__()
        self.sampler = sampler
        if preprocessors is None:
            preprocessors = []
        self.preprocessors = preprocessors
        self.memory = memory
        if memory is None:
            self.memory = Memory(location="./.cache/", verbose=0)
        if client is None:
            client = LocalCluster(n_workers=1, threads_per_worker=1).get_client()
        self.client = client
        self.skip_cache = skip_cache

    def preprocess_dataset(self, model_dataset: ModelDataSet):
        model_dataset_hash = model_dataset.__hash__()
        # Create a hash for the transformer by hashing the names of the preprocessors
        transformer_hash = zlib.adler32(
            (
                "SampleFeaturizer_"
                + "_".join(
                    [
                        preprocessor.__class__.__name__
                        for preprocessor in self.preprocessors
                    ]
                )
            ).encode()
        )
        if self.skip_cache:
            train, train_x, train_y = preprocess(
                model_dataset, self.preprocessors, model_dataset_hash, transformer_hash
            )
        else:
            train, train_x, train_y = self.memory.cache(
                preprocess,
                ignore=["preprocessors", "model_dataset"],
            )(
                model_dataset,
                self.preprocessors,
                model_dataset_hash,
                transformer_hash,
            )
        return model_dataset_hash, train, train_x

    def get_samples(
        self, model_dataset: ModelDataSet
    ) -> (pd.DataFrame, pd.DataFrame, pd.Series, int, int):
        model_dataset_hash, train, train_x = self.preprocess_dataset(model_dataset)
        assert len(train) > 0, "Training point must not be empty"
        sample = self.sampler.fit_transform(train, model_dataset=model_dataset)
        sample_x = sample[train_x.columns]
        target_col = sample.columns.difference(train_x.columns)[0]
        sample_y = sample[target_col]
        assert len(sample_x) == len(
            sample_y
        ), "Sample x and y must have the same length"
        assert isinstance(
            sample_y, pd.Series
        ), "Sample y must be a pandas series. Got {}".format(type(sample_y))

        sample_hash = hash_pandas_object(sample)

        return sample, sample_x, sample_y, model_dataset_hash, sample_hash
