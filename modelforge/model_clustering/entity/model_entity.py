import os
import zlib
from abc import ABC
from functools import lru_cache
from typing import Type

import joblib
import pandas as pd
from joblib import Memory
from pandas.core.util.hashing import hash_pandas_object
from sklearn.pipeline import Pipeline

from modelforge.model_clustering.entity.loss import (
    LossFunction,
    compute_loss_ignore_nan,
)


class ModelEntity(ABC):
    _path: str = None
    _memory: Memory = Memory("./.cache", verbose=0)
    _id: str = None

    def __init__(
        self,
        path: str,
        id: str = None,
        pipeline: Pipeline = None,
        loss: LossFunction = None,
        train_x: pd.DataFrame = None,
        train_y: pd.Series = None,
        test_x: pd.DataFrame = None,
        test_y: pd.Series = None,
        train_raw: pd.DataFrame = None,
        test_raw: pd.DataFrame = None,
        feature_list: list[str] = None,
        metadata: dict = None,
    ):
        if path is None:
            raise ValueError("path cannot be None")
        self._path = path

        # Make sure the path exists
        os.makedirs(self._path, exist_ok=True)

        if id is not None:
            self.id = id
        if pipeline is not None:
            self.pipeline = pipeline
        if loss is not None:
            self.loss = loss
        if train_x is not None:
            self.train_x = train_x
        if train_y is not None:
            self.train_y = train_y
        if test_x is not None:
            self.test_x = test_x
        if test_y is not None:
            self.test_y = test_y
        if train_raw is not None:
            self.train_raw = train_raw
        if test_raw is not None:
            self.test_raw = test_raw
        if feature_list is not None:
            self.feature_list = feature_list
        if metadata is not None:
            self.metadata = metadata

    @property
    def id(self) -> str:
        if self._id is None:
            self._id = self._load_from_disk("id")
        return self._id

    @id.setter
    def id(self, id: str):
        self._id = id
        self._save_to_disk("id", id)

    @property
    def pipeline(self) -> Pipeline:
        return self._load_from_disk("pipeline")

    @pipeline.setter
    def pipeline(self, pipeline: Pipeline):
        self._save_to_disk("pipeline", pipeline)

    @property
    def loss(self) -> LossFunction:
        return self._load_from_disk("loss")

    @loss.setter
    def loss(self, loss: LossFunction):
        self._save_to_disk("loss", loss)

    @property
    def train_x(self) -> pd.DataFrame:
        return self._load_from_disk("train_x")

    @train_x.setter
    def train_x(self, train_x: pd.DataFrame):
        self._save_to_disk("train_x", train_x)

    @property
    def train_y(self) -> pd.Series:
        return self._load_from_disk("train_y")

    @train_y.setter
    def train_y(self, train_y: pd.Series):
        self._save_to_disk("train_y", train_y)

    @property
    def test_x(self) -> pd.DataFrame:
        return self._load_from_disk("test_x")

    @test_x.setter
    def test_x(self, test_x: pd.DataFrame):
        self._save_to_disk("test_x", test_x)

    @property
    def test_y(self) -> pd.Series:
        return self._load_from_disk("test_y")

    @test_y.setter
    def test_y(self, test_y: pd.Series):
        self._save_to_disk("test_y", test_y)

    @property
    def train_raw(self) -> pd.DataFrame:
        return self._load_from_disk("train_raw")

    @train_raw.setter
    def train_raw(self, train_raw: pd.DataFrame):
        self._save_to_disk("train_raw", train_raw)

    @property
    def test_raw(self) -> pd.DataFrame:
        return self._load_from_disk("test_raw")

    @test_raw.setter
    def test_raw(self, test_raw: pd.DataFrame):
        self._save_to_disk("test_raw", test_raw)

    @property
    @lru_cache
    def feature_list(self) -> list[str]:
        return self._load_from_disk("feature_list")

    @feature_list.setter
    def feature_list(self, feature_list: list[str]):
        self._save_to_disk("feature_list", feature_list)

    @property
    @lru_cache
    def metadata(self) -> dict:
        return self._load_from_disk("metadata")

    @metadata.setter
    def metadata(self, metadata: dict):
        self._save_to_disk("metadata", metadata)

    def _load_from_disk(self, attribute: str):
        path = os.path.join(self._path, attribute)
        if attribute in ["train_x", "test_x", "train_raw", "test_raw"]:
            return pd.read_parquet(f"{path}.parquet")
        elif attribute in ["train_y", "test_y"]:
            return pd.read_parquet(f"{path}.parquet").iloc[:, 0]
        else:
            return joblib.load(path)

    def _save_to_disk(self, attribute: str, data):
        path = os.path.join(self._path, attribute)
        if attribute in ["train_x", "test_x", "train_raw", "test_raw"]:
            data.to_parquet(f"{path}.parquet")
        elif attribute in ["train_y", "test_y"]:
            data.to_frame().to_parquet(f"{path}.parquet")
        else:
            joblib.dump(data, path)

    def model_loss(
        self, x: pd.DataFrame = None, y: pd.Series = None, *_args, **_kwargs
    ) -> float:
        if x is None and y is not None:
            raise ValueError("x cannot be None if y is not None")
        if x is not None and y is None:
            raise ValueError("y cannot be None if x is not None")
        if x is None or y is None:
            return self.model_test_loss()
        return self.loss(y, self.predict(x))

    def predict(self, x: pd.DataFrame, *_args, **_kwargs) -> pd.Series:
        return self.pipeline.predict(x)

    def model_test_loss(self) -> float:
        return compute_loss_ignore_nan(
            self.test_y, self.predict(self.test_x), self.loss
        )

    def __hash__(self):
        obj_bytes = str(self.id).encode()
        return zlib.adler32(obj_bytes)

    def __post_init__(self):
        self.__hash__ = lambda: ModelEntity.__hash__(self)

    @classmethod
    def load(cls: Type, path: str):
        """
        Load a ModelEntity from a folder.

        Parameters
        ----------
        path : str
            Path to the folder containing the saved ModelEntity.

        Returns
        -------
        ModelEntity
            The loaded ModelEntity.
        """
        return cls(
            path=path,
        )

    @property
    def loss_minimize(self) -> bool:
        return True


def _predict(
    pipeline: Pipeline, x: pd.DataFrame, _entity_hash: int, _data_hash: int
) -> pd.Series:
    """
    Joblib memory needs a pure function to be cached. This function is a wrapper around the predict method of the
    estimator to allow for caching.
    """
    return pipeline.predict(x)


class MemorizedModelEntity(ModelEntity):
    _memory = Memory("./.cache", verbose=0)

    def __init__(self, path: str):
        super().__init__(path=path)

    def predict(self, x: pd.DataFrame, *_args, **_kwargs) -> pd.Series:
        entity_hash = super.__hash__(self)
        data_hash = hash_pandas_object(x).sum()
        # Again, we build a custom hash function to avoid hashing the memory object
        return self._memory.cache(_predict, ignore=["pipeline", "x"])(
            self.pipeline, x, _entity_hash=entity_hash, _data_hash=data_hash
        )
