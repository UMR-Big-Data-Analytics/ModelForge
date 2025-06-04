from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import paired_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


@dataclass
class CustomParameters:
    train_window_size: int = 50
    # standardize: bool = False  # does not really influence the quality
    n_estimators: int = 100
    learning_rate: float = 0.1
    booster: str = "gbtree"  # "gblinear", "dart"
    tree_method: str = "auto"  # "exact", "approx", "hist"
    n_trees: int = 1
    max_depth: Optional[int] = None
    max_samples: Optional[float] = None  # Subsample ratio of the training instance
    colsample_bytree: Optional[float] = (
        None  # Subsample ratio of columns when constructing each tree.
    )
    colsample_bylevel: Optional[float] = (
        None  # Subsample ratio of columns for each level.
    )
    colsample_bynode: Optional[float] = (
        None  # Subsample ratio of columns for each split.
    )
    random_state: int = 42
    verbose: int = 0
    n_jobs: int = -1
    silent_warning: bool = True


class SlidingWindowProcessor(BaseEstimator, TransformerMixin):

    def __init__(self, window_size: int, standardize: bool = False):
        self.window_size = window_size
        if standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def fit(
        self, x: np.ndarray, _y: Optional[np.ndarray] = None, **_fit_params
    ) -> "SlidingWindowProcessor":
        if self.scaler:
            self.scaler.fit(x)
        return self

    def transform(
        self, x: np.ndarray, _y: Optional[np.ndarray] = None
    ) -> (np.ndarray, np.ndarray):
        """
        y is unused (exists for compatibility)
        """
        if self.scaler:
            x = self.scaler.transform(x)
        x = x.reshape(-1)
        # the last window would have no target to predict, e.g. for n=10: [[1, 2] -> 3, ..., [8, 9] -> 10, [9, 10] -> ?]
        # fix xGBoost error by creating a copy instead of a view
        # xgboost.core.xGBoostError: ../modelforge/c_api/../point/array_interface.h:234: Check failed: valid: Invalid strides in array.  strides: (1,1), shape: (3500, 100)
        new_x = sliding_window_view(x, window_shape=self.window_size)[:-1].copy()
        new_y = np.roll(x, -self.window_size)[: -self.window_size]
        return new_x, new_y

    def transform_y(self, x: np.ndarray) -> np.ndarray:
        if self.scaler:
            x = self.scaler.transform(x)
        return np.roll(x, -self.window_size)[: -self.window_size]

    def inverse_transform_y(
        self, y: np.ndarray, skip_inverse_scaling: bool = False
    ) -> np.ndarray:
        result = np.full(shape=self.window_size + len(y), fill_value=np.nan)
        result[-len(y) :] = y
        if not skip_inverse_scaling and self.scaler:
            result = self.scaler.inverse_transform(result)
        return result


class RandomForestAnomalyDetector(Pipeline):
    def __init__(
        self,
        preprocessor: SlidingWindowProcessor,
        n_estimators: float = 100,
        learning_rate: float = 0.01,
        booster: str = "gbtree",
        tree_method: str = "auto",
        n_trees: int = 1,
        max_depth: Optional[int] = None,
        max_samples: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        random_state: int = 42,
        verbose: int = 0,
        n_jobs: int = 1,
        prediction_type: str = "anomaly",
        preprocess=False,
        *_args,
        **_kwargs,
    ):
        super().__init__([])
        self.preprocessor = preprocessor
        self.clf = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            booster=booster,
            tree_method=tree_method,
            num_parallel_tree=n_trees,
            max_depth=max_depth,
            subsamples=max_samples,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            random_state=random_state,
            verbosity=verbose,
            n_jobs=n_jobs,
        )
        if prediction_type not in ["score", "anomaly"]:
            raise ValueError("prediction_type must be 'score' or 'anomaly'")
        self.prediction_type = prediction_type
        self.preprocess = preprocess

    def fit(
        self,
        x: pd.DataFrame,
        y=None,
        *_args,
        **_kwargs,
    ) -> "RandomForestAnomalyDetector":
        y = x["internal_y"].values
        x = x.drop(columns="internal_y")
        x = self.to_numpy(x)
        assert (
            x.shape[1] == self.preprocessor.window_size
        ), f"Expected {self.preprocessor.window_size} columns, got {x.shape[1]}"
        if self.preprocess:
            x, y = self.preprocessor.fit_transform(x)

        self.clf.fit(x, y)
        return self

    @staticmethod
    def to_numpy(x: pd.DataFrame) -> np.ndarray:
        return x.values

    @staticmethod
    def to_series(x: np.ndarray) -> pd.Series:
        return pd.Series(x)

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if self.prediction_type == "score":
            val = self.predict_score(x)
        else:
            val = self.predict_anomaly(x)
        # Remove nan values
        val = val.dropna()
        return val

    def predict_score(self, x: pd.DataFrame) -> pd.Series:
        t = type(x)
        # Special treatment for dataframes.
        # We expect the target computed by the preprocessor to be in the column 'internal_y'
        if t == pd.DataFrame:
            x = x.drop(columns="internal_y")
        x = self.to_numpy(x)
        if self.preprocess:
            x, _ = self.preprocessor.transform(x)
        y_hat = self._predict_internal(x)
        anomaly_scores = self.preprocessor.inverse_transform_y(y_hat)

        return self.to_series(anomaly_scores)

    def predict_anomaly(
        self, x: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> pd.Series:
        t = type(x)
        # Special treatment for dataframes.
        # We expect the target computed by the preprocessor to be in the column 'internal_y'
        if t == pd.DataFrame:
            y = x["internal_y"].values
            x = x.drop(columns="internal_y")
        x = self.to_numpy(x)
        if self.preprocess:
            x, y = self.preprocessor.transform(x)

        y_hat = self._predict_internal(x)
        scores = paired_distances(y.reshape(-1, 1), y_hat.reshape(-1, 1)).reshape(-1)
        # Fill nan values with zeros
        scores[np.isnan(scores)] = 0
        is_anomaly = self.preprocessor.inverse_transform_y(
            scores, skip_inverse_scaling=True
        )
        return self.to_series(is_anomaly)

    def get_params(self, deep=True):
        return self.clf.get_params(deep)

    def _predict_internal(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(x)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RandomForestAnomalyDetector":
        return joblib.load(path)
