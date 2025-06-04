from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost_distribution import XGBDistribution


class WeatherModel(Pipeline, BaseEstimator, ABC):
    @abstractmethod
    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray = None,
        **fit_params
    ):
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame | np.ndarray, **predict_params):
        pass


class WeatherModelProbXGB(WeatherModel):
    def __init__(
        self,
        distribution="normal",
        n_estimators=1000,
        early_stopping_rounds=10,
        validation_fraction=0.2,
    ):
        super().__init__([])
        self.regressor = XGBDistribution(
            distribution=distribution,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
        )
        self._is_fitted = False
        self.validation_fraction = validation_fraction

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray = None,
        **fit_params
    ):

        if self.validation_fraction > 0:
            train_features, val_features, train_targets, val_targets = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            self.regressor.fit(
                train_features,
                train_targets,
                eval_set=[(val_features, val_targets)],
            )
        else:
            self.regressor.fit(x, y, eval_set=[(x, y)])
        self._is_fitted = True

    def predict(
        self, x: pd.DataFrame | np.ndarray, **predict_params
    ) -> (pd.Series, pd.Series):
        preds = self.regressor.predict(x)
        mean, std = preds.loc, preds.scale
        return pd.Series(mean), pd.Series(std)

    def get_params(self, deep=True):
        return self.regressor.get_params(deep=deep)
