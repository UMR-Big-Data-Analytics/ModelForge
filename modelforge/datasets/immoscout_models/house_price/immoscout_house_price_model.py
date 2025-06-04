import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBRegressor


class ImmoscoutHousePriceModel(Pipeline, BaseEstimator):
    """
    Model for predicting the advertised selling price of houses on immoscout24 point
    """

    def __init__(self, random_state=42):
        super().__init__([])
        self.regressor = make_pipeline(
            TransformedTargetRegressor(
                regressor=XGBRegressor(random_state=random_state),
                # Apply log1p to the target variable to make it more normally distributed and remove skewness
                func=np.log1p,
                inverse_func=np.expm1,
            )
        )
        self._is_fitted = False

    def fit(
        self, x_train: pd.DataFrame, y_train: pd.Series = None, **kwargs
    ) -> "ImmoscoutHousePriceModel":
        self.regressor.fit(x_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, x: pd.DataFrame, **kwargs) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Model is not fitted")
        pred = self.regressor.predict(x)
        if pd.isna(pred).any():
            raise ValueError("Model produced NaN predictions")
        return pred

    def get_params(self, deep=True) -> dict:
        return self.regressor.get_params(deep)

    def score(self, x: pd.DataFrame, y=None, sample_weight=None, **_params) -> float:
        if not self._is_fitted:
            raise ValueError("Model is not fitted")
        return self.regressor.score(x, y, sample_weight)
