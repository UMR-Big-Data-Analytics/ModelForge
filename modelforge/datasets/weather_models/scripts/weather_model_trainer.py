from logging import Logger

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline

from modelforge.datasets.weather_models.weather_model import (
    WeatherModelNN,
    WeatherModelProbQRF,
    WeatherModelProbXGB,
    WeatherModelXGBoost,
)
from modelforge.datasets.weather_models.weather_model_entity import (
    WeatherModelEntity,
    WeatherModelProbEntity,
)
from modelforge.shared.losses import crps, crps_unpacking


class WeatherModelTrainer:
    def __init__(self, logger: Logger, processed_file_path: str, model_type: str):
        self.logger = logger
        self.processed_file_path = processed_file_path
        self.model_type = model_type
        if model_type not in ["xgboost", "nn"]:
            raise ValueError(f"Invalid model type {model_type}")

    def train(
        self, entity_id: str, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> WeatherModelEntity | None:
        if df_test.shape[0] < 300:
            return None
        self.logger.info(f"Training model for entity {entity_id}")
        train_y = df_train["obs"]
        train_x = df_train.drop(columns=["obs", "date", "station"])
        test_y = df_test["obs"]
        test_x = df_test.drop(columns=["obs", "date", "station"])

        self.logger.info(f"Train size: {len(train_x)} Test size: {len(test_x)}")

        try:
            pipeline = make_pipeline(
                (
                    WeatherModelXGBoost()
                    if self.model_type == "xgboost"
                    else WeatherModelNN(n_features=train_x.shape[1], n_outputs=2)
                ),
            )
            pipeline.fit(train_x, train_y)

            metadata = {
                "model_params": pipeline.get_params(),
                "features": list(train_x.columns),
                "train_size": len(train_x),
                "test_size": len(test_x),
                "model_metrics": {
                    "train_score": mean_absolute_error(
                        train_y, pipeline.predict(train_x)
                    ),
                    "test_score": mean_absolute_error(test_y, pipeline.predict(test_x)),
                    "train_r2": r2_score(train_y, pipeline.predict(train_x)),
                    "test_r2": r2_score(test_y, pipeline.predict(test_x)),
                    "train_median_absolute_error": median_absolute_error(
                        train_y, pipeline.predict(train_x)
                    ),
                    "test_median_absolute_error": median_absolute_error(
                        test_y, pipeline.predict(test_x)
                    ),
                    "train_mean_absolute_error": mean_absolute_error(
                        train_y, pipeline.predict(train_x)
                    ),
                    "test_mean_absolute_error": mean_absolute_error(
                        test_y, pipeline.predict(test_x)
                    ),
                    "train_mean_absolute_percentage_error": mean_absolute_percentage_error(
                        train_y, pipeline.predict(train_x)
                    ),
                    "test_mean_absolute_percentage_error": mean_absolute_percentage_error(
                        test_y, pipeline.predict(test_x)
                    ),
                },
            }

            entity = WeatherModelEntity(
                path=f"{self.processed_file_path}/data/{entity_id}",
                id=entity_id,
                pipeline=pipeline,
                loss=mean_absolute_error,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                metadata=metadata,
            )
            self.logger.info(f"Model trained for entity {entity_id}")
            return entity
        except ValueError as e:
            self.logger.error(
                f"Failed to train model for entity {entity_id}. Error: {e}"
            )
            return None


class WeatherProbModelTrainer:
    def __init__(self, logger: Logger, processed_file_path: str, model_type="xgb"):
        self.logger = logger
        self.processed_file_path = processed_file_path
        self.model_type = model_type

    def train(
        self, entity_id: str, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> WeatherModelEntity | None:
        if df_test.shape[0] < 300:
            return None
        self.logger.info(f"Training model for entity {entity_id}")
        train_y = df_train["obs"]
        train_x = df_train.drop(columns=["obs", "date", "station"])
        test_y = df_test["obs"]
        test_x = df_test.drop(columns=["obs", "date", "station"])

        self.logger.info(f"Train size: {len(train_x)} Test size: {len(test_x)}")

        try:
            model = (
                WeatherModelProbQRF()
                if self.model_type == "qrf"
                else WeatherModelProbXGB()
            )
            pipeline = make_pipeline(model)
            pipeline.fit(train_x, train_y)
            train_mean, train_std = pipeline.predict(train_x)
            test_mean, test_std = pipeline.predict(test_x)

            metadata = {
                "model_params": pipeline.get_params(),
                "features": list(train_x.columns),
                "train_size": len(train_x),
                "test_size": len(test_x),
                "model_metrics": {
                    "train_score": crps(train_mean, train_std, train_y),
                    "test_score": crps(test_mean, test_std, test_y),
                    "train_crps": crps(train_mean, train_std, train_y),
                    "test_crps": crps(test_mean, test_std, test_y),
                    "train_r2": r2_score(train_y, train_mean),
                    "test_r2": r2_score(test_y, test_mean),
                    "train_median_absolute_error": median_absolute_error(
                        train_y, train_mean
                    ),
                    "test_median_absolute_error": median_absolute_error(
                        test_y, test_mean
                    ),
                    "train_mean_absolute_error": mean_absolute_error(
                        train_y, train_mean
                    ),
                    "test_mean_absolute_error": mean_absolute_error(test_y, test_mean),
                    "train_mean_absolute_percentage_error": mean_absolute_percentage_error(
                        train_y, train_mean
                    ),
                    "test_mean_absolute_percentage_error": mean_absolute_percentage_error(
                        test_y, test_mean
                    ),
                },
            }

            entity = WeatherModelProbEntity(
                path=f"{self.processed_file_path}/data/{entity_id}",
                id=entity_id,
                pipeline=pipeline,
                loss=crps_unpacking,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                metadata=metadata,
                feature_list=train_x.columns.tolist(),
            )
            self.logger.info(f"Model trained for entity {entity_id}")
            return entity
        except ValueError as e:
            self.logger.error(
                f"Failed to train model for entity {entity_id}. Error: {e}"
            )
            return None
