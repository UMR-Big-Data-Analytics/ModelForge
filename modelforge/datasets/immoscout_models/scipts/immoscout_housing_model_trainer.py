from logging import Logger

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from modelforge.datasets.immoscout_models.house_price.immoscout_house_price_model import (
    ImmoscoutHousePriceModel,
)
from modelforge.datasets.immoscout_models.house_price.immoscout_house_price_model_entity import (
    ImmoscoutHousePriceModelEntity,
)


class ImmoscoutHousingPriceModelTrainer:
    def __init__(self, logger: Logger, processed_file_path: str):
        self.logger = logger
        self.processed_file_path = processed_file_path

    def train(
        self, entity_id: str, df: pd.DataFrame
    ) -> ImmoscoutHousePriceModelEntity | None:
        self.logger.info(f"Training model for entity {entity_id}")
        y = df["kaufpreis"]
        X = df.drop(columns=["kaufpreis", "kid2015"])

        # Train the set with train test split
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.logger.info(f"Train size: {len(train_x)} Test size: {len(test_x)}")

        try:
            pipeline = make_pipeline(ImmoscoutHousePriceModel())
            pipeline.fit(train_x, train_y)

            metadata = {
                "model_params": pipeline.get_params(),
                "features": list(X.columns),
                "train_size": len(train_x),
                "test_size": len(test_x),
                "model_metrics": {
                    "train_score": mean_absolute_error(
                        train_y, pipeline.predict(train_x)
                    ),
                    "test_score": mean_absolute_error(test_y, pipeline.predict(test_x)),
                    "train_r2": pipeline.score(train_x, train_y),
                    "test_r2": pipeline.score(test_x, test_y),
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

            entity = ImmoscoutHousePriceModelEntity(
                path=f"{self.processed_file_path}/house_price/data/{entity_id}",
                id=entity_id,
                pipeline=pipeline,
                loss=mean_absolute_percentage_error,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                metadata=metadata,
                feature_list=X.columns.tolist(),
            )
            self.logger.info(f"Model trained for entity {entity_id}")
            return entity
        except ValueError as e:
            self.logger.error(
                f"Failed to train model for entity {entity_id}. Error: {e}"
            )
            return None
