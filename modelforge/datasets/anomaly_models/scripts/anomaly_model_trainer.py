import logging
from dataclasses import asdict

import pandas as pd
from sklearn.pipeline import make_pipeline

from modelforge.datasets.anomaly_models.anomaly_model import (
    CustomParameters,
    RandomForestAnomalyDetector,
    SlidingWindowProcessor,
)
from modelforge.datasets.anomaly_models.anomaly_model_entity import AnomalyModelEntity
from modelforge.datasets.anomaly_models.metrics import RocAUC


class AnomalyModelTrainer:
    def __init__(self, logger: logging.Logger, raw_path: str, processed_path: str):
        self.logger = logger
        self.raw_path = raw_path
        self.processed_path = processed_path

    def train(self, filename: str):
        basename = filename.replace(".test.csv", "")
        df_train = pd.read_csv(f"{basename}.train.csv")
        df_test = pd.read_csv(f"{basename}.test.csv")

        # Check if df_test has at least 1 anomaly value. Otherwise, no ROC AUC can be calculated on the test set
        if df_test["is_anomaly"].sum() == 0:
            self.logger.warn(
                f"Skipping model {filename} as it does not contain any anomalies"
            )
            return None

        params = CustomParameters()
        preprocessor = SlidingWindowProcessor(
            window_size=params.train_window_size, standardize=False
        )
        train_x, train_y = preprocessor.fit_transform(df_train["value"].values)
        test_x, test_y = preprocessor.transform(df_test["value"].values)

        columns = [f"window_{i}" for i in range(params.train_window_size)]

        train_x = pd.DataFrame(train_x, columns=columns)
        train_x = train_x.dropna()
        train_x["internal_y"] = train_y
        test_x = pd.DataFrame(test_x, columns=columns)
        test_x = test_x.dropna()
        test_x["internal_y"] = test_y

        loss = RocAUC().score
        metadata = asdict(params)

        for prediction_type in ["score", "anomaly"]:
            model = RandomForestAnomalyDetector(
                **metadata, preprocessor=preprocessor, prediction_type=prediction_type
            )
            model.fit(train_x, train_y)
            pipeline = make_pipeline(model)

            train_y = df_train["is_anomaly"]
            test_y = df_test["is_anomaly"]
            train_y = train_y[train_x.index]
            test_y = test_y[test_x.index]

            AnomalyModelEntity(
                path=f"{self.processed_path}/{prediction_type}/data/{basename.split('/')[-1]}",
                id=f"{basename}_{prediction_type}",
                pipeline=pipeline,
                loss=loss,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                metadata=metadata,
                feature_list=["value"],
            )
        self.logger.info(f"Finished training model {filename}")

        return filename
