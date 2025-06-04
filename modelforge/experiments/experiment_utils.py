import os
from dataclasses import asdict

import numpy as np
from distributed import LocalCluster
from dotenv import load_dotenv

from modelforge.datasets.anomaly_models.anomaly_model import (
    CustomParameters,
    RandomForestAnomalyDetector,
    SlidingWindowProcessor,
)
from modelforge.datasets.datasets import (
    anomaly_dataset,
    immoscout_house_price_dataset,
    weather_dataset_probabilistic_regression,
)
from modelforge.datasets.immoscout_models.house_price.immoscout_house_price_model import (
    ImmoscoutHousePriceModel,
)
from modelforge.datasets.weather_models.weather_model import WeatherModelProbXGB
from modelforge.experiments.multi_dataset_runner import (
    ModelConsolidationDatasetDefinition,
    MultiDatasetRunner,
)
from modelforge.model_clustering.consolidation.retraining_consolidation_strategy import (
    RetrainingConsolidationStrategy,
)
from modelforge.model_clustering.evaluator.model_entity_data_merger import (
    PandasModelEntityDataMerger,
)
from modelforge.shared.logger import logger_factory


def get_datasets(logger):
    logger.info("Load datasets")
    house_price = immoscout_house_price_dataset()
    weather_probabilistic = weather_dataset_probabilistic_regression()
    anomaly = anomaly_dataset("anomaly")
    logger.info("Define datasets and experiments")
    params = CustomParameters()

    datasets = [
        ModelConsolidationDatasetDefinition(
            "house_price",
            house_price,
            np.arange(0.01, 0.5, 0.01).tolist(),
            RetrainingConsolidationStrategy(
                ImmoscoutHousePriceModel(), PandasModelEntityDataMerger()
            ),
        ),
        ModelConsolidationDatasetDefinition(
            "weather_probabilistic",
            weather_probabilistic,
            np.arange(0.005, 0.25, 0.005).tolist(),
            RetrainingConsolidationStrategy(
                WeatherModelProbXGB(), PandasModelEntityDataMerger()
            ),
        ),
        ModelConsolidationDatasetDefinition(
            "anomaly",
            anomaly,
            np.arange(0.01, 0.5, 0.01).tolist(),
            RetrainingConsolidationStrategy(
                RandomForestAnomalyDetector(
                    **asdict(params),
                    preprocessor=SlidingWindowProcessor(params.train_window_size),
                ),
                PandasModelEntityDataMerger(approximate=True),
            ),
        ),
    ]
    return datasets


def setup_cluster():
    load_dotenv()
    n_workers = int(os.getenv("DASK_N_WORKERS", "2"))
    logger = logger_factory("model_consolidation_experiment")
    logger.info("Setup cluster")
    client = LocalCluster(n_workers=n_workers, threads_per_worker=1).get_client()
    n_worker = len(client.scheduler_info()["workers"])
    logger.info(f"Cluster setup with {n_worker} workers")
    return client, logger


def run(base_path, client, datasets, experiments, logger):
    # Create folder for storing results
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for dataset in datasets:
        if not os.path.exists(f"{base_path}/{dataset.identifier}"):
            os.makedirs(f"{base_path}/{dataset.identifier}")
    logger.info("Run experiments")
    runner = MultiDatasetRunner(base_path, client, datasets, experiments, logger)
    runner.run()


def run_experiments(base_path, experiments, datasets=None):
    client, logger = setup_cluster()
    if datasets is None:
        datasets = get_datasets(logger)
    run(
        base_path=base_path,
        client=client,
        datasets=datasets,
        experiments=experiments,
        logger=logger,
    )
