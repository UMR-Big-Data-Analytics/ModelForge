from dataclasses import asdict

import numpy as np
from distributed import Client
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

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
from modelforge.experiments.experiment_utils import (
    run,
    setup_cluster,
)
from modelforge.experiments.multi_dataset_runner import (
    AbstractModelConsolidationExperimentDefinition,
    ModelConsolidationDatasetDefinition,
)
from modelforge.experiments.pipeline_builders.prediction_loss_set_pipeline_builder import (
    PredictionLossSetPipelineBuilder,
)
from modelforge.model_clustering.consolidation.representative_consolidation_strategy import (
    RepresentativeConsolidationStrategy,
)
from modelforge.model_clustering.consolidation.retraining_consolidation_strategy import (
    RetrainingConsolidationStrategy,
)
from modelforge.model_clustering.evaluator.model_entity_data_merger import (
    PandasModelEntityDataMerger,
)
from modelforge.model_clustering.transformer.sampler.set.criterion_selector.target_selector import (
    TargetSelector,
)
from modelforge.model_clustering.transformer.sampler.set.measure.entropy import (
    Entropy,
)
from modelforge.model_clustering.transformer.sampler.set.uniform_set_sampler import (
    UniformSetSampler,
)

n_samples = 10


class ModelConsolidationStrategyExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition
):
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)
        use_train = True
        # Because the training data of the anomaly dataset has no anomalies, we use the test data
        if experiment.dataset.target == "is_anomaly":
            use_train = False
        return {
            f"prediction_loss_set_uniform_target_entropy_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Entropy()),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_loss_set"


def get_datasets(logger, retraining: bool):
    logger.info("Load datasets")
    house_price = immoscout_house_price_dataset()
    weather_probabilistic = weather_dataset_probabilistic_regression()
    anomaly = anomaly_dataset("anomaly")
    logger.info("Define datasets and experiments")
    params = CustomParameters()

    datasets = [
        ModelConsolidationDatasetDefinition(
            "house_price_retraining" if retraining else "house_price_representative",
            house_price,
            np.arange(0.01, 0.5, 0.01).tolist(),
            (
                RetrainingConsolidationStrategy(
                    ImmoscoutHousePriceModel(), PandasModelEntityDataMerger()
                )
                if retraining
                else RepresentativeConsolidationStrategy()
            ),
        ),
        ModelConsolidationDatasetDefinition(
            (
                "weather_probabilistic_retraining"
                if retraining
                else "weather_probabilistic_representative"
            ),
            weather_probabilistic,
            np.arange(0.005, 0.25, 0.005).tolist(),
            (
                RetrainingConsolidationStrategy(
                    WeatherModelProbXGB(), PandasModelEntityDataMerger()
                )
                if retraining
                else RepresentativeConsolidationStrategy()
            ),
        ),
        ModelConsolidationDatasetDefinition(
            "anomaly_retraining" if retraining else "anomaly_representative",
            anomaly,
            np.arange(0.01, 0.5, 0.01).tolist(),
            (
                RetrainingConsolidationStrategy(
                    RandomForestAnomalyDetector(
                        **asdict(params),
                        preprocessor=SlidingWindowProcessor(params.train_window_size),
                    ),
                    PandasModelEntityDataMerger(approximate=True),
                )
                if retraining
                else RepresentativeConsolidationStrategy()
            ),
        ),
    ]
    return datasets


def run_experiments(base_path, experiments):
    client, logger = setup_cluster()
    datasets_retraining = get_datasets(logger, True)
    datasets_representative = get_datasets(logger, False)
    run(
        base_path=base_path,
        client=client,
        datasets=datasets_retraining + datasets_representative,
        experiments=experiments,
        logger=logger,
    )


if __name__ == "__main__":
    experiments = [
        ModelConsolidationStrategyExperimentDefinition(),
    ]
    run_experiments("point/results/consolidation_strategy", experiments)
