import abc
from abc import abstractmethod
from dataclasses import asdict
from typing import List

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
from modelforge.experiments.experiment_utils import run_experiments
from modelforge.experiments.multi_dataset_runner import (
    AbstractModelConsolidationExperimentDefinition,
    ModelConsolidationDatasetDefinition,
)
from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.experiments.pipeline_builders.prediction_loss_set_pipeline_builder import (
    PredictionLossSetPipelineBuilder,
)
from modelforge.experiments.pipeline_builders.prediction_value_set_pipeline_builder import (
    PredictionValueSetPipelineBuilder,
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
from modelforge.model_clustering.transformer.sampler.set.measure.quantile import (
    Quantile,
)
from modelforge.model_clustering.transformer.sampler.set.measure.variance import (
    Variance,
)
from modelforge.model_clustering.transformer.sampler.set.statistical_set_sampler import (
    StatisticalSetSampler,
)
from modelforge.model_clustering.transformer.sampler.set.uniform_set_sampler import (
    UniformSetSampler,
)
from modelforge.shared.logger import logger_factory


class AbstractModelConsolidationDimensionalityExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition, abc.ABC
):
    embedding_dimensionalities: List[int] = list(range(1, 21, 1))

    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)
        pipelines = {}
        for dim in self.embedding_dimensionalities:
            for name, pipeline in self.get_pipeline(
                dim, pipeline_factory, experiment, client
            ).items():
                pipelines[f"{name}{dim}"] = pipeline
        return pipelines

    @abstractmethod
    def get_pipeline(
        self,
        dim: int,
        pipeline_factory: PipelineFactory,
        experiment: ModelConsolidationDatasetDefinition,
        client: Client,
    ) -> dict[str, Pipeline]:
        raise NotImplementedError(
            "Method get_pipeline not implemented in class AbstractModelConsolidationDimensionalityExperimentDefinition"
        )


class ModelConsolidationPredictionLossSetDimensionalityExperimentDefinition(
    AbstractModelConsolidationDimensionalityExperimentDefinition
):

    def get_pipeline(
        self,
        dim: int,
        pipeline_factory: PipelineFactory,
        experiment: ModelConsolidationDatasetDefinition,
        client: Client,
    ) -> dict[str, Pipeline]:
        use_train = True
        # Quick fix as many training sets have no anomalies
        if experiment.dataset.target == "is_anomaly":
            use_train = False
        return {
            f"prediction_loss_set_uniform_target_entropy_": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(dim, TargetSelector(), Entropy()),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_loss_set"


class ModelConsolidationPredictionValueSetDimensionalityExperimentDefinition(
    AbstractModelConsolidationDimensionalityExperimentDefinition
):
    def get_pipeline(
        self,
        dim: int,
        pipeline_factory: PipelineFactory,
        experiment: ModelConsolidationDatasetDefinition,
        client: Client,
    ) -> dict[str, Pipeline]:
        return {
            f"prediction_value_set_max_target_var_": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(dim, TargetSelector(), Variance(), "max"),
            ).build_pipeline(skip_cache=False),
            f"prediction_value_set_uniform_target_median_": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(dim, TargetSelector(), Quantile(0.5)),
            ).build_pipeline(
                skip_cache=False
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_value_set"


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
            [0.2],
            RetrainingConsolidationStrategy(
                ImmoscoutHousePriceModel(), PandasModelEntityDataMerger()
            ),
        ),
        ModelConsolidationDatasetDefinition(
            "weather_probabilistic",
            weather_probabilistic,
            [0.05],
            RetrainingConsolidationStrategy(
                WeatherModelProbXGB(), PandasModelEntityDataMerger()
            ),
        ),
        ModelConsolidationDatasetDefinition(
            "anomaly",
            anomaly,
            [0.2],
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


if __name__ == "__main__":
    logger = logger_factory(__name__)
    experiments = [
        ModelConsolidationPredictionLossSetDimensionalityExperimentDefinition(),
        ModelConsolidationPredictionValueSetDimensionalityExperimentDefinition(),
    ]
    run_experiments(
        "point/results/embedding_dimensionality", experiments, get_datasets(logger)
    )
