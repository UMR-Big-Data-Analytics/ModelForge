from distributed import Client
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from modelforge.experiments.experiment_utils import run_experiments
from modelforge.experiments.multi_dataset_runner import (
    AbstractModelConsolidationExperimentDefinition,
    ModelConsolidationDatasetDefinition,
)
from modelforge.experiments.pipeline_builders.hierarchical_pipeline_builder import (
    HierarchicalDistPipelineBuilder,
)
from modelforge.experiments.pipeline_builders.prediction_interpretation_point_pipeline_builder import (
    PredictionInterpretationPointPipelineBuilder,
)
from modelforge.experiments.pipeline_builders.prediction_interpretaton_set_pipeline_builder import (
    PredictionInterpretationSetPipelineBuilder,
)
from modelforge.experiments.pipeline_builders.prediction_loss_point_pipeline_builder import (
    PredictionLossPointPipelineBuilder,
)
from modelforge.experiments.pipeline_builders.prediction_loss_set_pipeline_builder import (
    PredictionLossSetPipelineBuilder,
)
from modelforge.experiments.pipeline_builders.prediction_value_point_pipeline_builder import (
    PredictionValuePointPipelineBuilder,
)
from modelforge.experiments.pipeline_builders.prediction_value_set_pipeline_builder import (
    PredictionValueSetPipelineBuilder,
)
from modelforge.model_clustering.distances.cross_performance import (
    CrossPerformanceDistance,
)
from modelforge.model_clustering.transformer.featurizer.prediction_interpretation_point_featurizer import (
    ShapAggregationStrategy,
)
from modelforge.model_clustering.transformer.sampler.point.linear_point_sampler import (
    LinearPointSampler,
)
from modelforge.model_clustering.transformer.sampler.point.min_max_point_sampler import (
    MinMaxPointSampler,
)
from modelforge.model_clustering.transformer.sampler.point.percentile_point_sampler import (
    PercentilePointSampler,
)
from modelforge.model_clustering.transformer.sampler.point.random_point_sampler import (
    RandomPointSampler,
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
from modelforge.model_clustering.transformer.sampler.set.random_set_sampler import (
    RandomSetSampler,
)
from modelforge.model_clustering.transformer.sampler.set.statistical_set_sampler import (
    StatisticalSetSampler,
)
from modelforge.model_clustering.transformer.sampler.set.uniform_set_sampler import (
    UniformSetSampler,
)
from modelforge.shared.losses import crps_unpacking

n_samples = 10


class ModelConsolidationPredictionInterpretationSetExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition
):
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)

        return {
            f"prediction_interpretation_set_random_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                RandomSetSampler(n_samples),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_min_target_var_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Variance(), "min"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_max_target_var_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Variance(), "max"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_min_target_median_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(
                    n_samples, TargetSelector(), Quantile(0.5), "min"
                ),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_max_target_median_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(
                    n_samples, TargetSelector(), Quantile(0.5), "max"
                ),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_min_target_entropy_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Entropy(), "min"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_max_target_entropy_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Entropy(), "max"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_uniform_target_var_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Variance()),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_uniform_target_median_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Quantile(0.5)),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_interpretation_set_uniform_target_entropy_{n_samples}": PredictionInterpretationSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Entropy()),
            ).build_pipeline(
                skip_cache=False
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_interpretation_set"


class ModelConsolidationPredictionInterpretationPointExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition
):
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)

        return {
            f"prediction_interpretation_random_mean_sample_{n_samples}": PredictionInterpretationPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                RandomPointSampler(n_samples),
                shap_aggregation_strategy=ShapAggregationStrategy.MEAN,
                shap_aggregation_axis=1,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_interpretation_min_target_mean_sample_{n_samples}": PredictionInterpretationPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                MinMaxPointSampler(n_samples, experiment.dataset.target, "min"),
                shap_aggregation_strategy=ShapAggregationStrategy.MEAN,
                shap_aggregation_axis=1,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_interpretation_max_target_mean_sample_{n_samples}": PredictionInterpretationPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                MinMaxPointSampler(n_samples, experiment.dataset.target, "max"),
                shap_aggregation_strategy=ShapAggregationStrategy.MEAN,
                shap_aggregation_axis=1,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_interpretation_linear_target_mean_sample_{n_samples}": PredictionInterpretationPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                LinearPointSampler(n_samples, experiment.dataset.target),
                shap_aggregation_strategy=ShapAggregationStrategy.MEAN,
                shap_aggregation_axis=1,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_interpretation_percentile_random_target_mean_sample_{n_samples}": PredictionInterpretationPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                PercentilePointSampler(
                    experiment.dataset.target, 0.25, 0.75, RandomPointSampler(n_samples)
                ),
                shap_aggregation_strategy=ShapAggregationStrategy.MEAN,
                shap_aggregation_axis=1,
            ).build_pipeline(
                skip_cache=False
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_interpretation_point"


class ModelConsolidationPredictionLossSetExperimentDefinition(
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
            f"prediction_loss_set_random_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                RandomSetSampler(n_samples),
            ).build_pipeline(skip_cache=False, use_train=use_train),
            f"prediction_loss_set_min_target_var_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Variance(), "min"),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
            f"prediction_loss_set_max_target_var_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Variance(), "max"),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
            f"prediction_loss_set_min_target_median_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(
                    n_samples, TargetSelector(), Quantile(0.5), "min"
                ),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
            f"prediction_loss_set_max_target_median_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(
                    n_samples, TargetSelector(), Quantile(0.5), "max"
                ),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
            f"prediction_loss_set_min_target_entropy_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Entropy(), "min"),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
            f"prediction_loss_set_max_target_entropy_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Entropy(), "max"),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
            f"prediction_loss_set_uniform_target_var_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Variance()),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
            f"prediction_loss_set_uniform_target_median_{n_samples}": PredictionLossSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Quantile(0.5)),
            ).build_pipeline(
                skip_cache=False, use_train=use_train
            ),
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


class ModelConsolidationPredictionValueExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition
):
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)

        return {
            f"prediction_value_point_random_{n_samples}": PredictionValuePointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                RandomPointSampler(n_samples),
                client=client,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_value_point_min_target_{n_samples}": PredictionValuePointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                MinMaxPointSampler(n_samples, experiment.dataset.target, "min"),
                client=client,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_value_point_max_target_{n_samples}": PredictionValuePointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                MinMaxPointSampler(n_samples, experiment.dataset.target, "max"),
                client=client,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_value_point_linear_target_{n_samples}": PredictionValuePointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                LinearPointSampler(n_samples, experiment.dataset.target),
                client=client,
            ).build_pipeline(
                skip_cache=True
            ),
            f"prediction_value_point_percentile_random_target_{n_samples}": PredictionValuePointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                PercentilePointSampler(
                    experiment.dataset.target, 0.25, 0.75, RandomPointSampler(n_samples)
                ),
                client=client,
            ).build_pipeline(
                skip_cache=False
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_value_point"


class ModelConsolidationPredictionValueSetExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition
):
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)
        return {
            f"prediction_value_set_random_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                RandomSetSampler(n_samples),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_min_target_var_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Variance(), "min"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_max_target_var_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Variance(), "max"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_min_target_median_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(
                    n_samples, TargetSelector(), Quantile(0.5), "min"
                ),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_max_target_median_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(
                    n_samples, TargetSelector(), Quantile(0.5), "max"
                ),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_min_target_entropy_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Entropy(), "min"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_max_target_entropy_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                StatisticalSetSampler(n_samples, TargetSelector(), Entropy(), "max"),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_uniform_target_var_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Variance()),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_uniform_target_median_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Quantile(0.5)),
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_value_set_uniform_target_entropy_{n_samples}": PredictionValueSetPipelineBuilder(
                pipeline_factory,
                KMeans(),
                UniformSetSampler(n_samples, TargetSelector(), Entropy()),
            ).build_pipeline(
                skip_cache=False
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_value_set"


class ModelConsolidationPredictionLossExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition
):
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)

        target_col_name = experiment.dataset.target

        loss = experiment.dataset.model_entities()[0].loss

        return {
            f"prediction_loss_point_random_{n_samples}": PredictionLossPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                RandomPointSampler(n_samples),
                loss,
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_loss_point_min_target_{n_samples}": PredictionLossPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                MinMaxPointSampler(n_samples, target_col_name, "min"),
                loss,
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_loss_point_max_target_{n_samples}": PredictionLossPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                MinMaxPointSampler(n_samples, target_col_name, "max"),
                loss,
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_loss_point_linear_target_{n_samples}": PredictionLossPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                LinearPointSampler(n_samples, target_col_name),
                loss,
            ).build_pipeline(
                skip_cache=False
            ),
            f"prediction_loss_point_percentile_random_target_{n_samples}": PredictionLossPointPipelineBuilder(
                pipeline_factory,
                KMeans(),
                PercentilePointSampler(
                    target_col_name, 0.25, 0.75, RandomPointSampler(n_samples)
                ),
                loss,
            ).build_pipeline(
                skip_cache=False
            ),
        }

    @property
    def identifier(self) -> str:
        return "prediction_loss_point"


class ModelConsolidationCrossPerformanceExperimentDefinition(
    AbstractModelConsolidationExperimentDefinition
):
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        pipeline_factory = self.get_pipeline_factory(experiment, client)
        # loss = mean_absolute_error
        loss = experiment.dataset.model_entities()[0].loss
        # For the weather data we need to unpack the crps loss
        if experiment.identifier == "weather_probabilistic":
            loss = crps_unpacking
        return {
            "cross_performance": HierarchicalDistPipelineBuilder(
                pipeline_factory,
                CrossPerformanceDistance(
                    client=client, loss_function=loss, skip_cache=False
                ),
            ).build_pipeline(),
        }

    @property
    def identifier(self) -> str:
        return "cross_performance"


if __name__ == "__main__":
    experiments = [
        ModelConsolidationPredictionLossSetExperimentDefinition(),
        ModelConsolidationPredictionValueExperimentDefinition(),
        ModelConsolidationPredictionLossExperimentDefinition(),
        ModelConsolidationCrossPerformanceExperimentDefinition(),
        ModelConsolidationPredictionValueSetExperimentDefinition(),
        ModelConsolidationPredictionInterpretationPointExperimentDefinition(),
        ModelConsolidationPredictionInterpretationSetExperimentDefinition(),
    ]
    run_experiments("point/results/method_ranking", experiments)
