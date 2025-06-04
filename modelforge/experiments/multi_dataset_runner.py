import abc
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from logging import Logger

from distributed import Client
from sklearn.pipeline import Pipeline

from modelforge.experiments.pipeline_builders.pipeline_factory import PipelineFactory
from modelforge.model_clustering.cluster.grid_search import ModelConsolidationResultSet
from modelforge.model_clustering.consolidation.consolidation_strategy import (
    ConsolidationStrategy,
)
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.model_selection.base_pipeline_search import (
    PipelineFitException,
)
from modelforge.model_clustering.model_selection.grid_pipeline_search import (
    GridPipelineSearch,
)


@dataclass
class ModelConsolidationDatasetDefinition:
    identifier: str
    dataset: ModelDataSet
    factors: list[float]
    consolidation_strategy: ConsolidationStrategy


class AbstractModelConsolidationExperimentDefinition(abc.ABC):
    @abstractmethod
    def build(
        self, experiment: ModelConsolidationDatasetDefinition, client: Client
    ) -> dict[str, Pipeline]:
        raise NotImplementedError(
            "Method build not implemented in class ModelConsolidationExperimentDefinition"
        )

    @property
    @abstractmethod
    def identifier(self) -> str:
        raise NotImplementedError(
            "Property identifier not implemented in class ModelConsolidationExperimentDefinition"
        )

    @staticmethod
    def get_pipeline_factory(
        experiment: ModelConsolidationDatasetDefinition, client: Client
    ):
        return PipelineFactory(
            experiment.dataset,
            client,
            experiment.consolidation_strategy,
        )


@dataclass
class MultiDatasetRunner:
    basepath: str
    client: Client
    dataset_definitions: list[ModelConsolidationDatasetDefinition]
    experiment_definitions: list[AbstractModelConsolidationExperimentDefinition]
    logger: Logger = None
    persist_results: bool = True

    def run(self):
        date_str = datetime.now().strftime("%Y-%m-%d-%H:%M")
        self.logger.info(
            f"Starting {len(self.experiment_definitions)} experiments on {len(self.dataset_definitions)} datasets"
        )
        for experiment in self.experiment_definitions:
            experiment_id = experiment.identifier
            for dataset_definition in self.dataset_definitions:
                self.logger.info(
                    f"Running experiment {experiment_id} on dataset {dataset_definition.identifier}"
                )
                start_time = datetime.now()
                consolidation_results = self.run_experiment(
                    experiment, dataset_definition
                )
                duration_min = (datetime.now() - start_time).total_seconds() / 60
                self.logger.info(
                    f"Experiment {experiment_id} on dataset {dataset_definition.identifier} took {duration_min}"
                )
                if self.persist_results:
                    self.save_results(
                        consolidation_results,
                        experiment_id,
                        dataset_definition.identifier,
                        date_str,
                    )

    def run_experiment(
        self,
        experiment: AbstractModelConsolidationExperimentDefinition,
        dataset_definition: ModelConsolidationDatasetDefinition,
    ):
        try:
            pipelines = experiment.build(dataset_definition, self.client)
            grid_search = GridPipelineSearch(
                pipelines=pipelines,
                client=self.client,
                factors=dataset_definition.factors,
            )
            grid_search.fit(dataset_definition.dataset)
            return grid_search.results_
        except PipelineFitException as e:
            self.logger.error(
                f"Error running experiment {experiment.identifier} with pipeline {e.pipeline_name} on dataset {dataset_definition.identifier}: {e}"
            )
            raise e

    def save_results(
        self,
        results: dict[str, ModelConsolidationResultSet],
        experiment_id: str,
        dataset_id: str,
        date_str: str,
    ):
        for pipeline_name, result in results.items():
            result.to_disk(
                f"{self.basepath}/{dataset_id}/{experiment_id}_{date_str}/{pipeline_name}"
            )
