import json
import zlib
from datetime import datetime

from distributed import Client
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from modelforge.model_clustering.cluster.grid_search import (
    GridSearch,
    ModelConsolidationResultSet,
)
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.shared.logger import logger_factory


class PipelineFitException(Exception):
    def __init__(self, exception: Exception, pipeline_name: str):
        super().__init__(f"Pipeline {pipeline_name} failed to fit: {str(exception)}")
        self.pipeline_name = pipeline_name
        self.base_exception = exception


def eval_pipeline(
    params: dict,
    pipeline: Pipeline,
    dataset: ModelDataSet,
    client: Client,
    _pipeline_hash: int,
    _model_dataset_hash: int,
    _params_hash: int,
):
    search = GridSearch(
        pipeline=pipeline,
        param_grid=params,
        client=client,
    )
    search.fit(dataset)
    return search.params()


class BasePipelineSearch:
    def __init__(
        self,
        pipelines: dict[str, Pipeline],
        client: Client,
        params: dict,
        memory: Memory = None,
        skip_cache: bool = True,
    ):
        self.results_: dict[str, ModelConsolidationResultSet] = {}
        self.logger = logger_factory(__name__)
        self.pipelines = pipelines
        self.client = client
        self.memory = memory
        self.params = params
        self.skip_cache = skip_cache
        if memory is None:
            self.memory = Memory("./.cache", verbose=0)

    def fit(self, dataset: ModelDataSet):
        results = {}

        for i, (name, pipeline) in enumerate(self.pipelines.items()):
            try:
                self.logger.info(
                    f"Start pipeline {i + 1}/{len(self.pipelines)}: {name}"
                )
                pipeline_hash = zlib.adler32(pipeline.__str__().encode())
                model_dataset_hash = dataset.__hash__()
                params_hash = zlib.adler32(json.dumps(self.params).encode())
                self.logger.debug(f"Pipeline hash: {pipeline_hash}")
                self.logger.debug(f"Model dataset hash: {model_dataset_hash}")

                if self.skip_cache:
                    results[name] = eval_pipeline(
                        self.params,
                        pipeline,
                        dataset,
                        self.client,
                        pipeline_hash,
                        model_dataset_hash,
                        params_hash,
                    )
                else:
                    results[name] = self.memory.cache(
                        eval_pipeline,
                        ignore=["params", "pipeline", "dataset", "client"],
                    )(
                        self.params,
                        pipeline,
                        dataset,
                        self.client,
                        pipeline_hash,
                        model_dataset_hash,
                        params_hash,
                    )
            except Exception as e:
                raise PipelineFitException(e, name)

        self.results_ = results
        return results

    def persist_results(self, directory: str = "."):
        check_is_fitted(self, "results_")
        # current date string
        date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        for name, result in self.results_.items():
            result.to_disk(f"{directory}/{name}_{date}")
