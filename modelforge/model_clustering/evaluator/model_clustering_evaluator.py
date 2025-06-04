from copy import deepcopy
from logging import Logger

import numpy as np
import pandas as pd
from distributed import Client
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from modelforge.model_clustering.consolidation.consolidation_strategy import (
    ConsolidationStrategy,
)
from modelforge.model_clustering.distances.cross_performance import (
    compute_loss_ignore_nan,
)
from modelforge.model_clustering.entity.clustered_model_dataset import (
    ClusteredModelDataSet,
)
from modelforge.model_clustering.entity.loss import LossFunction
from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.evaluator.model_clustering_scoring_calculator import (
    ModelClusteringScoringCalculator,
)
from modelforge.model_clustering.transformer.clustered_dataset_mapper import (
    ClusteredDataSetMapper,
)
from modelforge.shared.logger import logger_factory


class ModelClusteringEvaluator(BaseEstimator):
    """
    Mixin class for evaluating the set. Used for potential subclassing.
    """

    def __init__(
        self,
        model_dataset: ModelDataSet,
        consolidation_strategy: ConsolidationStrategy,
        client: Client,
        loss: LossFunction = None,
    ):
        self.clustering_: ClusteredModelDataSet | None = None
        self.scoring_ = None
        self.client = client
        self.model_dataset = model_dataset
        self.consolidation_strategy = consolidation_strategy
        self.logger = logger_factory(__name__)
        self.loss = loss
        self.clustering_pipeline_ref_: Pipeline | None = None

    def fit(self, cluster_dataframe: pd.DataFrame, _y=None):
        """
        Evaluate the clustering results. For each cluster, a cluster set is built by concatenating all training
        sets of the devices in the specific cluster and its prediction error is then measured on the test point of
        each device in the cluster. The mean of these losses is the cluster set score. For comparison,
        the mean prediction error of the devices of **their own** test set is also calculated. The mean of these
        losses is the baseline score.
        """
        clustered_dataset_mapper = ClusteredDataSetMapper(self.model_dataset)
        clustered_dataset = clustered_dataset_mapper.transform(cluster_dataframe)

        if self.loss is None:
            loss = self.model_dataset.model_entities()[0].loss
        else:
            loss = self.loss

        self.scoring_ = evaluate_clustering(
            clustered_dataset,
            self.consolidation_strategy,
            self.client,
            self.logger,
            loss,
        )
        self.clustering_ = clustered_dataset
        return self

    def transform(self, _x) -> (pd.DataFrame, ClusteredModelDataSet):
        """
        Transform the cluster dataframe to a clustered set dataset.
        """
        return self.scoring_, self.clustering_

    def predict(self, model_id: int | str):
        """
        Predict the cluster of the set.
        """
        check_is_fitted(self, ["clustering_", "scoring_"])
        return self.clustering_.predict(model_id)

    def score(self, _x, _y=None, **_kwargs):
        """
        Return the score of the clustering.
        """
        check_is_fitted(self, ["clustering_", "scoring_"])
        calculator = ModelClusteringScoringCalculator(scoring=self.scoring_)
        cluster_loss, cluster_var = calculator.cluster_scores()
        model_loss, model_var = calculator.model_scores()
        return {
            "cluster_loss": cluster_loss,
            "cluster_var": cluster_var,
            "model_loss": model_loss,
            "model_var": model_var,
            "num_clusters": self.clustering_.get_num_clusters(),
            "scoring": self.scoring_.copy(),
            "clustering": self.clustering_,
            "clustering_pipeline_params": self.get_pipeline_params(),
        }

    def set_clustering_pipeline_ref(self, pipeline: Pipeline):
        self.clustering_pipeline_ref_ = pipeline
        return self

    def get_pipeline_params(self):
        if self.clustering_pipeline_ref_ is None:
            return {}
        return self.clustering_pipeline_ref_.get_params()


def evaluate_clustering(
    dataset: ClusteredModelDataSet,
    consolidation_strategy: ConsolidationStrategy,
    client: Client,
    logger: Logger,
    loss: LossFunction,
) -> pd.DataFrame:
    """
    Evaluates the clustering results. For each cluster, a cluster set is built by concatenating all training
    sets of the devices in the specific cluster and its prediction error is then measured on the test point of
    each device in the cluster. The mean of these losses is the cluster set score. For comparison,
    the mean prediction error of the devices of **their own** test set is also calculated. The mean of these
    losses is the baseline score.

    Parameters
    ----------
    dataset : ClusteredModelDataSet
        The clustered dataset.
    consolidation_strategy : ConsolidationStrategy
        The consolidation strategy to use.
    client : Client
        The client.
    logger : Logger
        The logger.
    loss : LossFunction
        The loss function.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the cluster set score, the baseline score and the cluster size.
    """

    def process_cluster(cluster_number: int) -> dict:
        cluster = dataset.get_cluster(cluster_number)
        assert len(cluster.model_entity_ids()) > 0, "Cluster is empty"

        if len(cluster.model_entity_ids()) == 1:
            # If there is only one set in the cluster, we can just return a copy the set as the cluster set
            # As they should behave the same
            return {
                "pipeline": deepcopy(cluster.model_entities()[0].pipeline),
                "cluster_id": cluster_number,
            }

        pipeline_clone = consolidation_strategy.fit_transform(cluster)

        return {
            "pipeline": pipeline_clone,
            "cluster_id": cluster_number,
        }

    def measure_entity_in_cluster(
        cluster_model_dict: dict, entity: ModelEntity, loss: LossFunction
    ) -> tuple[float, float]:
        cluster_model = cluster_model_dict["pipeline"]
        y_pred = cluster_model.predict(entity.test_x)

        cluster_component_loss = compute_loss_ignore_nan(entity.test_y, y_pred, loss)
        model_component_loss = entity.model_test_loss()

        return cluster_component_loss, model_component_loss

    def aggregate_cluster_mae(losses: list[tuple[float, float]], cluster_size: int):
        model_losses = np.zeros(cluster_size)
        cluster_losses = np.zeros(cluster_size)
        for i, (cluster_loss, model_loss) in enumerate(losses):
            model_losses[i] = model_loss
            cluster_losses[i] = cluster_loss

        # Remove NaNs. Some models might not have been able to predict anything for some reason
        # e.g. due to test sets being too short for windowing models or similar
        model_losses = model_losses[~np.isnan(model_losses)]
        cluster_losses = cluster_losses[~np.isnan(cluster_losses)]

        if len(model_losses) != cluster_size:
            logger.warning(
                f"Cluster size mismatch: {cluster_size} != {len(model_losses)}"
            )
        if len(cluster_losses) != cluster_size:
            logger.warning(
                f"Cluster size mismatch: {cluster_size} != {len(cluster_losses)}"
            )

        return (
            np.mean(cluster_losses),
            np.mean(model_losses),
            cluster_losses,
            model_losses,
            cluster_size,
        )

    scores = []

    for cluster_id in dataset.get_clusters():
        cluster_model_dict = client.submit(process_cluster, cluster_id)

        cluster = dataset.get_cluster(cluster_id)
        components = []
        for entity_id in cluster.model_entity_ids():
            entity = cluster.model_entity_by_id(entity_id)
            component = client.submit(
                measure_entity_in_cluster, cluster_model_dict, entity, loss
            )
            components.append(component)

        cluster_scores = client.submit(aggregate_cluster_mae, components, cluster.size)
        scores.append(cluster_scores)

    cluster_scores = client.gather(scores)
    return pd.DataFrame(
        cluster_scores,
        columns=[
            "cluster_loss",
            "model_loss",
            "cluster_losses",
            "model_losses",
            "cluster_size",
        ],
    )
