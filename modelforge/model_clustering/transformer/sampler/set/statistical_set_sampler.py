from typing import List

import pandas as pd

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.sampler.set.criterion_selector.criterion_selector import (
    CriterionSelector,
)
from modelforge.model_clustering.transformer.sampler.set.measure.measure import (
    Measure,
)
from modelforge.model_clustering.transformer.sampler.set.set_sampler import (
    SetSampler,
)


class StatisticalSetSampler(SetSampler):
    """
    Samples models based on a statistical measure.
    """

    def __init__(
        self,
        num_samples: int,
        criterion_selector: CriterionSelector,
        statistical_measure: Measure,
        objective: str = "max",
    ):
        """
        Parameters
        ----------
        num_samples : int
            The number of samples to take.
        criterion_selector : CriterionSelector
            The criterion to sample on.
        statistical_measure : Measure
            The statistical measure to use.
        objective : 'min' | 'max'
            The objective to optimize. Either 'min' or 'max' for the num_samples with minimal or maximal variance.
        """
        super().__init__(num_samples)
        if objective not in ["min", "max"]:
            raise ValueError(f"Objective {objective} not in ['min', 'max']")
        self.criterion_selector = criterion_selector
        self.statistical_measure = statistical_measure
        self.objective = objective

    def get_sample_candidates(self, x: ModelDataSet) -> List[ModelEntity]:
        """
        Samples models based on a statistical measure.

        Parameters
        ----------
        x : ModelDataSet
            The dataset to sample from.

        Returns
        -------
        List[ModelEntity]
            The sampled models.
        """
        df = self.calculate_measure(x)
        df = df.sort_values("statistic", ascending=self.objective == "min")

        samples_entity_ids = df.iloc[: self.num_samples]["id"].tolist()
        return [
            x.model_entity_by_id(model_entity_id)
            for model_entity_id in samples_entity_ids
        ]

    def calculate_measure(self, model_data_set: ModelDataSet) -> pd.DataFrame:
        """
        Calculate the statistical measure for each set in the dataset.

        Parameters
        ----------
        model_data_set : ModelDataSet

        Returns
        -------
        pd.DataFrame
            A dataframe with the set entity id and the calculated measure.
        """
        ids = model_data_set.model_entity_ids()
        statistic = []
        for model_entity_id in ids:
            model_entity = model_data_set.model_entity_by_id(model_entity_id)
            criterion = self.criterion_selector.get_criterion(model_entity)
            calculated_measure = self.statistical_measure.calculate(criterion)
            statistic.append(calculated_measure)
        return pd.DataFrame({"id": ids, "statistic": statistic})

    def __repr__(self, n_max_chars=700):
        return self.__str__()

    def __str__(self):
        return f"StatisticalModelSampler(num_samples={self.num_samples}, criterion_selector={self.criterion_selector}, statistical_measure={self.statistical_measure}, objective={self.objective})"
