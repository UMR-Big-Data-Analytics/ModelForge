from typing import List

import numpy as np
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
from modelforge.shared.logger import logger_factory


class UniformSetSampler(SetSampler):
    """
    Samples models based on a uniform distribution given by a sorted list of a statistical measure.
    """

    def __init__(
        self,
        num_samples: int,
        criterion_selector: CriterionSelector,
        statistical_measure: Measure,
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
        """
        super().__init__(num_samples)
        self.criterion_selector = criterion_selector
        self.statistical_measure = statistical_measure
        self.logger = logger_factory(__name__)

    def get_sample_candidates(self, model_data_set: ModelDataSet) -> List[ModelEntity]:
        ids = model_data_set.model_entity_ids()
        statistic = []
        for model_entity_id in ids:
            model_entity = model_data_set.model_entity_by_id(model_entity_id)
            criterion = self.criterion_selector.get_criterion(model_entity)
            calculated_measure = self.statistical_measure.calculate(criterion)
            statistic.append(calculated_measure)
        df = pd.DataFrame({"id": ids, "statistic": statistic})
        df = df.sort_values("statistic")

        sample_indices = np.linspace(0, len(df) - 1, self.num_samples, dtype=int)
        # Check if there are duplicate indices
        if len(set(sample_indices)) != len(sample_indices):
            self.logger.warning(
                "There are duplicate indices in the sample indices. This might lead to duplicate samples."
            )
        samples_entity_ids = df.iloc[sample_indices]["id"].tolist()
        return [
            model_data_set.model_entity_by_id(model_entity_id)
            for model_entity_id in samples_entity_ids
        ]

    def __str__(self):
        return f"UniformModelSampler(num_samples={self.num_samples}, criterion_selector={self.criterion_selector}, statistical_measure={self.statistical_measure})"

    def __repr__(self, nchars=None):
        return self.__str__()
