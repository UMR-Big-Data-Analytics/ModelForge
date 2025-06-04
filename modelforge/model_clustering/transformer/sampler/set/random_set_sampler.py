import random
from random import sample
from typing import List

from modelforge.model_clustering.entity.model_dataset import ModelDataSet
from modelforge.model_clustering.entity.model_entity import ModelEntity
from modelforge.model_clustering.transformer.sampler.set.set_sampler import (
    SetSampler,
)


class RandomSetSampler(SetSampler):
    """
    Randomly samples models from a dataset.
    """

    def get_sample_candidates(self, x: ModelDataSet) -> List[ModelEntity]:
        """
        Randomly samples models from a dataset.

        Parameters
        ----------
        x : ModelDataSet
            The dataset to sample from.

        Returns
        -------
        List[ModelEntity]
            The sampled models.
        """
        models_entities = sorted(x.model_entities(), key=lambda entity: entity.id)
        random.seed(42)
        return sample(models_entities, self.num_samples)

    def __str__(self):
        return f"RandomModelSampler(num_samples={self.num_samples})"

    def __repr__(self, nchars=None):
        return self.__str__()
