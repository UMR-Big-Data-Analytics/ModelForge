import pandas as pd
from scipy.stats import entropy

from modelforge.model_clustering.transformer.sampler.set.measure.measure import (
    Measure,
)


class Entropy(Measure):
    def calculate(self, x: pd.Series) -> float:
        counts = x.groupby(x).count()
        probs = counts / counts.sum()
        return entropy(probs)

    def __str__(self):
        return "Entropy()"
