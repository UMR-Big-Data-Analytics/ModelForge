import pandas as pd

from .measure import Measure


class Variance(Measure):
    def calculate(self, x: pd.Series) -> float:
        return x.var()

    def __str__(self):
        return "Variance()"
