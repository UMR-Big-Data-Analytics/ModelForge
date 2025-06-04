import pandas as pd

from .measure import Measure


class Quantile(Measure):
    def __init__(self, q: float):
        self.q = q

    def calculate(self, x: pd.Series) -> float:
        return x.quantile(self.q)

    def __str__(self):
        return f"Quantile({self.q})"
