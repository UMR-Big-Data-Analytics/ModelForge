from abc import abstractmethod

import pandas as pd


class Measure:
    @abstractmethod
    def calculate(self, x: pd.Series) -> float:
        raise NotImplementedError("Measure should implement calculate()")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Measure should implement __str__()")

    def __repr__(self):
        return self.__str__()
