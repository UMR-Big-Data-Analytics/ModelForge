from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline


class PipelineBuilder(ABC):
    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        pass
