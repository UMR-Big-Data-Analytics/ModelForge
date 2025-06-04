import pandas as pd

from modelforge.model_clustering.entity.model_entity import ModelEntity


class WeatherModelProbEntity(ModelEntity):

    def model_test_loss(self) -> float:
        return self.metadata["model_metrics"]["test_score"]

    def predict(self, x: pd.DataFrame, *_args, **_kwargs):
        pred = self.pipeline.predict(x)
        return pd.DataFrame(
            {
                "mean": pred[0],
                "std": pred[1],
            }
        )
