from typing import Callable

from distributed import Client
from joblib import Memory
from sklearn.metrics import mean_absolute_error

from modelforge.model_clustering.distances.memorized_distance import MemorizedDistance
from modelforge.model_clustering.entity.loss import compute_loss_ignore_nan
from modelforge.model_clustering.entity.model_entity import ModelEntity


def compute_loss(
    a: ModelEntity, b: ModelEntity, loss_function: Callable, train=False
) -> float:
    y_pred = a.predict(b.train_x if train else b.test_x)
    y_true = b.train_y if train else b.test_y
    loss = compute_loss_ignore_nan(y_true, y_pred, loss_function)

    return loss


class CrossPerformanceDistance(MemorizedDistance):
    def __init__(
        self,
        factor: float = 1 / 2,
        loss_function=mean_absolute_error,
        memory: Memory = None,
        client: Client = None,
        skip_cache: bool = False,
    ):
        super().__init__(
            distance_measure=cross_performance_factory(
                factor=factor, loss_function=loss_function
            ),
            memory=memory,
            client=client,
            skip_cache=skip_cache,
        )
        self.factor = factor
        self.loss_function = loss_function

    def distance_function_str(self) -> str:
        return (
            "CrossPerformanceDistance_"
            + str(self.factor)
            + "_"
            + self.loss_function.__name__
        )


def cross_performance_factory(
    factor: float, loss_function: Callable
) -> Callable[[ModelEntity, ModelEntity], float]:
    def cross_performance_distance(a: ModelEntity, b: ModelEntity) -> float:
        """
        Calculate the distance between two models based on their cross-performance on each other's test set.

        @param:
            a (ModelEntity): The first set entity.
            b (ModelEntity): The second set entity.
            loss_function (callable, optional): The loss function to calculate the prediction error.
                Defaults to mean_absolute_error.

        Returns:
            float: The distance between the two models.
        """
        loss_a_on_b = compute_loss(a, b, loss_function)
        loss_b_on_a = compute_loss(b, a, loss_function)
        assert loss_a_on_b >= 0, f"loss_a_on_b={loss_a_on_b}"
        assert loss_b_on_a >= 0, f"loss_b_on_a={loss_b_on_a}"
        return factor * (loss_a_on_b + loss_b_on_a)

    return cross_performance_distance
