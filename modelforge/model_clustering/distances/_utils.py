import numpy as np
import pandas as pd

from modelforge.model_clustering.entity.model_entity import ModelEntity


def as_numpy(criterion: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(criterion, pd.Series) or isinstance(criterion, pd.DataFrame):
        return criterion.to_numpy()
    return criterion


def model_entities_as_numpy(
    criterion_selector, a: ModelEntity, b: ModelEntity
) -> tuple[np.ndarray, np.ndarray]:
    criterion_a = criterion_selector.get_criterion(a)
    criterion_b = criterion_selector.get_criterion(b)
    return as_numpy(criterion_a), as_numpy(criterion_b)
