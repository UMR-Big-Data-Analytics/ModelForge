from modelforge.model_clustering.entity.model_entity import ModelEntity


class AnomalyModelEntity(ModelEntity):
    @property
    def loss_minimize(self) -> bool:
        return False
