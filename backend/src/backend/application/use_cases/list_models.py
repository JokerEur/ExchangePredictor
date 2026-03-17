from dataclasses import dataclass

from src.backend.domain.ports.prediction_gateway import PredictionGateway
from src.backend.schemas import ModelRegistryResponse


@dataclass(frozen=True)
class ListModelsUseCase:
    gateway: PredictionGateway

    def execute(self) -> ModelRegistryResponse:
        return self.gateway.list_models()

