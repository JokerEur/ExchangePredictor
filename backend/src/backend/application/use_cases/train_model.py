from dataclasses import dataclass

from src.backend.domain.ports.prediction_gateway import PredictionGateway
from src.backend.schemas import TrainRequest, TrainResponse


@dataclass(frozen=True)
class TrainModelUseCase:
    gateway: PredictionGateway

    def execute(self, request: TrainRequest) -> TrainResponse:
        return self.gateway.train_model(request)

