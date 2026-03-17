from dataclasses import dataclass

from src.backend.domain.ports.prediction_gateway import PredictionGateway
from src.backend.schemas import PredictRequest, PredictionResponse


@dataclass(frozen=True)
class PredictUseCase:
    gateway: PredictionGateway

    def execute(self, request: PredictRequest) -> PredictionResponse:
        return self.gateway.predict(request)

