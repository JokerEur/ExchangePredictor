from dataclasses import dataclass

from src.backend.domain.ports.prediction_gateway import PredictionGateway
from src.backend.schemas import FeaturePreviewRequest, FeaturePreviewResponse


@dataclass(frozen=True)
class GetFeaturePreviewUseCase:
    gateway: PredictionGateway

    def execute(self, request: FeaturePreviewRequest) -> FeaturePreviewResponse:
        return self.gateway.get_feature_preview(request)

