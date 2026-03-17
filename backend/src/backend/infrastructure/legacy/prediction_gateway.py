from src.backend.config import Settings
from src.backend.schemas import (
    FeaturePreviewRequest,
    FeaturePreviewResponse,
    MarketCandlesRequest,
    MarketCandlesResponse,
    ModelRegistryResponse,
    PredictRequest,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)
from src.backend.service import PredictionService


class LegacyPredictionGateway:
    def __init__(self, service: PredictionService) -> None:
        self._service = service

    @classmethod
    def from_settings(cls, settings: Settings) -> "LegacyPredictionGateway":
        return cls(service=PredictionService(settings=settings))

    def get_feature_preview(self, request: FeaturePreviewRequest) -> FeaturePreviewResponse:
        return self._service.get_feature_preview(request)
    def get_market_candles(self, request: MarketCandlesRequest) -> MarketCandlesResponse:
        return self._service.get_market_candles(request)

    def train_model(self, request: TrainRequest) -> TrainResponse:
        return self._service.train_model(request)

    def predict(self, request: PredictRequest) -> PredictionResponse:
        return self._service.predict(request)

    def list_models(self) -> ModelRegistryResponse:
        return self._service.list_models()

