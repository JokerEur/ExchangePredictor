from typing import Protocol

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


class PredictionGateway(Protocol):
    def get_feature_preview(self, request: FeaturePreviewRequest) -> FeaturePreviewResponse:
        ...
    def get_market_candles(self, request: MarketCandlesRequest) -> MarketCandlesResponse:
        ...

    def train_model(self, request: TrainRequest) -> TrainResponse:
        ...

    def predict(self, request: PredictRequest) -> PredictionResponse:
        ...

    def list_models(self) -> ModelRegistryResponse:
        ...

