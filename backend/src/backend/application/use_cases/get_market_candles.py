from dataclasses import dataclass

from src.backend.domain.ports.prediction_gateway import PredictionGateway
from src.backend.schemas import MarketCandlesRequest, MarketCandlesResponse


@dataclass(frozen=True)
class GetMarketCandlesUseCase:
    gateway: PredictionGateway

    def execute(self, request: MarketCandlesRequest) -> MarketCandlesResponse:
        return self.gateway.get_market_candles(request)

