from dataclasses import dataclass
from functools import lru_cache

from src.backend.application.use_cases.get_feature_preview import GetFeaturePreviewUseCase
from src.backend.application.use_cases.get_market_candles import GetMarketCandlesUseCase
from src.backend.application.use_cases.list_models import ListModelsUseCase
from src.backend.application.use_cases.predict import PredictUseCase
from src.backend.application.use_cases.train_model import TrainModelUseCase
from src.backend.infrastructure.legacy.prediction_gateway import LegacyPredictionGateway
from src.backend.shared.settings import get_app_settings, to_legacy_settings


@dataclass(frozen=True)
class UseCaseContainer:
    get_feature_preview: GetFeaturePreviewUseCase
    get_market_candles: GetMarketCandlesUseCase
    train_model: TrainModelUseCase
    predict: PredictUseCase
    list_models: ListModelsUseCase


@lru_cache(maxsize=1)
def get_use_case_container() -> UseCaseContainer:
    app_settings = get_app_settings()
    gateway = LegacyPredictionGateway.from_settings(
        settings=to_legacy_settings(app_settings),
    )
    return UseCaseContainer(
        get_feature_preview=GetFeaturePreviewUseCase(gateway=gateway),
        get_market_candles=GetMarketCandlesUseCase(gateway=gateway),
        train_model=TrainModelUseCase(gateway=gateway),
        predict=PredictUseCase(gateway=gateway),
        list_models=ListModelsUseCase(gateway=gateway),
    )


def get_container() -> UseCaseContainer:
    return get_use_case_container()

